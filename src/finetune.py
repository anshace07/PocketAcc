"""
finetune.py - STEP 4b: QLoRA fine-tuning of Qwen2-VL-2B on statement pages.

What happens:
  1. load the pretrained model in 4-bit (frozen)            -> fits a 6 GB laptop GPU
  2. attach small LoRA matrices to the attention layers     -> only ~0.3% of weights train
  3. for each page: input = image + instruction, label = JSON of the rows on that page
     (loss only on the JSON tokens - prompt/image tokens are masked with -100)
  4. AdamW, cosine learning-rate schedule, gradient accumulation, gradient checkpointing
  5. validation loss every --eval-every steps; the best adapter is kept

Usage (GPU):
    python src/finetune.py --data data/vlm --out runs/lora_r8 --rank 8 --epochs 1
Low memory (6 GB laptop):
    python src/finetune.py --data data/vlm --out runs/lora_r8 --max-pixels 900000 --max-len 4096
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

from vlm import DEFAULT_MODEL, build_inputs, compute_dtype, load_model, load_processor


class PageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, limit=None):
        self.dir = Path(data_dir)
        self.rows = [json.loads(l) for l in open(self.dir / f"{split}.jsonl")]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return Image.open(self.dir / r["image"]).convert("RGB"), r["prompt"], r["target"]


def add_lora(model, rank, alpha=None, dropout=0.05, mlp=False, four_bit=True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if four_bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"] + (["gate_proj", "up_proj", "down_proj"] if mlp else [])
    cfg = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=dropout, target_modules=targets,
                     task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    return model


@torch.no_grad()
def val_loss(model, processor, ds, max_len, n=20):
    model.eval()
    losses = []
    for i in range(min(n, len(ds))):
        img, prompt, target = ds[i]
        batch = build_inputs(processor, img, prompt, target)
        if batch["input_ids"].shape[1] > max_len:
            continue
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast(model.device.type, dtype=compute_dtype(), enabled=model.device.type == "cuda"):
            losses.append(model(**batch).loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(model, processor, train_ds, val_ds, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.max_steps or math.ceil(len(train_ds) * args.epochs / args.grad_accum)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    use_scaler = model.device.type == "cuda" and compute_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    log = open(out / "training_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "train_loss", "val_loss", "lr", "seconds", "max_mem_gb"])
    order = list(range(len(train_ds)))
    step, micro, best, run_loss, t0 = 0, 0, float("inf"), [], time.time()
    model.train()
    for epoch in range(math.ceil(args.epochs)):
        random.Random(args.seed + epoch).shuffle(order)
        for i in order:
            img, prompt, target = train_ds[i]
            batch = build_inputs(processor, img, prompt, target)
            if batch["input_ids"].shape[1] > args.max_len:            # skip pages that would not fit
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast(model.device.type, dtype=compute_dtype(), enabled=model.device.type == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            run_loss.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % args.log_every == 0 or step == total:
                vl = val_loss(model, processor, val_ds, args.max_len, args.val_samples) if (
                    step % args.eval_every == 0 or step == total) else ""
                mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                tl = sum(run_loss) / len(run_loss)
                w.writerow([step, round(tl, 4), round(vl, 4) if vl != "" else "", sched.get_last_lr()[0],
                            round(time.time() - t0), round(mem, 2)])
                log.flush()
                print(f"step {step}/{total}  train_loss {tl:.4f}  val_loss {vl if vl == '' else round(vl, 4)}  "
                      f"mem {mem:.2f} GB  {time.time() - t0:.0f}s", flush=True)
                run_loss = []
                if vl != "" and vl < best:
                    best = vl
                    model.save_pretrained(out / "best_adapter")
            if step >= total:
                break
        if step >= total:
            break
    model.save_pretrained(out / "final_adapter")
    json.dump(vars(args) | {"best_val_loss": best, "steps": step}, open(out / "run_config.json", "w"), indent=1)
    log.close()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vlm")
    ap.add_argument("--out", default="runs/lora_r8")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--mlp", action="store_true", help="also put LoRA on the MLP layers")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=6144, help="skip examples longer than this many tokens")
    ap.add_argument("--max-pixels", type=int, default=1_400_000)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--val-samples", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="use only N training pages (quick test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    processor = load_processor(args.model, args.max_pixels)
    model = load_model(args.model, four_bit=not args.no_4bit)
    model = add_lora(model, args.rank, mlp=args.mlp, four_bit=not args.no_4bit)
    model.print_trainable_parameters()
    train_ds = PageDataset(args.data, "train", args.limit or None)
    val_ds = PageDataset(args.data, "val")
    print(f"train pages {len(train_ds)}  val pages {len(val_ds)}")
    best = train(model, processor, train_ds, val_ds, args)
    print("done. best val loss", best)


if __name__ == "__main__":
    main()
