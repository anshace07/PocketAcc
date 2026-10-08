"""
vram_test.py - run this FIRST on your GPU. Answers: "will fine-tuning fit on my card?"

It loads Qwen2-VL-2B in 4-bit, attaches LoRA, and measures peak GPU memory for
  (a) reading one page (inference)
  (b) one training step on the LONGEST training page (the worst case)
at the image size you choose. Then it prints a recommendation.

Run:   python scripts/vram_test.py --data data/vlm
Try smaller images if it fails:  --max-pixels 900000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finetune import add_lora  # noqa: E402
from vlm import DEFAULT_MODEL, build_inputs, compute_dtype, load_model, load_processor  # noqa: E402


def gb(x):
    return round(x / 1e9, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vlm")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-pixels", type=int, default=1_400_000)
    ap.add_argument("--rank", type=int, default=8)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "No CUDA GPU visible to PyTorch - install the CUDA build of torch."
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {p.name} | total {gb(total)} GB | free right now {gb(free)} GB | bf16: {torch.cuda.is_bf16_supported()}")

    proc = load_processor(a.model, a.max_pixels)
    rows = [json.loads(l) for l in open(Path(a.data) / "train.jsonl")]
    # longest target = most rows on a page = worst case for memory
    worst = max(rows, key=lambda r: len(r["target"]))
    img = Image.open(Path(a.data) / worst["image"]).convert("RGB")
    batch = build_inputs(proc, img, worst["prompt"], worst["target"])
    n_tok = batch["input_ids"].shape[1]
    n_img = int((batch["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    print(f"worst-case page: {worst['id']} | image {img.size} | {n_img} image tokens | {n_tok} total tokens")

    torch.cuda.reset_peak_memory_stats()
    model = load_model(a.model, four_bit=True)
    print(f"model loaded in 4-bit: {gb(torch.cuda.memory_allocated())} GB")

    # (a) inference on the same page
    torch.cuda.reset_peak_memory_stats()
    inf = build_inputs(proc, img, worst["prompt"]).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inf, max_new_tokens=128, do_sample=False)
    secs = time.time() - t0
    new = out.shape[1] - inf["input_ids"].shape[1]
    print(f"(a) inference peak: {gb(torch.cuda.max_memory_allocated())} GB | {new / secs:.1f} tokens/s "
          f"(a full page needs ~{len(worst['target']) // 3} tokens -> ~{len(worst['target']) // 3 / max(new / secs, 1e-6) / 60:.1f} min)")

    # (b) one LoRA training step
    model = add_lora(model, a.rank, four_bit=True)
    model.train()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    try:
        b = {k: v.to("cuda") for k, v in batch.items()}
        t0 = time.time()
        with torch.autocast("cuda", dtype=compute_dtype()):
            loss = model(**b).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        peak = torch.cuda.max_memory_allocated()
        print(f"(b) training step peak: {gb(peak)} GB | loss {loss.item():.3f} | {time.time() - t0:.1f}s per page")
        headroom = (total - peak) / total
        print("\nRESULT:", "FITS with headroom - go ahead and fine-tune." if headroom > 0.15 else
              "FITS but tight - close other apps, or use --max-pixels 900000.")
    except torch.cuda.OutOfMemoryError:
        print("\nRESULT: OUT OF MEMORY on the worst-case page.\n"
              "Try in order: --max-pixels 900000  ->  --max-pixels 700000  ->  train on Kaggle (free T4 16 GB).")


if __name__ == "__main__":
    main()
