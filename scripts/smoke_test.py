"""
smoke_test.py - prove the training/inference code works BEFORE spending GPU hours.

Builds a TINY randomly-initialised Qwen2-VL (same architecture, ~1M parameters) with a
small tokenizer, then runs the real code paths on CPU:
  1. build_inputs(): the prompt + image tokens are masked, only the JSON answer is trained
  2. finetune.train(): a few optimisation steps with LoRA, logging and adapter saving
  3. over-fitting check: loss on one page must fall a lot -> gradients reach the LoRA weights
  4. Extractor.read_image(): generation + JSON parsing path runs end to end
  5. evaluate.score(): perfect predictions score 100%

Run:  python scripts/smoke_test.py data/vlm
"""
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders  # noqa: E402
from transformers import Qwen2TokenizerFast, Qwen2VLConfig, Qwen2VLForConditionalGeneration  # noqa: E402
from transformers import Qwen2VLImageProcessor, Qwen2VLProcessor  # noqa: E402

import finetune  # noqa: E402
from extract import Extractor  # noqa: E402
from vlm import build_inputs  # noqa: E402

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
           "<|image_pad|>", "<|video_pad|>"]
TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}"
            "{{ message['content'] }}{% else %}{% for c in message['content'] %}{% if c['type'] == 'image' %}"
            "<|vision_start|><|image_pad|><|vision_end|>{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
            "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def tiny_processor(texts):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = Qwen2TokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.add_special_tokens({"additional_special_tokens": SPECIAL[1:]})
    ip = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=160_000)
    return Qwen2VLProcessor(image_processor=ip, tokenizer=fast, chat_template=TEMPLATE)


def tiny_model(proc):
    t = proc.tokenizer
    cfg = Qwen2VLConfig(
        vocab_size=len(t), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=32768, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_config=dict(depth=2, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, in_channels=3),
        image_token_id=t.convert_tokens_to_ids("<|image_pad|>"), video_token_id=t.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=t.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=t.convert_tokens_to_ids("<|vision_end|>"),
        bos_token_id=t.convert_tokens_to_ids("<|endoftext|>"), eos_token_id=t.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=t.convert_tokens_to_ids("<|endoftext|>"), tie_word_embeddings=True)
    return Qwen2VLForConditionalGeneration(cfg)


def main(data_dir):
    torch.manual_seed(0)
    data = Path(data_dir)
    train_rows = [json.loads(l) for l in open(data / "train.jsonl")][:4]
    texts = [r["prompt"] + r["target"] for r in train_rows] * 3
    proc = tiny_processor(texts)
    model = tiny_model(proc)
    print("tiny model params:", sum(p.numel() for p in model.parameters()))

    # 1 - masking
    ds = finetune.PageDataset(data, "train", limit=4)
    img, prompt, target = ds[0]
    b = build_inputs(proc, img, prompt, target)
    keep = b["labels"][0] != -100
    decoded = proc.tokenizer.decode(b["input_ids"][0][keep])
    n_img = int((b["input_ids"][0] == model.config.image_token_id).sum())
    print(f"[1] sequence {b['input_ids'].shape[1]} tokens, {n_img} image tokens, {int(keep.sum())} trained tokens")
    assert decoded.replace("<|im_end|>", "").strip() == target.strip(), "label masking is wrong"
    assert n_img > 0 and int(keep.sum()) < b["input_ids"].shape[1]
    print("    OK: only the JSON answer (+ end token) is trained, prompt and image are masked")

    # 2 - training loop with LoRA
    model = finetune.add_lora(model, rank=8, four_bit=False)
    model.print_trainable_parameters()
    tmp = Path(tempfile.mkdtemp())
    args = Namespace(out=str(tmp / "run"), lr=2e-3, epochs=1, max_steps=3, grad_accum=1, max_len=20000,
                     log_every=1, eval_every=3, val_samples=2, seed=0)
    best = finetune.train(model, proc, ds, ds, args)
    assert (tmp / "run" / "final_adapter" / "adapter_config.json").exists()
    assert (tmp / "run" / "best_adapter").exists()
    print(f"[2] OK: 3 steps ran, val loss {best:.3f}, adapters + training_log.csv saved")

    # 3 - gradients reach the LoRA weights and the loss falls when we repeat one page
    one = finetune.PageDataset(data, "train", limit=1)
    b1 = build_inputs(proc, *one[0])
    model.train()
    model(**b1).loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                if p.requires_grad and "lora_B" in n and p.grad is not None)
    model.zero_grad()
    assert gnorm > 0, "no gradient reaches the LoRA weights"
    args = Namespace(out=str(tmp / "overfit"), lr=1e-2, epochs=60, max_steps=60, grad_accum=1, max_len=20000,
                     log_every=20, eval_every=60, val_samples=1, seed=0)
    first = finetune.val_loss(model, proc, one, 20000, 1)
    last = finetune.train(model, proc, one, one, args)
    print(f"[3] LoRA gradient norm {gnorm:.3f}; repeating one page: loss {first:.3f} -> {last:.3f}")
    assert last < first - 0.1, "loss did not fall"
    print("    OK: gradients flow and the loss falls (a random 0.2M-param model cannot memorise a page; the real 2B model can)")

    # 4 - inference path
    ex = Extractor(model=model, processor=proc, max_new_tokens=40)
    raw, parsed, secs = ex.read_image(img)
    print(f"[4] OK: generate() + parse ran in {secs:.1f}s -> {len(parsed.get('rows', []))} rows parsed (tiny model, so rows are meaningless)")

    # 5 - scorer sanity: perfect prediction must score 100
    from evaluate import score
    test = [json.loads(l) for l in open(data / "test_unseen.jsonl")][:3]
    pf = tmp / "perfect.jsonl"
    with open(pf, "w") as f:
        for r in test:
            f.write(json.dumps({"id": r["id"], "raw": r["target"]}) + "\n")
    res = score(pf, data, "test_unseen")["before|all"]
    print(f"[5] perfect predictions -> numeric acc {res['numeric_field_acc']}, row exact {res['row_exact']}, "
          f"chain verified {res['chain_verified_rows']}")
    assert res["numeric_field_acc"] == 100 and res["row_exact"] == 100 and res["chain_verified_rows"] == 100
    shutil.rmtree(tmp)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/vlm")
