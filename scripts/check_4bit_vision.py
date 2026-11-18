"""
check_4bit_vision.py - regression test for the 4-bit vision bug (docs/FAILURES.md, Nov 2026).

Qwen2-VL (transformers 4.51) casts the page image to the dtype of a vision-encoder weight.
If the vision encoder is quantised to 4-bit, that dtype is uint8 and the image is destroyed.
This test builds a tiny random Qwen2-VL, loads it in 4-bit with OUR config (vlm.bnb_config),
and checks on CPU that:
  1. the old config really does destroy the image (the bug exists),
  2. our config gives the vision encoder exactly the same pixels as the unquantised model,
  3. add_lora + 2 training steps run on the 4-bit model and gradients reach LoRA.

Run (CPU, needs bitsandbytes >= 0.45):   python scripts/check_4bit_vision.py
"""
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import transformers.integrations as integ  # noqa: E402
import transformers.utils as tu  # noqa: E402
from transformers import BitsAndBytesConfig, Qwen2VLConfig, Qwen2VLForConditionalGeneration  # noqa: E402

import finetune  # noqa: E402
from smoke_test import tiny_processor  # noqa: E402
from vlm import bnb_config, build_inputs, check_vision_dtype  # noqa: E402

# test harness only: let transformers use bitsandbytes' CPU backend (on the GPU this path is CUDA)
integ.validate_bnb_backend_availability = lambda **k: True
tu.is_bitsandbytes_multi_backend_available = lambda: True


def tiny_model(proc):
    """Like smoke_test.tiny_model, but layers >= 64 wide (bitsandbytes' 4-bit block size)."""
    t = proc.tokenizer
    tid = t.convert_tokens_to_ids
    cfg = Qwen2VLConfig(
        vocab_size=len(t), hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=32768, rope_scaling={"type": "mrope", "mrope_section": [4, 6, 6]},
        vision_config=dict(depth=2, embed_dim=64, hidden_size=128, num_heads=2, mlp_ratio=2, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, in_channels=3),
        image_token_id=tid("<|image_pad|>"), video_token_id=tid("<|video_pad|>"),
        vision_start_token_id=tid("<|vision_start|>"), vision_end_token_id=tid("<|vision_end|>"),
        bos_token_id=tid("<|endoftext|>"), eos_token_id=tid("<|im_end|>"), pad_token_id=tid("<|endoftext|>"),
        tie_word_embeddings=True)
    return Qwen2VLForConditionalGeneration(cfg)


def pixels_seen_by_vision(model, inputs):
    got = {}
    h = model.visual.patch_embed.register_forward_pre_hook(lambda m, a: got.setdefault("x", a[0].detach().clone()))
    with torch.no_grad():
        model(**inputs)
    h.remove()
    return got["x"]


class OnePage(torch.utils.data.Dataset):
    def __init__(self, img, prompt, target):
        self.item = (img, prompt, target)

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return self.item


def main():
    torch.manual_seed(0)
    target = '{"header": null, "rows": [["01/03/26", "UPI-OLA CABS", "236.13", "", "20,504.00"]]}'
    proc = tiny_processor([target, "Extract every transaction"] * 50)
    d = tempfile.mkdtemp()
    tiny_model(proc).save_pretrained(d)
    img = Image.new("RGB", (560, 420), "white")
    draw = ImageDraw.Draw(img)
    for y in range(20, 400, 30):
        draw.rectangle([20, y, 500, y + 8], fill="black")
    inputs = build_inputs(proc, img, "Extract every transaction")

    def load(cfg):
        return Qwen2VLForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32, quantization_config=cfg,
                                                              device_map={"": "cpu"})

    ref = pixels_seen_by_vision(Qwen2VLForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32), inputs)
    print(f"unquantised        : vision input dtype {ref.dtype}, range [{ref.min():.2f}, {ref.max():.2f}]")

    old = load(BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float32))
    x = pixels_seen_by_vision(old, inputs)
    print(f"[1] old 4-bit config: vision dtype {old.visual.get_dtype()}, image range [{x.min():.0f}, {x.max():.0f}], "
          f"{x.unique().numel()} distinct values")
    assert not torch.allclose(x.float(), ref), "expected the old config to destroy the image"
    try:
        check_vision_dtype(old)
        raise AssertionError("check_vision_dtype should have refused the old config")
    except RuntimeError:
        print("    OK: bug reproduced, and check_vision_dtype() refuses this model")

    new = load(bnb_config())
    check_vision_dtype(new)
    x = pixels_seen_by_vision(new, inputs)
    print(f"[2] our 4-bit config: vision {type(new.visual.blocks[0].mlp.fc2).__name__} ({new.visual.get_dtype()}), "
          f"language {type(new.model.layers[0].self_attn.q_proj).__name__}, image identical: {torch.allclose(x, ref)}")
    assert torch.allclose(x, ref) and "4bit" in type(new.model.layers[0].self_attn.q_proj).__name__
    print("    OK: language model is 4-bit, vision encoder sees the real image")

    # fresh 4-bit model for training; bitsandbytes' CPU-only inference repacking (AVX512-BF16) breaks
    # training after an eval forward, so switch it off - this does not exist on the GPU
    model = load(bnb_config())
    for m in model.modules():
        if hasattr(m, "support_avx512bf16_for_cpu"):
            m.support_avx512bf16_for_cpu = False
    model = finetune.add_lora(model, rank=8, four_bit=True)
    check_vision_dtype(model)
    args = Namespace(out=str(Path(d) / "run"), lr=1e-3, epochs=1, max_steps=2, grad_accum=1, max_len=20000,
                     log_every=1, eval_every=2, val_samples=1, seed=0)
    ds = OnePage(img, "Extract every transaction", target)
    best = finetune.train(model, proc, ds, ds, args)
    b = build_inputs(proc, img, "Extract every transaction", target)
    model.train()
    model(**b).loss.backward()
    g = sum(p.grad.abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n and p.grad is not None)
    assert g > 0 and best == best, "no gradient reached LoRA / loss is NaN"
    print(f"[3] OK: add_lora + 2 training steps on the 4-bit model, val loss {best:.3f}, LoRA grad norm {g:.4f}")
    print("\nALL 4-BIT VISION CHECKS PASSED")


if __name__ == "__main__":
    main()
