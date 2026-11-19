"""
vlm.py - loading Qwen2-VL (optionally in 4-bit + a LoRA adapter) and building its inputs.

Shared by finetune.py (training) and extract.py (inference) so both see the page in
exactly the same way.
"""
import torch

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def compute_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_processor(model_id=DEFAULT_MODEL, max_pixels=900_000):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id, min_pixels=64 * 28 * 28, max_pixels=max_pixels)


def bnb_config():
    """4-bit NF4 for the LANGUAGE model only. The vision encoder stays in fp16/bf16.

    Why: Qwen2-VL (transformers 4.51) casts the page image to the dtype of a vision-encoder
    weight before reading it. A 4-bit weight is stored as uint8, so with the vision encoder
    quantised the image arrives as a handful of integers (0..255) and is destroyed.
    Keeping the ~0.6B-parameter vision encoder unquantised costs about 1 GB more memory.
    """
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=compute_dtype(),
                              llm_int8_skip_modules=["visual", "lm_head"])


def check_vision_dtype(model):
    """Stop early if the page image would be cast to an integer type (see bnb_config)."""
    visual = getattr(model, "visual", None)
    if visual is not None and hasattr(visual, "get_dtype") and not visual.get_dtype().is_floating_point:
        raise RuntimeError(f"vision encoder dtype is {visual.get_dtype()}: the page image would be cast to it and "
                           "destroyed. Keep 'visual' out of 4-bit quantisation (vlm.bnb_config).")


def load_model(model_id=DEFAULT_MODEL, four_bit=True, adapter=None):
    """4-bit NF4 quantisation (QLoRA) of the language model; the vision encoder stays fp16/bf16."""
    try:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    except ImportError:                                   # newer transformers
        from transformers import AutoModelForImageTextToText as Cls
    kw = dict(torch_dtype=compute_dtype())
    if four_bit and torch.cuda.is_available():
        kw["quantization_config"] = bnb_config()
        kw["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    model = Cls.from_pretrained(model_id, **kw)
    check_vision_dtype(model)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model


def messages_for(prompt):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def build_inputs(processor, image, prompt, target=None):
    """Returns tokenised inputs; if target is given also returns labels with the prompt masked.

    Loss masking: the model should only be graded on the JSON answer it writes, not on
    re-predicting the image tokens or the instruction, so those label positions are -100.
    """
    prompt_text = processor.apply_chat_template(messages_for(prompt), tokenize=False, add_generation_prompt=True)
    if target is None:
        return processor(text=[prompt_text], images=[image], return_tensors="pt")
    eos = processor.tokenizer.eos_token or "<|im_end|>"
    full = processor(text=[prompt_text + target + eos], images=[image], return_tensors="pt")
    n_prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    return full
