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


def load_processor(model_id=DEFAULT_MODEL, max_pixels=1_400_000):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id, min_pixels=64 * 28 * 28, max_pixels=max_pixels)


def load_model(model_id=DEFAULT_MODEL, four_bit=True, adapter=None):
    """4-bit NF4 quantisation (QLoRA) keeps the 2B model at ~1.5 GB of GPU memory."""
    try:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    except ImportError:                                   # newer transformers
        from transformers import AutoModelForImageTextToText as Cls
    kw = dict(torch_dtype=compute_dtype())
    if four_bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=compute_dtype())
        kw["device_map"] = {"": 0}
    elif torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    model = Cls.from_pretrained(model_id, **kw)
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
