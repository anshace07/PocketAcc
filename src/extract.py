"""
extract.py - STEP 4c: read a statement with the (fine-tuned) vision-language model.

    page image -> preprocess -> Qwen2-VL (+ LoRA adapter) -> JSON -> normalised transactions
    whole statement -> pages merged -> balance-chain verification + correction (corrector.py)

Usage:
    python src/extract.py statement.pdf --adapter runs/lora_r8/best_adapter
    python src/extract.py photo.jpg --adapter runs/lora_r8/best_adapter --out result.json
    python src/extract.py photo.jpg                     # zero-shot base model (baseline)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

from common import PROMPT, parse_amount, parse_model_output, rows_to_transactions
from corrector import Txn, correct
from preprocess import prepare, render_pages
from vlm import DEFAULT_MODEL, build_inputs, load_model, load_processor


class Extractor:
    def __init__(self, model_id=DEFAULT_MODEL, adapter=None, four_bit=True, max_pixels=1_400_000,
                 max_new_tokens=4096, model=None, processor=None):
        self.processor = processor or load_processor(model_id, max_pixels)
        self.model = model or load_model(model_id, four_bit=four_bit, adapter=adapter)
        self.model.eval()
        self.max_pixels, self.max_new_tokens = max_pixels, max_new_tokens

    @torch.no_grad()
    def read_image(self, image, already_prepared=False):
        """image: PIL.Image or BGR numpy array. Returns (raw_text, parsed_dict, seconds)."""
        if not isinstance(image, Image.Image):
            img = image if already_prepared else prepare(image, self.max_pixels)
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        batch = build_inputs(self.processor, image, PROMPT).to(self.model.device)
        t0 = time.time()
        out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text, parse_model_output(text), time.time() - t0

    def read_statement(self, path, dpi=150):
        path = Path(path)
        pages = render_pages(str(path), dpi) if path.suffix.lower() == ".pdf" else [cv2.imread(str(path))]
        header, rows, timing = None, [], []
        for i, page in enumerate(pages, 1):
            _, parsed, secs = self.read_image(page)
            timing.append(round(secs, 1))
            header = header or parsed.get("header")
            rows += parsed.get("rows", [])
            print(f"page {i}/{len(pages)}: {len(parsed.get('rows', []))} rows in {secs:.1f}s", flush=True)
        return finalize(header, rows) | {"seconds_per_page": timing}


def finalize(header, rows):
    """Normalise rows and run the balance-chain verifier/corrector."""
    txns = rows_to_transactions(rows)
    opening = parse_amount((header or {}).get("opening_balance"))
    closing = parse_amount((header or {}).get("closing_balance"))
    report = None
    if opening is not None and txns:
        chain = [Txn(t["debit"], t["credit"], t["balance"] if t["balance"] is not None else 0.0) for t in txns]
        report = correct(opening, chain, closing)
        for t, c in zip(txns, chain):
            t["debit"], t["credit"], t["balance"], t["flags"] = c.debit, c.credit, c.balance, c.flags
    return {"header": header, "transactions": txns, "verification": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="PDF or image of a statement")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=1_400_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ex = Extractor(a.model, a.adapter, not a.no_4bit, a.max_pixels)
    res = ex.read_statement(a.path)
    v = res["verification"] or {}
    print(f"\n{len(res['transactions'])} transactions | trust score {v.get('trust_score')}% | "
          f"auto-corrected {v.get('corrections')} | flagged {v.get('flagged')}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("saved", a.out)


if __name__ == "__main__":
    main()
