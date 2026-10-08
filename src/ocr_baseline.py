"""
ocr_baseline.py - the TRADITIONAL baseline: Tesseract OCR + rule-based table parsing.

This is how statement extraction was done before vision-language models:
  1. Tesseract reads every word and its (x, y) box
  2. words are grouped into text lines
  3. a line that starts with a date starts a new transaction; other lines continue the narration
  4. column positions (found from the table header, or from the bank's known layout) decide
     whether an amount is a debit, a credit or the balance

Our model has to beat this to justify itself. Both systems are scored by evaluate.py.
"""
import re
import time

import cv2
import numpy as np
import pytesseract
from PIL import Image

from common import parse_date

# fractional column layout (x from left edge of the paper), derived from the bank templates
MARGIN_FRAC = 26 / 595
LAYOUT = {
    "HDFC": [("date", 8), ("narration", 38), ("ref", 13), ("value_date", 8), ("debit", 11), ("credit", 11), ("balance", 12)],
    "SBI": [("date", 9), ("value_date", 9), ("narration", 36), ("ref", 13), ("debit", 10), ("credit", 10), ("balance", 12)],
    "PNB": [("date", 9), ("ref", 12), ("narration", 45), ("debit", 11), ("credit", 11), ("balance", 12)],
    "KOTAK": [("date", 8), ("narration", 44), ("ref", 14), ("debit", 11), ("credit", 11), ("balance", 12)],
    "ICICI": [("sr", 4), ("value_date", 9), ("date", 9), ("ref", 8), ("narration", 36), ("debit", 11), ("credit", 11), ("balance", 12)],
    "BOB": [("sr", 4), ("date", 9), ("value_date", 9), ("narration", 36), ("ref", 9), ("debit", 10), ("credit", 10), ("balance", 12)],
    "AXIS": [("date", 9), ("ref", 8), ("narration", 40), ("debit", 11), ("credit", 11), ("balance", 12), ("init", 5)],
}
HEADER_WORDS = {"debit": ["withdrawal", "debit", "amount(dr)", "dr)"], "credit": ["deposit", "credit", "amount(cr)", "cr)"],
                "balance": ["balance"], "narration": ["narration", "description", "remarks", "particulars"]}
DATE_RE = re.compile(r"^(\d{2}[/-]\d{2}[/-]\d{2,4}|\d{2}-[A-Za-z]{3}-\d{2,4})$")
DATE_WORDS_RE = re.compile(r"^\d{2}$")          # '01' 'Mar' '2026' (SBI style) handled below
AMOUNT_RE = re.compile(r"^-?\d{1,3}(,\d{2})*(,\d{3})?\.\d{2}$|^\d+\.\d{2}$")


def column_spans(bank):
    cols = LAYOUT.get(bank, LAYOUT["HDFC"])
    total = sum(w for _, w in cols)
    x, spans = MARGIN_FRAC, {}
    for k, w in cols:
        span = (1 - 2 * MARGIN_FRAC) * w / total
        spans[k] = (x, x + span)
        x += span
    return spans


def words(img: Image.Image):
    """Tesseract word boxes (image upscaled 2x - Tesseract prefers ~30 px tall text)."""
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    arr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    d = pytesseract.image_to_data(arr, config="--psm 6", output_type=pytesseract.Output.DICT)
    W = arr.shape[1]
    out = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) > -1:
            out.append(dict(text=t, x0=d["left"][i] / W, x1=(d["left"][i] + d["width"][i]) / W,
                            y=d["top"][i] + d["height"][i] / 2, h=d["height"][i],
                            key=(d["block_num"][i], d["par_num"][i], d["line_num"][i])))
    return out


def lines_of(ws):
    lines = {}
    for w in ws:
        lines.setdefault(w["key"], []).append(w)
    out = [sorted(v, key=lambda w: w["x0"]) for v in lines.values()]
    return sorted(out, key=lambda ln: np.mean([w["y"] for w in ln]))


def header_spans(lines, fallback):
    """Refine amount-column centres from the printed header if Tesseract read it."""
    for ln in lines[:40]:
        low = [(w["text"].lower(), (w["x0"] + w["x1"]) / 2) for w in ln]
        found = {}
        for col, keys in HEADER_WORDS.items():
            for t, xc in low:
                if any(t.startswith(k) or t == k for k in keys):
                    found.setdefault(col, xc)
        has_amount = any(AMOUNT_RE.match(w["text"]) for w in ln)
        if {"debit", "credit", "balance"} <= found.keys() and not has_amount \
                and found["debit"] < found["credit"] < found["balance"] and found["debit"] > 0.35:
            spans = dict(fallback)
            for col in ("debit", "credit", "balance"):
                a, b = fallback[col]
                half = (b - a) / 2
                spans[col] = (found[col] - half, found[col] + half)
            return spans
    return fallback


def ocr_page(img: Image.Image, bank: str):
    t0 = time.time()
    lines = lines_of(words(img))
    spans = header_spans(lines, column_spans(bank))
    centres = {k: (a + b) / 2 for k, (a, b) in spans.items() if k in ("debit", "credit", "balance")}
    nar_lo, nar_hi = spans.get("narration", (0.2, 0.6))
    rows = []
    for ln in lines:
        texts = [w["text"] for w in ln]
        # date at the start of the line (one token like 01/03/26, or three like 01 Mar 2026)
        date_str, start = None, 0
        for k in range(min(4, len(ln))):
            if DATE_RE.match(texts[k]) and parse_date(texts[k]):
                date_str, start = texts[k], k + 1
                break
            if k + 2 < len(ln) and DATE_WORDS_RE.match(texts[k]) and parse_date(" ".join(texts[k:k + 3])):
                date_str, start = " ".join(texts[k:k + 3]), k + 3
                break
        joined = " ".join(texts).upper()
        if "OPENING BALANCE" in joined or "BALANCE B/F" in joined:
            continue
        amounts = [(w["text"], (w["x0"] + w["x1"]) / 2) for w in ln if AMOUNT_RE.match(w["text"])]
        nar = " ".join(w["text"] for w in ln[start:] if nar_lo - 0.01 <= w["x0"] <= nar_hi and not AMOUNT_RE.match(w["text"]))
        if date_str and amounts:
            row = {"date": date_str, "narration": nar, "debit": "", "credit": "", "balance": ""}
            for txt, xc in amounts:
                col = min(centres, key=lambda c: abs(centres[c] - xc))
                if not row[col]:
                    row[col] = txt
            rows.append(row)
        elif rows and not date_str and nar and not amounts:
            rows[-1]["narration"] += nar                 # wrapped narration line
    out = [[r["date"], r["narration"], r["debit"], r["credit"], r["balance"]] for r in rows]
    return out, time.time() - t0
