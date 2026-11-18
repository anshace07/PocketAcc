"""
demo_live.py - the LIVE part of the demo notebook (notebooks/pocketacc_demo_kaggle.ipynb).

    import demo_live
    demo_live.load()                                         # loads the fine-tuned model once (~1-2 min)
    demo_live.show("demo/samples/A_page1_photo_heavy.jpg")    # reads it now (~2-3 min per page on a T4)

show() prints the page, the table the fine-tuned model read (with corrected / needs-review / unusual rows
coloured like the web app), the balance-chain trust score, the answer-key check for the demo
statements, and - for comparison - what Tesseract OCR reads from the same page.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

import pipeline  # noqa: E402

_MODELS = None
COLORS = {"corrected": "#e3f4e3", "review": "#fbe4e4", "unusual": "#fff3d6"}


def load():
    """Load the classifier, the anomaly detector and the fine-tuned vision model (once)."""
    global _MODELS
    t0 = time.time()
    _MODELS = pipeline.load_models()
    if not pipeline.vlm_available():
        raise RuntimeError(f"No GPU or no adapter ({pipeline.ADAPTER}). Settings: Accelerator = GPU T4 x2.")
    pipeline.read_vlm([])                        # builds the model + adapter without reading anything
    four_bit, px = pipeline.vlm_settings()
    print(f"Fine-tuned model ready in {time.time() - t0:.0f} s  (Qwen2-VL-2B + LoRA adapter "
          f"{Path(pipeline.ADAPTER).name}, {'4-bit' if four_bit else 'fp16'}, pages resized to <= {px:,} pixels)")


def _thumb(path, height=760):
    img = pipeline.load_pages(path)[0]
    s = height / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * s), height), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def _state(t):
    if t["status"] in ("corrected", "review"):
        return t["status"]
    return "unusual" if t["anomaly"] else "ok"


def table(r):
    rows = []
    for i, t in enumerate(r["transactions"], 1):
        st = _state(t)
        why = [f for f in t["flags"] if "->" in f] + ([t["anomaly_reason"]] if t["anomaly"] else [])
        rows.append({"#": i, "date": t["date"], "narration": t["narration"],
                     "debit": t["debit"], "credit": t["credit"], "balance": t["balance"],
                     "category": t["category_label"],
                     "check": {"ok": "ok", "corrected": "corrected", "review": "NEEDS REVIEW",
                               "unusual": "UNUSUAL"}[st] + (": " + "; ".join(why) if why else ""),
                     "_state": st})
    return pd.DataFrame(rows)


def styled(df):
    def paint(row):
        c = COLORS.get(row["_state"])
        return [f"background-color: {c}" if c else "" for _ in row]
    fmt = {k: (lambda x: "" if pd.isna(x) else f"{x:,.2f}") for k in ("debit", "credit", "balance")}
    return df.style.apply(paint, axis=1).format(fmt).hide(axis="index").hide(subset=["_state"], axis="columns")


def summary_line(r, reader):
    s, k = r["summary"], r.get("answer_key")
    trust = "not checked (no opening balance)" if s["trust_score"] is None else \
        f"{s['trust_score']}% ({s['verified']} of {s['transactions']} rows pass the balance check)"
    line = (f"{reader}: {s['transactions']} rows in {r['seconds']} s | trust score {trust} | "
            f"auto-corrected {s['auto_corrected']} | needs review {s['needs_review']} | unusual {s['anomalies']}")
    if k:
        line += (f"\n   answer key: {k['numbers_right']} of {k['answer_rows']} transactions with date and every amount "
                 f"exactly right ({k['numbers_pct']}%), {k['fully_right']} also with the exact text")
    return line


def show(path, compare_tesseract=True):
    """Read one statement (PDF or photo) live with the fine-tuned model and display everything."""
    from IPython.display import Image, Markdown, display
    if _MODELS is None:
        load()
    path = Path(path)
    display(Markdown(f"### {path.name}"))
    display(Image(data=_thumb(path)))
    n = len(pipeline.load_pages(path))
    print(f"Reading {n} page(s) with the fine-tuned model ... about 2-3 minutes per page on a T4", flush=True)
    r = pipeline.attach_answer_key(pipeline.analyse(path, "vlm", None, _MODELS), path.name)
    print(summary_line(r, "FINE-TUNED MODEL"))
    for note in r["notes"]:
        print("   note:", note)
    display(styled(table(r)))
    if compare_tesseract:
        rt = pipeline.attach_answer_key(pipeline.analyse(path, "tesseract", None, _MODELS), path.name)
        print(summary_line(rt, "Tesseract OCR (traditional baseline) on the same file"))
    return r
