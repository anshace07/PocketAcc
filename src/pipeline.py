"""
pipeline.py - STEP 10 engine: one statement in -> verified, categorised, checked transactions out.

    PDF / photo -> pages -> pre-process (crop, deskew, resize)
                -> reader: fine-tuned Qwen2-VL (GPU + adapter)  or  Tesseract + rule parser (CPU)
                -> balance-chain verifier + corrector            (corrector.py, steps 5-6)
                -> transaction category, Linear SVM               (classifier.py, step 7)
                -> unusual-transaction check, hybrid forest       (anomaly.py, step 8)
                -> summary + CSV / Excel / JSON

Used by the web app (src/app.py) and from the command line:
    python src/pipeline.py statement.pdf --out result.json --csv result.csv --xlsx result.xlsx
    python src/pipeline.py --train-models            # (re)build models/*.joblib from the ground truth
"""
import argparse
import csv
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

from anomaly import AnomalyDetector
from classifier import TxnClassifier, all_transactions
from common import parse_amount
from corrector import finalize
from preprocess import prepare, render_pages

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("POCKETACC_MODELS", ROOT / "models"))
ADAPTER = os.environ.get("POCKETACC_ADAPTER", str(ROOT / "adapter/lora_r8"))

IFSC_PREFIX = {"HDFC": "HDFC", "SBIN": "SBI", "PUNB": "PNB", "KKBK": "KOTAK", "ICIC": "ICICI", "BARB": "BOB", "UTIB": "AXIS"}
BANK_NAMES = {"HDFC BANK": "HDFC", "STATE BANK OF INDIA": "SBI", "PUNJAB NATIONAL": "PNB", "KOTAK": "KOTAK",
              "ICICI": "ICICI", "BANK OF BARODA": "BOB", "AXIS BANK": "AXIS"}
BANK_LABEL = {"HDFC": "HDFC Bank", "SBI": "State Bank of India", "PNB": "Punjab National Bank", "KOTAK": "Kotak Mahindra Bank",
              "ICICI": "ICICI Bank", "BOB": "Bank of Baroda", "AXIS": "Axis Bank"}
CATEGORY_LABEL = {
    "upi_p2m": "UPI to merchant", "upi_p2p_in": "UPI received", "upi_p2p_out": "UPI to person", "neft_in": "NEFT received",
    "neft_out": "NEFT sent", "imps_out": "IMPS sent", "rtgs_in": "RTGS received", "salary": "Salary", "atm_wdl": "ATM withdrawal",
    "cash_dep": "Cash deposit", "bill_pay": "Bill payment", "nach_emi": "Loan EMI (NACH)", "sip": "SIP / investment",
    "charge": "Bank charges", "interest": "Interest", "pos": "Card payment (POS)", "cheque_paid": "Cheque paid",
    "cheque_dep": "Cheque deposited", "self": "Self transfer", "remit_in": "Foreign remittance", "dbt": "Govt. benefit (DBT)"}
AMOUNT = re.compile(r"\d{1,3}(?:,\d{2})*(?:,\d{3})?\.\d{2}|\d+\.\d{2}")


# ------------------------------------------------------------------------------- models

def train_models(out=MODELS):
    """Classifier on all 20,142 labelled transactions; anomaly forest on the 20 six-month statements."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gt = [ROOT / "data/original/ground_truth.json", ROOT / "data/generated/ground_truth.json"]
    TxnClassifier().fit(all_transactions(*gt)).save(out / "classifier.joblib")
    orig = json.load(open(gt[0])).values()
    AnomalyDetector().fit([(st["transactions"], [t["txn_type"] for t in st["transactions"]]) for st in orig]) \
        .save(out / "anomaly.joblib")
    return out


def load_models():
    if not (MODELS / "classifier.joblib").exists() or not (MODELS / "anomaly.joblib").exists():
        train_models()
    return TxnClassifier.load(MODELS / "classifier.joblib"), AnomalyDetector.load(MODELS / "anomaly.joblib")


# ------------------------------------------------------------------------------- readers

def vlm_available():
    try:
        import torch
        return torch.cuda.is_available() and Path(ADAPTER).exists()
    except ImportError:
        return False


def load_pages(path, dpi=150):
    """PDF -> one BGR image per page; photo/scan -> one page."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return render_pages(str(path), dpi)
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image {path.name}")
    return [img]


def detect_bank(lines):
    """Bank from the account's own IFSC code (the line labelled IFSC - narrations also contain other
    banks' IFSC codes), else from the bank name in the page header."""
    for ln in lines:
        up = ln.upper()
        if "IFSC" in up:
            m = re.search(r"(HDFC|SBIN|PUNB|KKBK|ICIC|BARB|UTIB)\s?0[A-Z0-9]{6}", up)
            if m:
                return IFSC_PREFIX[m.group(1)]
    head = " ".join(lines[:15]).upper()
    for name, code in BANK_NAMES.items():
        if name in head:
            return code
    return None


def header_from_lines(lines):
    """Opening / closing balance from OCR'd lines: the 'OPENING BALANCE B/F' row, or the summary box
    (labels on one line, the four amounts on the next)."""
    texts = [" ".join(w["text"] for w in ln) for ln in lines]
    opening = closing = None
    for i, t in enumerate(texts):
        up = t.upper()
        if opening is None and ("B/F" in up or "BROUGHT FORWARD" in up) and "BALANCE" in up:
            amts = AMOUNT.findall(t)
            if amts:
                opening = amts[-1]
        if "OPENING" in up and "CLOSING" in up and i + 1 < len(texts):
            amts = AMOUNT.findall(texts[i + 1])
            if len(amts) >= 2:
                opening = opening or amts[0]
                closing = amts[-1]
    return {"opening_balance": opening, "closing_balance": closing}


def tesseract_pages(pages, bank=None):
    """Tesseract + rule parser, page by page -> header, bank, [{"rows", "seconds"}]."""
    from ocr_baseline import lines_of, ocr_page, words
    header, out = {}, []
    for i, page in enumerate(pages):
        t0 = time.time()
        img = Image.fromarray(cv2.cvtColor(prepare(page), cv2.COLOR_BGR2RGB))
        lines = lines_of(words(img))
        if i == 0:
            bank = bank or detect_bank([" ".join(w["text"] for w in ln) for ln in lines])
            header = header_from_lines(lines)
        page_rows, _ = ocr_page(img, bank or "HDFC", lines=lines, max_gap=1.8)
        out.append({"rows": page_rows, "seconds": round(time.time() - t0, 1)})
    return header, bank, out


_EXTRACTOR = None


def split_opening_row(rows):
    """The model often writes the 'OPENING BALANCE B/F' line as a row. It is not a transaction: take its
    amount as the opening balance and drop it."""
    keep, opening = [], None
    for r in rows:
        nar = str(r[1]).upper() if len(r) > 1 else ""
        if "OPENING BALANCE" in nar or "B/F" in nar or "BROUGHT FORWARD" in nar:
            amounts = [a for a in (parse_amount(x) for x in r[2:]) if a is not None]
            if amounts and opening is None:
                opening = amounts[-1]
            continue
        keep.append(r)
    return keep, opening


def vlm_pages(pages, cache=None):
    """Fine-tuned Qwen2-VL, page by page -> header, bank, [{"rows", "seconds"}].
    cache: outputs recorded earlier on a GPU ({"raw": [...], "seconds": [...]}, one per page), so the
    model's real readings can be shown on a laptop without a GPU."""
    global _EXTRACTOR
    from common import parse_model_output
    header, out = None, []
    for i, page in enumerate(pages):
        if cache is not None:
            parsed, secs = parse_model_output(cache["raw"][i]), cache["seconds"][i]
        else:
            if _EXTRACTOR is None:
                from extract import Extractor
                _EXTRACTOR = Extractor(adapter=ADAPTER)
            _, parsed, secs = _EXTRACTOR.read_image(page)
        header = header or parsed.get("header")
        page_rows, page_opening = split_opening_row(parsed.get("rows", []))
        out.append({"rows": page_rows, "opening": page_opening, "seconds": round(secs, 1)})
    header = dict(header) if isinstance(header, dict) else {}
    if parse_amount(header.get("opening_balance")) is None and out and out[0]["opening"] is not None:
        header["opening_balance"] = f"{out[0]['opening']:.2f}"
    return header, detect_bank(["IFSC " + str(header.get("ifsc", "")), str(header.get("bank", ""))]), out


def _first_row_opening(rows):
    if not rows:
        return None
    _, _, dr, cr, bal = (list(rows[0]) + [""] * 5)[:5]
    b = parse_amount(bal)
    return None if b is None else round(b + (parse_amount(dr) or 0.0) - (parse_amount(cr) or 0.0), 2)


def chain_check(rows, opening):
    """Share of rows whose balance chain verifies AFTER the corrector has run (what the user gets),
    and the last balance."""
    import copy
    from common import rows_to_transactions
    from corrector import Txn, correct, verify
    t = rows_to_transactions(copy.deepcopy(rows))
    if not t or opening is None:
        return 0.0, None
    chain = [Txn(x["debit"], x["credit"], x["balance"] if x["balance"] is not None else float("nan")) for x in t]
    correct(opening, chain)
    ok = verify(opening, chain)
    last = next((c.balance for c in reversed(chain) if c.balance == c.balance), None)
    return sum(ok) / len(ok), last


def read_both(pages, bank=None, cache=None):
    """Read every page with BOTH readers and keep, per page, the reading whose balance chain verifies
    on more rows after correction (ties -> Tesseract). The choice uses only the statement's own arithmetic."""
    th, tbank, tp = tesseract_pages(pages, bank)
    vh, vbank, vp = vlm_pages(pages, cache)
    printed = [o for o in (parse_amount(th.get("opening_balance")), parse_amount(vh.get("opening_balance"))) if o is not None]
    rows, info, prev, opening = [], [], None, None
    for i, (t, v) in enumerate(zip(tp, vp)):
        best = {}
        for name, r in (("tesseract", t["rows"]), ("vlm", v["rows"])):
            starts = [prev] if prev is not None else (printed or [_first_row_opening(r)])
            best[name] = max(((chain_check(r, o), o) for o in starts), key=lambda x: x[0][0])
        pick = "vlm" if best["vlm"][0][0] > best["tesseract"][0][0] else "tesseract"
        (frac, last), used = best[pick]
        if i == 0:
            opening = used
        chosen = v["rows"] if pick == "vlm" else t["rows"]
        rows += chosen
        prev = last if last is not None else prev
        info.append({"page": i + 1, "rows": len(chosen), "seconds": round(t["seconds"] + v["seconds"], 1), "reader": pick,
                     "ocr_seconds": t["seconds"], "model_seconds": v["seconds"], "model_recorded": cache is not None,
                     "verified": {"tesseract": round(100 * best["tesseract"][0][0], 1), "vlm": round(100 * best["vlm"][0][0], 1)}})
    header = {"opening_balance": None if opening is None else f"{opening:.2f}",
              "closing_balance": th.get("closing_balance") or vh.get("closing_balance")}
    return header, rows, bank or tbank or vbank, info


def read_tesseract(pages, bank=None):
    header, bank, out = tesseract_pages(pages, bank)
    return header, [r for p in out for r in p["rows"]], bank, \
        [{"page": i + 1, "rows": len(p["rows"]), "seconds": p["seconds"], "reader": "tesseract"} for i, p in enumerate(out)]


def read_vlm(pages, cache=None):
    header, bank, out = vlm_pages(pages, cache)
    return header, [r for p in out for r in p["rows"]], bank, \
        [{"page": i + 1, "rows": len(p["rows"]), "seconds": p["seconds"], "reader": "vlm"} for i, p in enumerate(out)]


# ------------------------------------------------------------------------------- analysis

def status_of(flags):
    if any("UNRESOLVED" in f for f in flags):
        return "review"
    if any("->" in f for f in flags):
        return "corrected"
    return "ok"


def analyse(path, backend="auto", bank=None, models=None, vlm_cache=None):
    """backend: auto | both | vlm | tesseract. auto = both readers when the model is available (GPU, or
    outputs recorded on a GPU in vlm_cache), else Tesseract."""
    t0 = time.time()
    clf, det = models or load_models()
    if backend == "auto":
        backend = "both" if (vlm_cache is not None or vlm_available()) else "tesseract"
    pages = load_pages(path)
    if backend == "both":
        header, rows, bank_found, info = read_both(pages, bank, vlm_cache)
    elif backend == "vlm":
        header, rows, bank_found, info = read_vlm(pages, vlm_cache)
    else:
        header, rows, bank_found, info = read_tesseract(pages, bank)
    header = dict(header or {})
    notes = []
    opening_source = "printed"
    if parse_amount(header.get("opening_balance")) is None and rows:
        # not printed / not read: take it from the first row (balance + debit - credit) and say so
        _, _, dr, cr, bal = (list(rows[0]) + [""] * 5)[:5]
        b, d, c = parse_amount(bal), parse_amount(dr) or 0.0, parse_amount(cr) or 0.0
        if b is not None:
            header["opening_balance"] = f"{b + d - c:.2f}"
            opening_source = "inferred from the first row"
            notes.append("Opening balance was not found on the page, so it was worked out from the first row; "
                         "an error in that row cannot be detected.")
    res = finalize(header, rows)
    txns = res["transactions"]
    cats = clf.predict(txns)
    scores, anomalous, reasons = det.score(txns, cats)
    for t, c, s, a, r in zip(txns, cats, scores, anomalous, reasons):
        t.setdefault("flags", [])
        t.update(category=c, category_label=CATEGORY_LABEL.get(c, c), status=status_of(t["flags"]),
                 anomaly=bool(a), anomaly_reason=r, anomaly_score=round(float(s), 3))
    v = res["verification"] or {"rows": len(txns), "verified": None, "trust_score": None, "corrections": 0,
                                "flagged": None}
    spend, income = defaultdict(float), defaultdict(float)
    for t in txns:
        if t["debit"]:
            spend[t["category_label"]] += t["debit"]
        if t["credit"]:
            income[t["category_label"]] += t["credit"]
    if not txns:
        notes.append("No transactions were read. Try a straighter, brighter photo, or a PDF.")
    elif v["trust_score"] is not None and v["trust_score"] < 80:
        notes.append(f"LOW TRUST: only {v['trust_score']}% of rows pass the balance check. The page is probably too "
                     "damaged for this reader - retake the photo flat and well lit, upload the PDF, or use the GPU model.")
    elif backend == "tesseract":
        notes.append("Read with Tesseract OCR (CPU). It is reliable on clean PDFs and scans, weak on damaged photos; "
                     "the fine-tuned vision model is used automatically when a GPU and adapter are available.")
    elif backend == "both" and vlm_cache is not None:
        notes.append("Tesseract ran live on this computer; the fine-tuned model's reading was recorded on a Kaggle "
                     "Tesla T4 GPU for this sample. Each page keeps the reading whose balance chain verifies best.")
    return {
        "file": Path(path).name, "backend": backend, "bank": bank_found, "bank_label": BANK_LABEL.get(bank_found, "Unknown bank"),
        "pages": info, "seconds": round(time.time() - t0, 1), "notes": notes,
        "opening_balance": parse_amount(header.get("opening_balance")), "opening_source": opening_source,
        "closing_balance": parse_amount(header.get("closing_balance")),
        "summary": {
            "transactions": len(txns), "verified": v["verified"], "trust_score": v["trust_score"],
            "auto_corrected": sum(t["status"] == "corrected" for t in txns),
            "needs_review": sum(t["status"] == "review" for t in txns),
            "anomalies": sum(t["anomaly"] for t in txns),
            "total_debit": round(sum(t["debit"] or 0 for t in txns), 2),
            "total_credit": round(sum(t["credit"] or 0 for t in txns), 2),
            "closing_matches": (abs(txns[-1]["balance"] - parse_amount(header.get("closing_balance"))) < 0.005
                                if txns and txns[-1]["balance"] is not None
                                and parse_amount(header.get("closing_balance")) is not None else None),
            "spend_by_category": sorted(([k, round(x, 2)] for k, x in spend.items()), key=lambda kv: -kv[1]),
            "income_by_category": sorted(([k, round(x, 2)] for k, x in income.items()), key=lambda kv: -kv[1]),
        },
        "transactions": txns,
    }


# ------------------------------------------------------------------------------- exports

COLUMNS = ["date", "narration", "debit", "credit", "balance", "category", "status", "corrections", "unusual", "why_unusual"]


def export_rows(result):
    for t in result["transactions"]:
        yield [t["date"] or "", t["narration"], t["debit"] if t["debit"] is not None else "",
               t["credit"] if t["credit"] is not None else "", t["balance"] if t["balance"] is not None else "",
               t["category_label"], t["status"], "; ".join(f for f in t["flags"] if "UNRESOLVED" not in f),
               "yes" if t["anomaly"] else "", t["anomaly_reason"]]


def to_csv(result):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(COLUMNS)
    w.writerows(export_rows(result))
    return buf.getvalue()


def to_xlsx(result):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Transactions"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    fill = {"corrected": PatternFill("solid", fgColor="DDF3E4"), "review": PatternFill("solid", fgColor="FBE0E0")}
    unusual = PatternFill("solid", fgColor="FFF1CC")
    for t, row in zip(result["transactions"], export_rows(result)):
        ws.append(row)
        f = fill.get(t["status"]) or (unusual if t["anomaly"] else None)
        if f:
            for c in ws[ws.max_row]:
                c.fill = f
    for col in "CDE":
        for c in ws[col][1:]:
            c.number_format = "#,##,##0.00"
    for col, wdt in zip("ABCDEFGHIJ", [11, 60, 12, 12, 14, 20, 10, 30, 8, 40]):
        ws.column_dimensions[col].width = wdt
    ws.freeze_panes = "A2"
    s = result["summary"]
    sm = wb.create_sheet("Summary")
    for k, v in [("File", result["file"]), ("Bank", result["bank_label"]), ("Read with", result["backend"]),
                 ("Opening balance", result["opening_balance"]), ("Closing balance", result["closing_balance"]),
                 ("Transactions", s["transactions"]), ("Balance chain verified", s["verified"]),
                 ("Trust score %", s["trust_score"]), ("Auto-corrected", s["auto_corrected"]),
                 ("Needs review", s["needs_review"]), ("Unusual", s["anomalies"]), ("Total debit", s["total_debit"]),
                 ("Total credit", s["total_credit"]), ("", ""), ("Spending by category", "")] + s["spend_by_category"]:
        sm.append([k, v])
    sm.column_dimensions["A"].width = 28
    sm.column_dimensions["B"].width = 22
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--backend", default="auto", choices=["auto", "both", "tesseract", "vlm"])
    ap.add_argument("--bank", default=None, choices=sorted(BANK_LABEL))
    ap.add_argument("--out")
    ap.add_argument("--csv")
    ap.add_argument("--xlsx")
    ap.add_argument("--train-models", action="store_true")
    a = ap.parse_args()
    if a.train_models:
        print("saved models to", train_models())
        if not a.path:
            return
    r = analyse(a.path, a.backend, a.bank)
    s = r["summary"]
    print(f"{r['file']}: {r['bank_label']}, {len(r['pages'])} page(s), read with {r['backend']} in {r['seconds']}s")
    print(f"{s['transactions']} transactions | trust score {s['trust_score']}% | auto-corrected {s['auto_corrected']} | "
          f"needs review {s['needs_review']} | unusual {s['anomalies']} | closing balance matches: {s['closing_matches']}")
    for n in r["notes"]:
        print("note:", n)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1, ensure_ascii=False)
    if a.csv:
        open(a.csv, "w", newline="").write(to_csv(r))
    if a.xlsx:
        open(a.xlsx, "wb").write(to_xlsx(r))


if __name__ == "__main__":
    main()
