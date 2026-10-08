"""
build_dataset.py - STEP 4a: turn statements into (page image -> JSON) training examples.

For every page of every statement we know exactly which transactions are printed on it,
so each example is:
    image  : the page (clean or degraded, then pre-processed exactly like at inference)
    target : {"header": {...} on page 1 / null, "rows": [[date, narration, debit, credit, balance], ...]}

Split design (so we can measure generalisation honestly):
    train        generated statements of the TRAIN banks (HDFC, SBI, PNB, KOTAK) - clean + degraded copies
    val          2 held-out generated statements per train bank
    test_seen    the ORIGINAL hand-made statements of the train banks (same banks, different author/layout details)
    test_unseen  every statement of the UNSEEN banks (ICICI, BOB, AXIS) - never seen in training

Usage:
    python src/build_dataset.py --out data/vlm                       # default experiment
    python src/build_dataset.py --out data/vlm --train-banks all     # final model on all 7 banks
"""
import argparse
import json
import random
import re
import subprocess
from pathlib import Path

import cv2

from common import PROMPT, inr, target_json
from degrade import degrade
from preprocess import prepare, render_pages

ROOT = Path(__file__).resolve().parents[1]
DATE_FMT = {"HDFC": "%d/%m/%y", "SBI": "%d %b %Y", "PNB": "%d/%m/%Y", "KOTAK": "%d-%b-%y",
            "ICICI": "%d-%m-%Y", "BOB": "%d/%m/%Y", "AXIS": "%d-%m-%Y"}
ALL_BANKS = list(DATE_FMT)


def page_text(pdf, n):
    return subprocess.run(["pdftotext", "-layout", "-f", str(n), "-l", str(n), str(pdf), "-"],
                          capture_output=True, text=True).stdout


def assign_pages(pdf, st):
    """Original dataset has no page numbers: walk the pages and find each balance in order."""
    from datetime import date
    n_pages = st["summary"]["pages"]
    texts = [re.sub(r"\s+", " ", page_text(pdf, p)) for p in range(1, n_pages + 1)]
    p = 0
    for t in st["transactions"]:
        key = f'{inr(t["balance"])}'
        while p < n_pages and key not in texts[p]:
            p += 1
        if p >= n_pages:
            raise ValueError(f"could not place row {t['sr']} of {pdf.name}")
        t["page"] = p + 1
        texts[p] = texts[p].replace(key, "", 1)
        d = date.fromisoformat(t["date"])
        t["printed"] = [d.strftime(DATE_FMT[st["account"]["bank_code"]]), t["narration"],
                        inr(t["debit"]), inr(t["credit"]), inr(t["balance"])]
    a, s = st["account"], st["summary"]
    from datetime import date as D
    st["header_printed"] = {"bank": a["bank"], "holder": a["holder"].upper(), "account_no": a["account_no"],
                            "period": f'{D.fromisoformat(a["period_from"]):%d-%b-%Y} to {D.fromisoformat(a["period_to"]):%d-%b-%Y}',
                            "opening_balance": inr(s["opening_balance"]), "closing_balance": inr(s["closing_balance"])}
    return st


def load_statements():
    out = []
    for source in ("generated", "original"):
        d = ROOT / "data" / source
        gt = json.load(open(d / "ground_truth.json"))
        for fname, st in gt.items():
            pdf = d / "pdf" / fname
            if "page" not in st["transactions"][0]:
                st = assign_pages(pdf, st)
            out.append((source, pdf, st))
    return out


def pages_of(st):
    """-> list of (page_no, header|None, rows, prev_balance, gt_transactions)"""
    by_page = {}
    for t in st["transactions"]:
        by_page.setdefault(t["page"], []).append(t)
    res, prev = [], st["summary"]["opening_balance"]
    for p in sorted(by_page):
        txns = by_page[p]
        header = st["header_printed"] if p == 1 else None
        rows = [t["printed"] for t in txns]
        res.append((p, header, rows, prev,
                    [{k: t[k] for k in ("date", "narration", "debit", "credit", "balance", "txn_type")} for t in txns]))
        prev = txns[-1]["balance"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vlm")
    ap.add_argument("--train-banks", default="HDFC,SBI,PNB,KOTAK", help="comma list or 'all'")
    ap.add_argument("--aug", type=int, default=2, help="degraded copies per training page (plus 1 clean)")
    ap.add_argument("--test-pages", type=int, default=12, help="pages per test split (inference is slow)")
    ap.add_argument("--test-levels", default="clean,medium,heavy")
    ap.add_argument("--max-pixels", type=int, default=1_400_000)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    train_banks = ALL_BANKS if a.train_banks == "all" else a.train_banks.split(",")
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    stmts = load_statements()
    val_ids = set()
    for b in train_banks:
        gen = [s for s in stmts if s[0] == "generated" and s[2]["account"]["bank_code"] == b]
        val_ids |= {str(s[1]) for s in gen[:2]}

    def split_of(source, pdf, st):
        bank = st["account"]["bank_code"]
        if bank not in train_banks:
            return "test_unseen"
        if source == "original":
            return "test_seen" if a.train_banks != "all" else "val"
        return "val" if str(pdf) in val_ids else "train"

    records = {k: [] for k in ("train", "val", "test_seen", "test_unseen")}
    test_pool = {"test_seen": [], "test_unseen": []}
    for source, pdf, st in stmts:
        split = split_of(source, pdf, st)
        pages = pages_of(st)
        if split.startswith("test"):
            test_pool[split] += [(source, pdf, st, pg) for pg in pages]
            continue
        rendered = render_pages(str(pdf), dpi=a.dpi)
        for pg in pages:
            p, header, rows, prev, gtx = pg
            levels = ["clean"] + ([rng.choice(["light", "medium", "heavy"]) for _ in range(a.aug)] if split == "train" else [])
            for k, lv in enumerate(levels):
                img, _ = degrade(rendered[p - 1], lv, seed=rng.randint(0, 10**6))
                img = prepare(img, a.max_pixels)
                name = f"{pdf.stem}_p{p}_{lv}{k}.jpg"
                cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                records[split].append(dict(id=name[:-4], image=f"images/{name}", split=split, source=source,
                                           bank=st["account"]["bank_code"], file=pdf.name, page=p, level=lv,
                                           prompt=PROMPT, target=target_json(header, rows),
                                           prev_balance=prev, gt=gtx, header=header))
        print(f"{split:11} {pdf.name:48} pages={len(pages)}")

    # test: a fixed random sample of pages, each rendered at several damage levels
    for split, pool in test_pool.items():
        rng.shuffle(pool)
        for source, pdf, st, (p, header, rows, prev, gtx) in pool[:a.test_pages]:
            page = render_pages(str(pdf), dpi=a.dpi)[p - 1]
            for lv in a.test_levels.split(","):
                img, _ = degrade(page, lv, seed=rng.randint(0, 10**6))
                img = prepare(img, a.max_pixels)
                name = f"{pdf.stem}_p{p}_{lv}.jpg"
                cv2.imwrite(str(out / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                records[split].append(dict(id=name[:-4], image=f"images/{name}", split=split, source=source,
                                           bank=st["account"]["bank_code"], file=pdf.name, page=p, level=lv,
                                           prompt=PROMPT, target=target_json(header, rows),
                                           prev_balance=prev, gt=gtx, header=header))
            print(f"{split:11} {pdf.name:48} page={p}")

    for split, recs in records.items():
        with open(out / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {k: len(v) for k, v in records.items()}
    summary["train_banks"] = train_banks
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
