"""
evaluate.py - STEP 4d: score any extraction system against the ground truth.

Two sub-commands:
  predict  run a system over a test split and save its raw outputs (resumable)
           systems: vlm (base or fine-tuned Qwen2-VL)  |  tesseract (traditional OCR baseline)
  score    compare predictions with ground truth, before and after balance correction

Metrics (all per page, then averaged / pooled):
  * row recall / precision      - did we find the right number of transactions?
  * field accuracy              - date, debit, credit, balance exactly right (after alignment)
  * numeric field accuracy      - amounts + balances only (what the corrector can repair)
  * narration CER               - character error rate of the narration text
  * chain-verified rows         - % of predicted rows whose balance arithmetic checks out
  * the same numbers AFTER running the balance corrector

Usage:
    python src/evaluate.py predict --system vlm --adapter runs/lora_r8/best_adapter --split test_unseen --out results/ft_unseen.jsonl
    python src/evaluate.py predict --system tesseract --split test_unseen --out results/tess_unseen.jsonl
    python src/evaluate.py score results/ft_unseen.jsonl --split test_unseen
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from common import parse_model_output, rows_to_transactions
from corrector import Txn, correct, verify

NUM_FIELDS = ("debit", "credit", "balance")


# ------------------------------------------------------------------ text metrics
def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    return levenshtein(pred, gt) / max(len(gt), 1)


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 0.005


def row_score(p, g):
    s = sum(_eq(p[k], g[k]) for k in NUM_FIELDS) + (p["date"] == g["date"])
    return s + (1 - min(1.0, cer(p["narration"], g["narration"])))


def align(pred, gt):
    """Needleman-Wunsch style alignment of predicted rows to ground-truth rows."""
    n, m = len(pred), len(gt)
    S = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            S[i][j] = max(S[i - 1][j], S[i][j - 1], S[i - 1][j - 1] + row_score(pred[i - 1], gt[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if S[i][j] == S[i - 1][j - 1] + row_score(pred[i - 1], gt[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif S[i][j] == S[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def score_page(pred, gt):
    pairs = align(pred, gt)
    matched = {j: pred[i] for i, j in pairs}
    c = defaultdict(float)
    c["gt_rows"], c["pred_rows"], c["matched"] = len(gt), len(pred), len(pairs)
    for j, g in enumerate(gt):
        p = matched.get(j)
        for k in NUM_FIELDS + ("date",):
            c[f"ok_{k}"] += bool(p) and (_eq(p[k], g[k]) if k != "date" else p[k] == g[k])
        c["narration_cer"] += cer(p["narration"], g["narration"]) if p else 1.0
        c["row_exact"] += bool(p) and all(_eq(p[k], g[k]) for k in NUM_FIELDS) and p["date"] == g["date"] \
            and p["narration"] == g["narration"]
    return c


def chain_ok_frac(prev_balance, txns):
    if not txns:
        return 0.0
    chain = [Txn(t["debit"], t["credit"], t["balance"] if t["balance"] is not None else 0.0) for t in txns]
    return sum(verify(prev_balance, chain)) / len(chain)


def apply_corrector(prev_balance, txns):
    chain = [Txn(t["debit"], t["credit"], t["balance"] if t["balance"] is not None else 0.0) for t in txns]
    if chain:
        correct(prev_balance, chain)
    out = []
    for t, c in zip(txns, chain):
        out.append(dict(t, debit=c.debit, credit=c.credit, balance=c.balance))
    return out


def summarise(counts):
    g = max(counts["gt_rows"], 1)
    return {
        "pages": int(counts["pages"]),
        "row_recall": round(100 * counts["matched"] / g, 2),
        "row_precision": round(100 * counts["matched"] / max(counts["pred_rows"], 1), 2),
        "date_acc": round(100 * counts["ok_date"] / g, 2),
        "debit_acc": round(100 * counts["ok_debit"] / g, 2),
        "credit_acc": round(100 * counts["ok_credit"] / g, 2),
        "balance_acc": round(100 * counts["ok_balance"] / g, 2),
        "numeric_field_acc": round(100 * (counts["ok_debit"] + counts["ok_credit"] + counts["ok_balance"]) / (3 * g), 2),
        "narration_cer": round(100 * counts["narration_cer"] / g, 2),
        "row_exact": round(100 * counts["row_exact"] / g, 2),
        "chain_verified_rows": round(100 * counts["chain_ok"] / max(counts["pred_rows"], 1), 2),
        "json_ok": round(100 * counts["json_ok"] / max(counts["pages"], 1), 2),
    }


def score(pred_file, data_dir, split):
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(Path(data_dir) / f"{split}.jsonl")}
    preds = [json.loads(l) for l in open(pred_file)]
    groups = defaultdict(lambda: defaultdict(float))
    for pr in preds:
        r = recs[pr["id"]]
        parsed = parse_model_output(pr["raw"]) if "raw" in pr else {"rows": pr["rows"]}
        txns = rows_to_transactions(parsed.get("rows", []))
        gt = r["gt"]
        for stage, t in (("before", txns), ("after", apply_corrector(r["prev_balance"], txns))):
            c = score_page(t, gt)
            c["chain_ok"] = chain_ok_frac(r["prev_balance"], t) * len(t)
            c["json_ok"] = float(bool(parsed.get("rows")))
            c["pages"] = 1
            for key in (f"{stage}|all", f"{stage}|level={r['level']}", f"{stage}|bank={r['bank']}"):
                for k, v in c.items():
                    groups[key][k] += v
    result = {k: summarise(v) for k, v in sorted(groups.items())}
    return result


def print_table(result):
    cols = ["pages", "row_recall", "date_acc", "numeric_field_acc", "balance_acc", "narration_cer", "row_exact",
            "chain_verified_rows"]
    print(f"{'group':28}" + "".join(f"{c[:14]:>15}" for c in cols))
    for k, v in result.items():
        print(f"{k:28}" + "".join(f"{v[c]:>15}" for c in cols))


# ------------------------------------------------------------------ predict
def predict(args):
    from PIL import Image
    data = Path(args.data)
    recs = [json.loads(l) for l in open(data / f"{args.split}.jsonl")]
    if args.limit:
        recs = recs[:args.limit]
    done = set()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out)}
    if args.system == "vlm":
        from extract import Extractor
        ex = Extractor(args.model, args.adapter, not args.no_4bit, args.max_pixels, args.max_new_tokens)
    with open(out, "a") as f:
        for k, r in enumerate(recs, 1):
            if r["id"] in done:
                continue
            img = Image.open(data / r["image"]).convert("RGB")
            if args.system == "vlm":
                raw, _, secs = ex.read_image(img)
                row = {"id": r["id"], "raw": raw, "seconds": round(secs, 1)}
            else:
                from ocr_baseline import ocr_page
                rows, secs = ocr_page(img, r["bank"])
                row = {"id": r["id"], "rows": rows, "seconds": round(secs, 1)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(recs)}] {r['id']}  {row['seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--system", choices=["vlm", "tesseract"], default="vlm")
    p.add_argument("--data", default="data/vlm")
    p.add_argument("--split", default="test_unseen")
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-pixels", type=int, default=1_400_000)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("pred_file")
    s.add_argument("--data", default="data/vlm")
    s.add_argument("--split", default="test_unseen")
    s.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if a.cmd == "predict":
        predict(a)
    else:
        res = score(a.pred_file, a.data, a.split)
        print_table(res)
        if a.json_out:
            json.dump(res, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
