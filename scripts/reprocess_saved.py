"""
reprocess_saved.py - redo every step AFTER reading for saved vision-model results, on CPU.

Each saved result (demo/results/*.vlm.json, made by scripts/run_demo.py on a GPU) keeps the model's raw
text for every page. This script parses that text again with the current code - tolerant parsing,
row repair, balance-chain corrector, classifier, anomaly detector, answer-key check - so improvements to
the post-processing reach the saved results without another GPU run. The reading itself is unchanged.

    python scripts/reprocess_saved.py                     # every demo/results/*.vlm.json
    python scripts/reprocess_saved.py path/to/x.vlm.json  # one file
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402
from common import parse_model_output_tolerant  # noqa: E402


def reprocess(r, models):
    header, rows, info = None, [], []
    for p in r["pages"]:
        parsed = parse_model_output_tolerant(p.get("raw", ""))
        page_header, page_rows = parsed.get("header"), parsed.get("rows")
        page_rows = [x for x in page_rows if isinstance(x, (list, tuple))] if isinstance(page_rows, list) else []
        header = header or (page_header if isinstance(page_header, dict) else None)
        rows += page_rows
        info.append(dict(p, rows=len(page_rows)))
    header = header or {}
    bank = pipeline.detect_bank(["IFSC " + str(header.get("ifsc", "")), str(header.get("bank", ""))])
    new = pipeline.build_result(r["file"], "vlm", header, rows, bank, info, models)
    new["seconds"] = r["seconds"]                   # the original reading time on the GPU
    pipeline.attach_answer_key(new, r["file"])
    new["saved"] = dict(r.get("saved") or {}, reprocessed=datetime.now().isoformat(timespec="seconds"))
    return new


def main():
    files = [Path(f) for f in sys.argv[1:]] or sorted((ROOT / "demo/results").glob("*.vlm.json"))
    models = pipeline.load_models()
    for f in files:
        old = json.load(open(f))
        new = reprocess(old, models)
        json.dump(new, open(f, "w"), indent=1, ensure_ascii=False)
        o, n = old.get("answer_key") or {}, new.get("answer_key") or {}
        print(f"{f.name:34} rows {old['summary']['transactions']:3} -> {new['summary']['transactions']:3} | trust "
              f"{old['summary']['trust_score']} -> {new['summary']['trust_score']}"
              + (f" | amounts right {o.get('numbers_right')} -> {n.get('numbers_right')} of {n.get('answer_rows')}"
                 if n else ""))


if __name__ == "__main__":
    main()
