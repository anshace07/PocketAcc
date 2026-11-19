"""
score_step9.py - STEP 9: final comparison of the readers on the same 24 test pages per split.

Needs: data/vlm (python src/build_dataset.py) and the Kaggle outputs in results/
(tess_*.jsonl, base_*.jsonl, ft_*.jsonl). No GPU: the model's outputs were recorded on Kaggle.

Systems
  tess            Tesseract + rule parser (traditional baseline)
  base / ft       zero-shot / fine-tuned Qwen2-VL-2B, scored with the STRICT parser (as measured on Kaggle)
                  and the TOLERANT parser (same outputs; repairs broken JSON and odd column counts -
                  designed after looking at these outputs, so treat as optimistic)
  both            the app's reader: each page read by Tesseract and by the fine-tuned model, keep the reading
                  whose balance chain verifies best after correction (no answer key used for the choice)

    python scripts/score_step9.py          -> results/step9.json, results/hybrid_*.jsonl, printed table
"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import score  # noqa: E402
from pipeline import read_both  # noqa: E402

LIMIT = 24
DATA = ROOT / "data/vlm"
RES = ROOT / "results"


def both_predictions(split):
    recs = [json.loads(l) for l in open(DATA / f"test_{split}.jsonl")][:LIMIT]
    ft = {json.loads(l)["id"]: json.loads(l) for l in open(RES / f"ft_{split}.jsonl")}
    picks = {}
    with open(RES / f"hybrid_{split}.jsonl", "w") as f:
        for r in recs:
            cache = {"raw": [ft[r["id"]]["raw"]], "seconds": [ft[r["id"]]["seconds"]]}
            _, rows, _, info = read_both([cv2.imread(str(DATA / r["image"]))], None, cache)
            picks[r["id"]] = info[0]["reader"]
            f.write(json.dumps({"id": r["id"], "rows": rows, "reader": info[0]["reader"],
                                "verified": info[0]["verified"]}) + "\n")
            print(f"  {r['id'][:44]:44} -> {info[0]['reader']:9} {info[0]['verified']}", flush=True)
    return picks


def main():
    out = {"pages_per_split": LIMIT, "metric": "numeric field accuracy (%) after balance correction", "splits": {}}
    for split in ["unseen", "seen"]:
        print(f"[{split}] both readers, page by page:")
        picks = both_predictions(split)
        rows = {}
        for name, file, strict in [("Tesseract OCR", "tess", True), ("Zero-shot Qwen2-VL (strict)", "base", True),
                                   ("Zero-shot Qwen2-VL (tolerant)", "base", False),
                                   ("Fine-tuned (strict)", "ft", True), ("Fine-tuned (tolerant)", "ft", False),
                                   ("Both readers (app)", "hybrid", False)]:
            r = score(str(RES / f"{file}_{split}.jsonl"), str(DATA), f"test_{split}", LIMIT, strict)
            rows[name] = {lvl: {"before": r[f"before|level={lvl}"]["numeric_field_acc"],
                                "after": r[f"after|level={lvl}"]["numeric_field_acc"],
                                "rows_found": r[f"after|level={lvl}"]["row_recall"]} for lvl in ["clean", "medium", "heavy"]}
            rows[name]["all"] = {"before": r["before|all"]["numeric_field_acc"], "after": r["after|all"]["numeric_field_acc"],
                                 "rows_found": r["after|all"]["row_recall"]}
        out["splits"][split] = {"systems": rows, "both_picked_model_on": sum(p == "vlm" for p in picks.values())}
    json.dump(out, open(RES / "step9.json", "w"), indent=1)
    for split, d in out["splits"].items():
        print(f"\n== test_{split} (24 pages: 8 clean, 8 medium, 8 heavy) - numeric accuracy after correction ==")
        print(f"{'system':32}{'clean':>8}{'medium':>8}{'heavy':>8}{'all':>8}{'rows found':>12}")
        for name, v in d["systems"].items():
            print(f"{name:32}" + "".join(f"{v[l]['after']:8.1f}" for l in ["clean", "medium", "heavy", "all"]) + f"{v['all']['rows_found']:12.1f}")
        print(f"both readers chose the model on {d['both_picked_model_on']} of 24 pages")
    print("\nsaved results/step9.json")


if __name__ == "__main__":
    main()
