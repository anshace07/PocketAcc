"""
build_demo_viewer.py - the PocketAcc dashboard as ONE offline HTML file with the saved results inside.

Takes the saved results (demo/results/*.json - fine-tuned model from the Kaggle GPU, Tesseract from any
CPU), the page images of the demo statements, the test-set scores and the loss curve, and embeds them
into the dashboard page (src/static/index.html). The file opens in any browser, on any laptop or phone,
with no Python, no server and no internet - nothing can fail during the presentation.

    python scripts/build_demo_viewer.py                      # -> demo/pocketacc_results.html
    python scripts/build_demo_viewer.py --out my_page.html
"""
import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402

ORDER = ["A_salaried_statement", "A_page1_photo_light", "A_page1_photo_medium", "A_page1_photo_heavy",
         "B_shop_statement"]


def data_uri(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved", default=str(ROOT / "demo/results"))
    ap.add_argument("--out", default=str(ROOT / "demo/pocketacc_results.html"))
    ap.add_argument("--title", default="Fine-tuned Qwen2-VL-2B vs Tesseract OCR on the demo statements.")
    a = ap.parse_args()

    files = []
    for f in Path(a.saved).glob("*.json"):
        stem, _, backend = f.stem.rpartition(".")
        files.append((ORDER.index(stem) if stem in ORDER else 99, stem, 0 if backend == "vlm" else 1, f))
    index, results, pages = [], {}, {}
    for _, _, _, f in sorted(files):
        r = json.load(open(f))
        s = r.get("saved") or {}
        if "FAKE" in str(s.get("hardware", "")):
            continue                                   # test fixtures never go into the real page
        name = f.stem
        results[name] = r
        index.append({"name": name, "file": r.get("file"), "label": s.get("label", r.get("file")),
                      "detail": s.get("detail", ""), "reader": s.get("reader", r.get("backend")),
                      "backend": r.get("backend"), "hardware": s.get("hardware"), "created": s.get("created"),
                      "rows": r["summary"]["transactions"], "trust": r["summary"]["trust_score"],
                      "answer_key": r.get("answer_key")})
        src = s.get("source")
        if src and src not in pages and (ROOT / src).is_file():
            img = pipeline.page_preview(ROOT / src, height=1600, quality=80)
            if img:
                pages[src] = data_uri(img, "image/jpeg")
    loss = ROOT / "results/finetune/loss_curve.png"
    payload = {"title": a.title, "index": index, "results": results, "pages": pages,
               "evaluation": pipeline.evaluation_summary(),
               "loss_curve": data_uri(loss.read_bytes(), "image/png") if loss.exists() else None}
    html = (ROOT / "src/static/index.html").read_text()
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("<script>", f"<script>window.POCKETACC_OFFLINE = {blob};</script>\n<script>", 1)
    html = html.replace("<title>PocketAcc – bank statement to verified data</title>",
                        "<title>PocketAcc – saved results</title>")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    readers = sorted({x["backend"] for x in index})
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB): {len(index)} results ({', '.join(readers)}), "
          f"{len(pages)} page images")


if __name__ == "__main__":
    main()
