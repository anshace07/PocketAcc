"""
run_demo.py - run the full PocketAcc pipeline on the demo statements and save every result.

The saved results are what the web app shows under "Saved results" and what
scripts/build_demo_viewer.py turns into a single offline page - so the fine-tuned model's
output (computed on a Kaggle GPU) can be shown on any laptop, instantly.

    python scripts/run_demo.py --backend vlm          # on a GPU with the trained adapter (Kaggle)
    python scripts/run_demo.py --backend tesseract    # on any CPU machine
    python scripts/run_demo.py --backend vlm --files demo/samples/A_page1_photo_heavy.jpg

Output: demo/results/<file stem>.<backend>.json  (the same JSON the app returns, plus a 'saved' block
and, for the demo statements, an 'answer_key' check against the generator's ground truth).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402

DEMO_FILES = [  # (file, label, detail) - in the order the demo walks through them
    ("A_salaried_statement.pdf", "Salaried customer · digital PDF", "2 pages · 49 transactions · a bank the model never saw"),
    ("A_page1_photo_light.jpg", "Page 1 as a phone photo · light damage", "slight tilt, shadow, noise"),
    ("A_page1_photo_medium.jpg", "Page 1 as a phone photo · medium damage", "tilt, blur, stamp, shadow, JPEG"),
    ("A_page1_photo_heavy.jpg", "Page 1 as a phone photo · heavy damage", "strong tilt, blur, stamp, noise, low JPEG quality"),
    ("B_shop_statement.pdf", "Kirana-shop owner · digital PDF", "3 pages · 102 transactions · large business payments"),
]
READER = {"vlm": "Fine-tuned Qwen2-VL-2B", "tesseract": "Tesseract OCR"}


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vlm", "tesseract"], required=True)
    ap.add_argument("--files", nargs="*", help="default: every demo statement in demo/samples")
    ap.add_argument("--out", default=str(ROOT / "demo/results"))
    a = ap.parse_args()
    if a.backend == "vlm" and not pipeline.vlm_available():
        sys.exit(f"The vision model needs a CUDA GPU and the adapter ({pipeline.ADAPTER}).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {f: (lab, det) for f, lab, det in DEMO_FILES}
    files = [Path(f) for f in a.files] if a.files else [ROOT / "demo/samples" / f for f, _, _ in DEMO_FILES]
    models = pipeline.load_models()
    hw = gpu_name()
    for path in files:
        t0 = time.time()
        r = pipeline.analyse(path, a.backend, None, models)
        pipeline.attach_answer_key(r, path.name)
        label, detail = meta.get(path.name, (path.stem, ""))
        r["saved"] = {"label": label, "detail": detail, "reader": READER[a.backend], "backend": a.backend,
                      "source": f"demo/samples/{path.name}", "hardware": hw,
                      "created": datetime.now().isoformat(timespec="seconds")}
        if a.backend == "vlm":
            four_bit, max_pixels = pipeline.vlm_settings()
            r["saved"]["settings"] = {"adapter": Path(pipeline.ADAPTER).name, "four_bit": four_bit,
                                      "max_pixels": max_pixels}
        dst = out / f"{path.stem}.{a.backend}.json"
        json.dump(r, open(dst, "w"), indent=1, ensure_ascii=False)
        s, k = r["summary"], r.get("answer_key")
        print(f"{path.name:28} {a.backend:9} {time.time() - t0:6.1f}s | rows {s['transactions']:3} | trust "
              f"{s['trust_score']}% | fixed {s['auto_corrected']} | review {s['needs_review']} | unusual {s['anomalies']}"
              + (f" | answer key: {k['numbers_right']}/{k['answer_rows']} amounts right, "
                 f"{k['fully_right']} fully right" if k else ""), flush=True)


if __name__ == "__main__":
    main()
