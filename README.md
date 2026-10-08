# PocketAcc — self-verifying extraction of Indian bank statements

Turn a scanned or phone-photographed Indian bank statement into clean, structured, categorised
transactions — and know which numbers to trust.

* **Reads** the page with a small vision-language model (Qwen2-VL-2B) fine-tuned with **QLoRA** on a 6 GB GPU
* **Verifies** every row with the statement's own arithmetic: `previous balance − debit + credit = balance`
* **Corrects** misread digits (3↔8, 1↔7, dropped digits, swapped columns) and flags anything it cannot prove
* **Categorises** transactions with classical ML (TF-IDF + Linear SVM) and flags anomalies

Mini-project, CSET312, Bennett University — Ansh Singh, Navya Khandelwal.

---

## Pipeline

| Step | What | Code |
|---|---|---|
| 1 | Synthetic statements for 7 banks with page-level ground truth | `src/generate.py` |
| 2 | Domain-randomised damage (desk, rotation, perspective, blur, shadow, stamp, noise, JPEG) | `src/degrade.py` |
| 3 | Classical pre-processing: paper crop (Otsu + contours), deskew (Hough), resize to the VLM patch grid | `src/preprocess.py` |
| 4 | Page → JSON dataset, QLoRA fine-tuning, extraction, evaluation | `src/build_dataset.py`, `src/finetune.py`, `src/extract.py`, `src/evaluate.py` |
| 5–6 | Balance-chain verification and correction | `src/corrector.py` |
| 7 | Transaction classifier (LR / NB / Linear SVM / RF) | `src/classifier.py` |
| — | Traditional baseline: Tesseract OCR + rule-based parser | `src/ocr_baseline.py` |

## Data

| Set | Statements | Pages | Transactions | Notes |
|---|---|---|---|---|
| `data/original` | 20 | 110 | 4,774 | first hand-built set, 7 banks, 21 categories |
| `data/generated` | 140 | 460 | 15,368 | `generate.py --n 140 --seed 7`, 20 per bank, page numbers recorded |

All statements are synthetic — no real customer data. Every balance chain in both sets is verified (0 errors).

**Experiment split** (`build_dataset.py`): train on HDFC, SBI, PNB, Kotak (generated, clean + 2 damaged copies per page);
`test_seen` = the original hand-made statements of those banks; `test_unseen` = ICICI, Bank of Baroda, Axis — never seen in training.

## Results so far

| Component | Result |
|---|---|
| Balance corrector (simulation, 5% of rows with injected digit errors) | numeric field accuracy 97.5% → **99.6%**; 86.5% auto-fixed, 12.8% flagged, 0% missed |
| Transaction classifier, original data, unseen bank | Linear SVM **95.7%** accuracy (LR 94.5, NB 84.3, RF 68.8) |
| Deskew (heavy damage) | mean tilt 1.53° → 0.57° |
| Tesseract OCR baseline (unseen banks) | amounts 89.8% on clean pages (97.4% with corrector), 3.2% on heavy damage |
| Qwen2-VL-2B fine-tuning | **in progress** — needs a GPU, see below |

## Setup

```bash
git clone https://github.com/anshace07/pocketacc.git
cd pocketacc
pip install -r requirements.txt
```
System tools: **Poppler** (for `pdf2image`) and, only for the OCR baseline, **Tesseract**.
Windows: download Poppler for Windows and add its `bin` folder to PATH; install Tesseract from the UB Mannheim build.

GPU (fine-tuning / inference):
```bash
# PyTorch with CUDA first - pick the command for your CUDA version at pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
```

## Run

```bash
python src/generate.py --n 140 --out data/generated --seed 7      # step 1 (already done, PDFs in repo)
python src/build_dataset.py --out data/vlm                         # steps 2-4a: page images + JSON targets (~20 min)
python scripts/smoke_test.py data/vlm                              # CPU test of the training code
python scripts/vram_test.py --data data/vlm                        # GPU: will it fit?
python src/finetune.py --data data/vlm --out runs/lora_r8 --rank 8 --epochs 1
python src/evaluate.py predict --system vlm --adapter runs/lora_r8/best_adapter --split test_unseen --limit 24 --out results/ft_unseen.jsonl
python src/evaluate.py score results/ft_unseen.jsonl --split test_unseen
python src/extract.py some_statement.pdf --adapter runs/lora_r8/best_adapter --out result.json
```

No local GPU? Use **`notebooks/pocketacc_kaggle.ipynb`** on Kaggle (free T4 GPU) — it runs every step above.

## Repository layout

```
src/        all pipeline code (each file explains itself at the top)
scripts/    smoke_test.py, vram_test.py, corrector_sim.py
notebooks/  Kaggle notebook for GPU training
data/       original/ and generated/ statements + ground truth
results/    experiment outputs
docs/       milestone notes and FAILURES.md (what broke and how we fixed it)
```
