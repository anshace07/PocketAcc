# PocketAcc — self-verifying extraction of Indian bank statements

Turn a scanned or phone-photographed Indian bank statement into clean, structured, categorised
transactions — and know which numbers to trust.

* **Reads** the page with a small vision-language model (Qwen2-VL-2B) fine-tuned with **QLoRA** on a 6 GB GPU
* **Verifies** every row with the statement's own arithmetic: `previous balance − debit + credit = balance`
* **Corrects** misread digits (3↔8, 1↔7, dropped digits, swapped columns) and flags anything it cannot prove
* **Categorises** transactions with classical ML (TF-IDF + Linear SVM) and flags unusual ones (Isolation Forest + rules)
* **Web app**: upload a PDF or photo → table with corrected / needs-review / unusual rows, trust score, spending by category, CSV / Excel / JSON

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
| 8 | Unusual-transaction detection: Isolation Forest on per-account features + duplicate-debit and robust z-score rules | `src/anomaly.py` |
| 10 | Web app: FastAPI + dashboard + Docker | `src/pipeline.py`, `src/app.py`, `src/static/index.html`, `Dockerfile` |
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
| Balance corrector (simulation, 5% of rows with injected digit errors) | numeric field accuracy 97.5% → **99.7%**; 90.2% auto-fixed, 9.4% flagged, 0% missed, 0.4% wrong fixes |
| Transaction classifier, original data, unseen bank | Linear SVM **95.7%** accuracy (LR 94.5, NB 84.3, RF 68.8) |
| Unusual transactions (1% injected, leave-one-statement-out) | hybrid: AUC **95.6**, recall **61.9%** at 1.2% false alarms (3-sigma rule: AUC 92.6, recall 41.9%, 0.5%) |
| Deskew (heavy damage) | mean tilt 1.53° → 0.57° |
| Tesseract OCR baseline (unseen banks) | amounts 89.8% on clean pages (**98.0%** with corrector), 3.2% on heavy damage |
| Whole app on a real 9-page PDF (PNB, 592 rows, Tesseract) | 98.1% of rows verified, closing balance matches, 574/592 rows exactly right, categories 592/592 |
| Qwen2-VL-2B fine-tuning (Kaggle T4, 300 pages) | unseen banks, numeric accuracy after correction: zero-shot 14.9% → fine-tuned **61.8%** (88.5% with the tolerant parser); heavy-damage photos 60.0% / 95.6% vs Tesseract 5.1% |
| **Both readers (the app)** | keep, per page, the reading whose balance chain verifies: **93.8%** unseen banks (clean 100, medium 85.9, heavy 95.6) |

All numbers: `docs/milestone2.md`, step 9: `docs/milestone3.md`.

## Setup

```bash
git clone https://github.com/anshace07/PocketAcc.git
cd PocketAcc
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
python scripts/check_4bit_vision.py                               # CPU test: in 4-bit the vision encoder still sees the real image
python scripts/vram_test.py --data data/vlm                        # GPU: will it fit?
python src/finetune.py --data data/vlm --out runs/lora_r8 --rank 8 --epochs 1
python src/evaluate.py predict --system vlm --adapter runs/lora_r8/best_adapter --split test_unseen --limit 24 --out results/ft_unseen.jsonl
python src/evaluate.py score results/ft_unseen.jsonl --split test_unseen
python src/extract.py some_statement.pdf --adapter runs/lora_r8/best_adapter --out result.json
```

No local GPU? Use **`notebooks/pocketacc_kaggle.ipynb`** on Kaggle (free T4 GPU) — it runs every step above.

Classical ML (CPU):
```bash
python src/classifier.py data/original/ground_truth.json --out results/classifier_original.json   # step 7
python src/anomaly.py --out results/anomaly.json                                                  # step 8
python scripts/corrector_sim.py data/original/ground_truth.json results/corrector_simulation.json # steps 5-6
```

## Web app (step 10)

**On a Windows laptop (no GPU needed):**
1. Install Tesseract once: `winget install UB-Mannheim.TesseractOCR` (PDFs are rendered with PyMuPDF, no Poppler needed)
2. `pip install -r requirements.txt -r requirements-app.txt`
3. `python -m uvicorn app:app --app-dir src --port 8000` and open http://localhost:8000
4. Click the **Bank of Baroda** samples: Tesseract runs live, and the fine-tuned model's reading recorded on a Kaggle T4
   is used for the "both readers" choice. Try the heavy-damage sample with *Tesseract OCR only*, then with *Automatic*.


```bash
pip install -r requirements.txt -r requirements-app.txt
uvicorn app:app --app-dir src --port 8000          # open http://localhost:8000
python scripts/test_app.py                         # end-to-end API test
python src/pipeline.py statement.pdf --csv out.csv --xlsx out.xlsx    # same pipeline, command line
```
or with Docker (CPU, Tesseract reader):
```bash
docker build -t pocketacc .
docker run -p 8000:8000 pocketacc
```
Upload a PDF or phone photo (or click a sample). The table marks rows **corrected** (green), **needs review** (red) and
**unusual** (amber), with the reason; export CSV, Excel or JSON. On a GPU machine with the trained adapter in
`adapter/lora_r8` (in the repo), Automatic reads every page with both readers and keeps the one that verifies.

![dashboard](docs/proof/5_app_dashboard.png)

## Repository layout

```
src/        all pipeline code (each file explains itself at the top); static/ = dashboard page
scripts/    smoke_test.py, check_4bit_vision.py, vram_test.py, corrector_sim.py, test_app.py
notebooks/  Kaggle notebook for GPU training
data/       original/ and generated/ statements + ground truth
results/    experiment outputs
docs/       milestone notes and FAILURES.md (what broke and how we fixed it)
```
