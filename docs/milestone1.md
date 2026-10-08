# Milestone 1 — status (8 October 2026)

## Done

| Step | Deliverable | Evidence |
|---|---|---|
| 1. Data | `generate.py`: 7 bank layouts, 5 spending profiles, page-level ground truth. 140 new statements (460 pages, 15,368 transactions) + the original 20 (110 pages, 4,774 transactions) | 0 balance-chain errors across 20,142 transactions |
| 2. Damage | `degrade.py`: desk background, rotation, perspective, blur, shadow, stamp, noise, JPEG at light / medium / heavy | `results/` figures, dataset images |
| 3. Pre-processing | `preprocess.py`: paper crop (Otsu threshold + largest contour), deskew (Hough transform), resize to the 28-px patch grid | heavy damage: mean tilt 1.53° → 0.57° |
| 4. VLM pipeline | `build_dataset.py` (690 train / 24 val / 36 + 36 test images), `finetune.py` (QLoRA), `extract.py`, `evaluate.py`, Kaggle notebook | `scripts/smoke_test.py` passes: label masking, LoRA gradients, generation, scoring |
| Baseline | `ocr_baseline.py`: Tesseract OCR + rule-based parser | numbers below |
| 5–6. Corrector | `corrector.py` | simulation + real OCR errors below |
| 7. Classifier | `classifier.py`: LR / NB / Linear SVM / RF | numbers below |

## Traditional OCR baseline (Tesseract) — per damage level

Numeric field accuracy = debit, credit and balance exactly right.

| Test split | Damage | Rows found | Numeric acc. | + balance corrector |
|---|---|---|---|---|
| unseen banks (ICICI, BOB, Axis) | clean | 100.0% | 89.8% | **97.4%** |
| | medium | 79.3% | 62.8% | 63.5% |
| | heavy | 5.6% | 3.2% | 3.2% |
| seen banks (original statements) | clean | 100.0% | 99.2% | 99.2% |
| | medium | 37.4% | 27.8% | 27.7% |
| | heavy | 1.8% | 0.9% | 0.9% |

**Reading:** classical OCR is fine on clean digital PDFs and collapses on photographed / damaged
pages — this is the gap the fine-tuned vision-language model must close. On clean pages the
balance corrector already repairs real OCR mistakes (89.8% → 97.4% on unseen banks, mostly
debit/credit column confusions). When whole rows are missing, the corrector cannot help — it flags them.

## Balance corrector — simulation (original 20 statements, mean of 5 seeds)

| Rows with injected errors | Accuracy before | After | Auto-fixed | Flagged | Missed | Wrong fix |
|---|---|---|---|---|---|---|
| 1% | 99.46% | 99.91% | 86.5% | 13.5% | 0.0% | 0.0% |
| 5% | 97.48% | 99.62% | 86.6% | 12.6% | 0.0% | 0.8% |
| 10% | 95.00% | 99.02% | 82.7% | 15.6% | 0.1% | 1.6% |
| 20% | 89.97% | 97.04% | 73.6% | 23.8% | 0.2% | 2.4% |

## Transaction classifier (TF-IDF char n-grams + amount + direction)

| Model | Original data: random split | Original data: unseen bank | Generated data: unseen bank |
|---|---|---|---|
| Logistic Regression | 99.50% | 94.45% | 99.32% |
| Naive Bayes | 99.43% | 84.25% | 94.54% |
| **Linear SVM** | **99.64%** | **95.68%** | **99.36%** |
| Random Forest | 99.41% | 68.75% | 96.36% |

## Next (needs GPU)

1. `scripts/vram_test.py` on the RTX 3050 (or Kaggle T4)
2. zero-shot Qwen2-VL-2B baseline on the same test pages
3. QLoRA fine-tuning (rank 8, 1 epoch) → evaluate → compare against the Tesseract table above
