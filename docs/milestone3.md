# Milestone 3 — step 9: the trained model (19 November 2026)

## Training run (Kaggle, Tesla T4 x2, notebook `notebooks/pocketacc_kaggle.ipynb`)

| Setting | Value | Why |
|---|---|---|
| Model | Qwen2-VL-2B-Instruct, 4-bit NF4 language model, vision encoder fp16 | 4-bit image bug fixed (milestone 2) |
| LoRA | rank 8, alpha 16, q/k/v/o projections | |
| Images | at most 900,000 pixels | 1.4 MP ran out of memory in the training step on a 15 GB T4 |
| Data | 300 of the 690 training pages, 1 epoch, 37 optimizer steps (grad. accumulation 8) | 40 s per training step-page; whole run had to fit Kaggle's 12-hour limit |
| Memory / time | peak 14.1 GB; training 1 h 50 min | `results/training_log.csv` |
| Loss | train 0.135 → 0.071; best validation loss 0.060 | `results/loss_curve.png` |
| Adapter | `adapter/lora_r8` (8.4 MB) | |
| Reading speed | ~2.5 min per page on a T4 (up to 4.5 min on the densest pages) | |

## Results — numeric field accuracy after balance correction (%), same 24 test pages per split (8 clean, 8 medium, 8 heavy)

Unseen banks (ICICI, Bank of Baroda, Axis — never in training):

| System | Clean | Medium | Heavy | All | Rows found |
|---|---|---|---|---|---|
| Tesseract OCR + rule parser | **100.0** | 72.6 | 5.1 | 59.2 | 64.0 |
| Zero-shot Qwen2-VL-2B | 22.0 | 5.7 | 17.0 | 14.9 | 39.1 |
| Fine-tuned Qwen2-VL-2B | 78.0 | 47.3 | 60.0 | 61.8 | 71.9 |
| Fine-tuned, tolerant parser ¹ | 84.2 | 85.6 | **95.6** | 88.5 | 91.6 |
| **Both readers (the app)** ² | **100.0** | **85.9** | **95.6** | **93.8** | **94.5** |

Seen banks (hand-built statements of HDFC, SBI, PNB, Kotak — a different look from the generated training pages):

| System | Clean | Medium | Heavy | All | Rows found |
|---|---|---|---|---|---|
| Tesseract OCR + rule parser | **100.0** | 27.1 | 1.1 | 42.7 | 46.2 |
| Zero-shot Qwen2-VL-2B | 5.2 | 0.0 | 1.3 | 2.2 | 2.8 |
| Fine-tuned Qwen2-VL-2B | 44.6 | 54.8 | 44.6 | 48.0 | 79.8 |
| Fine-tuned, tolerant parser ¹ | 67.7 | 61.5 | 58.5 | 62.6 | 91.6 |
| **Both readers (the app)** ² | **100.0** | **68.4** | **58.6** | **75.7** | **96.8** |

¹ Same model outputs, more forgiving parser: repairs broken JSON (missing or stray brackets), rows flattened into one
list, and rows with extra or missing columns (amount goes to debit, the balance chain decides the direction). It was
designed **after** looking at these test outputs, so these numbers are optimistic until confirmed on new pages.
² Each page read by both; the reading whose balance chain verifies best after correction is kept. The choice uses only
the statement's own arithmetic, never the answer key. Chose the model on 16 of 24 pages in each split.

Reproduce (no GPU): `python scripts/score_step9.py` → `results/step9.json`.

## What the numbers say

1. **Fine-tuning works:** unseen banks 14.9% → 61.8% (strict); the zero-shot model mostly ignores the output format.
2. **The model's errors are mostly format, not reading:** it reads the digits but sometimes breaks the JSON, shifts
   columns, or (on the densest pages, 68–85 rows) runs past the output limit or repeats rows.
3. **The two readers fail in opposite places:** Tesseract is perfect on clean pages and collapses on photos (5.1% heavy);
   the fine-tuned model is the reverse. The balance chain tells which one to trust, page by page → 93.8% on unseen banks.
4. **Seen banks score lower than unseen banks** because the seen-bank test pages come from the hand-built statements,
   which look different from the generated training pages; the model learned the generator's style.
5. **Limits:** only 300 training pages and 1 epoch; ~2.5 min per page on a T4; 24 test pages per split (small sample).
