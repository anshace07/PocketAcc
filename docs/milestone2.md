# Milestone 2 — status (18 November 2026)

## What changed since milestone 1

| Step | Deliverable | Evidence |
|---|---|---|
| 4 (fix) | 4-bit loading bug found before the first GPU run: the page image reached the vision encoder as 4 integer values. Vision encoder now kept out of 4-bit | `scripts/check_4bit_vision.py` (reproduces the bug, verifies the fix) |
| 5–6 (better) | Corrector fills missing values, accepts a one-digit balance fix only when two rows confirm it, never repairs from a failed balance | `results/corrector_simulation.json`, FAILURES.md |
| 8 | Unusual-transaction detection: Isolation Forest + two rules | `src/anomaly.py`, `results/anomaly.json` |
| 10 | Web app: FastAPI API, dashboard, CSV / Excel / JSON export, Dockerfile | `src/app.py`, `src/pipeline.py`, `scripts/test_app.py`, `docs/proof/5–7_*.png` |
| Kaggle | Notebook stops on any failed step, quick check first, both T4 GPUs used, all systems scored on the same 24 pages | `notebooks/pocketacc_kaggle.ipynb` |

**Still needs the GPU:** QLoRA fine-tuning of Qwen2-VL-2B and its evaluation (step 4 run, step 9).

## Balance corrector — simulation (original 20 statements, mean of 5 seeds)

| Rows with injected errors | Accuracy before | After | Auto-fixed | Flagged | Missed | Wrong fix | Clean rows damaged |
|---|---|---|---|---|---|---|---|
| 1% | 99.46% | 99.96% | 91.9% | 8.1% | 0.0% | 0.0% | 0.0 |
| 2% | 99.03% | 99.90% | 89.9% | 9.9% | 0.0% | 0.2% | 0.0 |
| **5%** | **97.48%** | **99.74%** | **90.2%** | **9.4%** | **0.0%** | **0.4%** | **0.0** |
| 10% | 95.00% | 99.21% | 85.2% | 13.7% | 0.1% | 1.0% | 0.2 |
| 20% | 89.97% | 97.23% | 74.0% | 24.0% | 0.2% | 1.8% | 0.4 |

Milestone 1 (old rules) at 5%: 99.62% after, 86.4% auto-fixed, 0.7% wrong fixes, 2.2 clean rows damaged per run.

## Tesseract OCR baseline + corrector (numeric field accuracy, 12 pages per cell)

| Test split | Damage | Rows found | Before | After corrector |
|---|---|---|---|---|
| unseen banks (ICICI, BOB, Axis) | clean | 100.0% | 89.8% | **98.0%** |
| | medium | 79.3% | 62.8% | 64.5% |
| | heavy | 5.6% | 3.2% | 3.2% |
| seen banks (original statements) | clean | 100.0% | 99.2% | **100.0%** |
| | medium | 37.4% | 27.8% | 28.2% |
| | heavy | 1.8% | 0.9% | 0.9% |

The Kaggle comparison against the VLM uses the first 24 pages of each split for every system
(`results/score24_*.json`), because the VLM is evaluated on 24 pages.

## Step 8 — unusual transactions

Leave-one-statement-out on the 20 six-month original statements: fit on 19 clean statements (no labels),
test on the 20th with ~1% injected anomalies of 4 kinds; mean of 5 seeds (270 anomalies in 24,140 rows).
Categories come from the Linear SVM trained on the generated data (98.3% correct on these statements).

| Method | AUC | Avg. precision | Precision | Recall | False alarms | Spike | Big transfer | Duplicate | Cash deposit |
|---|---|---|---|---|---|---|---|---|---|
| 3-sigma amount rule (per account) | 92.6 | 44.0 | **46.5** | 41.9 | **0.5%** | 13.9 | 74.5 | 0.0 | 75.0 |
| Isolation Forest, raw features | 69.6 | 4.6 | 12.2 | 7.0 | 0.6% | 0.0 | 2.7 | 0.0 | 80.0 |
| Isolation Forest, account features | 90.6 | 21.0 | 23.6 | 27.0 | 1.0% | 0.0 | 47.3 | 4.0 | 100 |
| **Hybrid: forest + 2 rules (deployed)** | **95.6** | **61.7** | 36.2 | **61.9** | 1.2% | **25.2** | **84.5** | **100** | **100** |

Reading: account context is what makes the forest work (AUC 69.6 → 90.6). The forest alone cannot
see duplicate debits (that feature is constant in clean training data) and is weak on single-feature
spikes, so two transparent rules cover those. The hybrid catches the most anomalies and ranks them
best; the price is more false alarms than the plain rule (1.2% vs 0.5% of normal rows).
Amount spikes stay hard (25%): everyday UPI spending varies so much that 3–10× a past maximum is often
still inside the account's normal spread.

## Step 10 — the app on real files (CPU, Tesseract reader)

| Statement | Rows | Verified | Auto-corrected | Needs review | Exactly right vs ground truth |
|---|---|---|---|---|---|
| ICICI PDF, 4 pages (unseen bank) | 168 | 100% | 81 (column swaps + 1 missing amount) | 0 | 168 / 168 |
| PNB PDF, 9 pages (original) | 592 | 98.1% | 20 | 11 | 574 / 592 (2 wrong fixes) |
| ICICI phone photo, light damage | 35 | 100% | — | 0 | — |
| ICICI phone photo, medium damage | 34 | 76.5% → low-trust warning | 5 | 8 | — |

Known limit: when OCR misreads a *run* of consecutive balances by the same amount, every row inside
the run still adds up, so the chain cannot see it (PNB: 2 wrong fixes at the edge of such a run).
