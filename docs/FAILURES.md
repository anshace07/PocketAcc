# FAILURES.md — what broke, and what we changed

One line per problem: what we expected, what happened, what we did. This is the honest
history of the project (and the best viva preparation there is).

| Date | Expected | What actually happened | Fix |
|---|---|---|---|
| Sep 2026 | 8 GB laptop GPU | `nvidia-smi` showed RTX 3050 **6 GB**, only **5.1 GB free** (display + Windows reserve ~0.9 GB) | Chose the 2B model, 4-bit QLoRA, LoRA rank 8, gradient checkpointing, batch size 1, `--max-pixels` knob |
| Oct 2026 | Original 20 statements usable for page-level training | Ground truth had no page numbers — the model is trained page by page | `build_dataset.assign_pages()` recovers each row's page from the PDF text layer by matching balances in order |
| Oct 2026 | 20 statements (110 pages) enough to fine-tune | Far too few pages for a VLM; no generator code existed to make more | Wrote `generate.py`: 7 bank layouts, 5 spending profiles, page-level ground truth → 140 statements / 460 pages |
| Oct 2026 | `HoughLinesP` returns `(N,1,4)` | Newer OpenCV returned `(N,4)` → `cannot unpack non-iterable int32` | `lines.reshape(-1, 4)` |
| Oct 2026 | Deskew removes all tilt | Residual tilt ≈0.3–0.6° remains: perspective warp is not a pure rotation | Accepted; reported honestly (1.53° → 0.57° on heavy damage). Training on damaged copies makes the model robust to the rest |
| Oct 2026 | OCR baseline finds the table header | It matched the **summary boxes** ("TOTAL WITHDRAWALS … CLOSING BALANCE") first → debit/credit/balance columns wrong, balance accuracy 0% | Header line must contain no amounts and have debit < credit < balance from left to right |
| Oct 2026 | OCR baseline narration text | Words were glued together ("PENSIONDISBURSINGAUTH") | Join words within a line with spaces; join wrapped lines without |
| Oct 2026 | Generator never crashes | Empty counterparty name for some transaction types → `IndexError` in UPI handle builder | Fall back to a random handle when the name is empty |
| Oct 2026 | Tiny-model smoke test can over-fit one page | Loss only fell 6.35 → 6.23: a random 0.2M-parameter model cannot memorise 1,400 tokens | Changed the check to: LoRA gradient norm > 0 and loss falls; real capacity is tested on GPU |
| Oct 2026 | Corrector simulation is deterministic | Numbers moved by ±0.1 between runs (set iteration order changes with Python's hash seed) | Iterate the confusion pairs in sorted order |
| Oct 2026 | Classifier generalises equally on generated data | Unseen-bank accuracy 99.4% on generated data vs 95.7% on the original data | Generated narrations share merchant vocabulary across banks → easier. We report the original-data number as the honest one |
| Oct 2026 | Literature: arXiv 2410.01609 is "SynJAC" | The paper's current title is **DAViD** (same arXiv id, renamed) | Cite it under its current title |
| Oct 2026 | Dev environment can download Qwen2-VL | Hugging Face is blocked from the build environment | All model code smoke-tested with a tiny random Qwen2-VL; real runs happen on the laptop GPU / Kaggle |
