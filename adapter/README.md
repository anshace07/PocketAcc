# Trained LoRA adapter

`lora_r8/` is the QLoRA adapter for **Qwen/Qwen2-VL-2B-Instruct** trained on Kaggle (Tesla T4, 18 Nov 2026)
with `notebooks/pocketacc_kaggle.ipynb`: rank 8, 1 epoch over 300 training pages (37 optimizer steps,
gradient accumulation 8), images limited to 900,000 pixels, best validation loss 0.060.
Training log: `results/training_log.csv`, settings: `results/run_config.json`.

The app (`src/pipeline.py`) uses it automatically on a machine with a CUDA GPU; on CPU it reads with Tesseract.
