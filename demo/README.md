# Demo samples

`bob_page1_{clean,medium,heavy}.jpg` - page 1 of a Bank of Baroda statement (a bank never used in training) as a clean
scan and as phone photos with medium and heavy damage (test split `test_unseen`, after pre-processing).

`gpu_outputs.json` - what the fine-tuned model read on each of these pages, recorded on a Kaggle Tesla T4 during the
step-9 evaluation (`results/ft_unseen.jsonl`). The app uses these recordings so the model's real output can be shown
on a laptop without a GPU; Tesseract, the corrector, the classifier and the anomaly check run live.
