# PocketAcc web app - CPU image (Tesseract reader + corrector + classifier + anomaly check).
#   docker build -t pocketacc .
#   docker run -p 8000:8000 pocketacc          ->  http://localhost:8000
# The fine-tuned vision model needs a GPU: run the app on the GPU machine instead
# (pip install -r requirements-gpu.txt, put the adapter in runs/lora_r8/best_adapter).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt

COPY src ./src
COPY scripts/test_app.py ./scripts/test_app.py
COPY data/original/ground_truth.json ./data/original/ground_truth.json
COPY data/original/pdf/03_PNB_Rakesh-Agarwal_Indore.pdf ./data/original/pdf/
COPY data/generated/ground_truth.json ./data/generated/ground_truth.json
COPY data/generated/pdf/G007_AXIS_Suresh-Ansari.pdf data/generated/pdf/G005_ICICI_Rahul-Bhatt.pdf ./data/generated/pdf/
# train the classifier and the anomaly model once, at build time (~20 s)
RUN python src/pipeline.py --train-models

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
