"""
app.py - STEP 10: the PocketAcc web app (FastAPI).

Upload a bank statement (PDF or phone photo) -> table of verified, categorised transactions,
with corrected rows in green, rows needing review in red, unusual rows in amber,
a trust score, spending by category, and CSV / Excel / JSON export.

Run:
    uvicorn app:app --app-dir src --port 8000          then open http://localhost:8000
    docker build -t pocketacc . && docker run -p 8000:8000 pocketacc

API:
    GET  /api/health                     readers available
    GET  /api/samples                    demo statements
    POST /api/extract   (file, reader, bank)
    POST /api/samples/{name}             run a demo statement
    GET  /api/results/{id}.csv|.xlsx|.json
"""
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

import pipeline
from degrade import degrade
from preprocess import render_pages

ROOT = pipeline.ROOT
STATIC = Path(__file__).resolve().parent / "static"
ALLOWED = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MAX_MB = 25
KEEP = 50                                   # results kept in memory for export

app = FastAPI(title="PocketAcc", version="1.0",
              description="Bank statement photo/PDF -> verified, categorised transactions")
MODELS = pipeline.load_models()
RESULTS: "OrderedDict[str, dict]" = OrderedDict()
SAMPLE_DIR = Path(tempfile.mkdtemp(prefix="pocketacc_samples_"))
SAMPLES = {
    "axis_pdf": dict(label="Digital PDF", detail="Axis Bank · 2 pages · a bank the model never trained on",
                     path=ROOT / "data/generated/pdf/G007_AXIS_Suresh-Ansari.pdf"),
    "pnb_pdf": dict(label="Digital PDF", detail="Punjab National Bank · kirana shop · 9 pages (~40 s)",
                    path=ROOT / "data/original/pdf/03_PNB_Rakesh-Agarwal_Indore.pdf"),
    "icici_photo_light": dict(label="Phone photo, light damage", detail="ICICI Bank · 1 page",
                              page=(ROOT / "data/generated/pdf/G005_ICICI_Rahul-Bhatt.pdf", "light")),
    "icici_photo_medium": dict(label="Phone photo, medium damage", detail="ICICI Bank · 1 page · shows the low-trust warning",
                               page=(ROOT / "data/generated/pdf/G005_ICICI_Rahul-Bhatt.pdf", "medium")),
}


def sample_path(name):
    s = SAMPLES[name]
    if "path" in s:
        return s["path"]
    out = SAMPLE_DIR / f"{name}.jpg"
    if not out.exists():                    # photo samples are made on demand with our own damage generator
        pdf, level = s["page"]
        img, _ = degrade(render_pages(str(pdf), 150)[0], level, seed=3)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def keep(result):
    rid = uuid.uuid4().hex[:12]
    result["id"] = rid
    RESULTS[rid] = result
    while len(RESULTS) > KEEP:
        RESULTS.popitem(last=False)
    return result


def run(path, reader, bank):
    if reader not in ("auto", "tesseract", "vlm"):
        raise HTTPException(400, "reader must be auto, tesseract or vlm")
    if reader == "vlm" and not pipeline.vlm_available():
        raise HTTPException(400, "The vision model needs a GPU and a trained adapter (runs/lora_r8/best_adapter).")
    bank = None if bank in (None, "", "auto") else bank
    if bank is not None and bank not in pipeline.BANK_LABEL:
        raise HTTPException(400, f"unknown bank {bank}")
    try:
        return keep(pipeline.analyse(path, reader, bank, MODELS))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "readers": {"tesseract": True, "vlm": pipeline.vlm_available()},
            "banks": pipeline.BANK_LABEL}


@app.get("/api/samples")
def samples():
    return [{"name": k, "label": v["label"], "detail": v["detail"]} for k, v in SAMPLES.items()]


@app.post("/api/samples/{name}")
def run_sample(name: str, reader: str = Form("auto")):
    if name not in SAMPLES:
        raise HTTPException(404, "no such sample")
    return run(sample_path(name), reader, None)


@app.post("/api/extract")
def extract(file: UploadFile = File(...), reader: str = Form("auto"), bank: str = Form("auto")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, "Upload a PDF or an image (JPG, PNG, WEBP, TIFF).")
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"upload{suffix}"
        path.write_bytes(data)
        result = run(path, reader, bank)
    result["file"] = Path(file.filename).name
    return result


@app.get("/api/results/{rid}.{fmt}")
def export(rid: str, fmt: str):
    if rid not in RESULTS:
        raise HTTPException(404, "result expired - run the statement again")
    r = RESULTS[rid]
    stem = Path(r["file"]).stem + "_pocketacc"
    if fmt == "csv":
        return Response(pipeline.to_csv(r), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.csv"'})
    if fmt == "xlsx":
        return Response(pipeline.to_xlsx(r),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.xlsx"'})
    if fmt == "json":
        return JSONResponse(r, headers={"Content-Disposition": f'attachment; filename="{stem}.json"'})
    raise HTTPException(400, "format must be csv, xlsx or json")
