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
    POST /api/extract   (file, reader = auto | both | tesseract | vlm, bank)
    POST /api/samples/{name}             run a demo statement
    GET  /api/results/{id}.csv|.xlsx|.json
"""
import json
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

import pipeline

ROOT = pipeline.ROOT
STATIC = Path(__file__).resolve().parent / "static"
ALLOWED = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
MAX_MB = 25
KEEP = 50                                   # results kept in memory for export

app = FastAPI(title="PocketAcc", version="1.0",
              description="Bank statement photo/PDF -> verified, categorised transactions")
MODELS = pipeline.load_models()
RESULTS: "OrderedDict[str, dict]" = OrderedDict()
DEMO = ROOT / "demo"
GPU_OUTPUTS = json.loads((DEMO / "gpu_outputs.json").read_text()) if (DEMO / "gpu_outputs.json").exists() else {}
SAMPLES = {
    "bob_page1_clean": dict(label="Clean scan", detail="Bank of Baroda · page 1 · never used in training · both readers"),
    "bob_page1_medium": dict(label="Phone photo, medium damage", detail="Same page · tilt, blur, stamp, shadow · both readers"),
    "bob_page1_heavy": dict(label="Phone photo, heavy damage", detail="Same page · strong tilt, blur, noise · both readers"),
    "axis_pdf": dict(label="Digital PDF", detail="Axis Bank · 2 pages · Tesseract only (no GPU here)",
                     path=ROOT / "data/generated/pdf/G007_AXIS_Suresh-Ansari.pdf"),
    "pnb_pdf": dict(label="Digital PDF, 9 pages", detail="Punjab National Bank · kirana shop · 592 rows (~40 s)",
                    path=ROOT / "data/original/pdf/03_PNB_Rakesh-Agarwal_Indore.pdf"),
}
for name in SAMPLES:
    if "path" not in SAMPLES[name]:
        SAMPLES[name]["path"] = DEMO / f"{name}.jpg"
SAMPLES = {k: v for k, v in SAMPLES.items() if Path(v["path"]).exists()}


def sample_path(name):
    return SAMPLES[name]["path"]


def keep(result):
    rid = uuid.uuid4().hex[:12]
    result["id"] = rid
    RESULTS[rid] = result
    while len(RESULTS) > KEEP:
        RESULTS.popitem(last=False)
    return result


def run(path, reader, bank, cache=None):
    if reader not in ("auto", "both", "tesseract", "vlm"):
        raise HTTPException(400, "reader must be auto, both, tesseract or vlm")
    if reader in ("vlm", "both") and cache is None and not pipeline.vlm_available():
        raise HTTPException(400, "The vision model needs a GPU and the trained adapter (adapter/lora_r8). "
                                 "Try a Bank of Baroda sample: their model readings were recorded on a GPU.")
    bank = None if bank in (None, "", "auto") else bank
    if bank is not None and bank not in pipeline.BANK_LABEL:
        raise HTTPException(400, f"unknown bank {bank}")
    try:
        return keep(pipeline.analyse(path, reader, bank, MODELS, cache))
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
    return [{"name": k, "label": v["label"], "detail": v["detail"], "recorded_model": k in GPU_OUTPUTS}
            for k, v in SAMPLES.items()]


@app.post("/api/samples/{name}")
def run_sample(name: str, reader: str = Form("auto")):
    if name not in SAMPLES:
        raise HTTPException(404, "no such sample")
    return run(sample_path(name), reader, None, GPU_OUTPUTS.get(name))


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
