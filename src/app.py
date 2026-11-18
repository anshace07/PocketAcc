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
    POST /api/extract   (file, reader, bank, wait)
    POST /api/samples/{name}  (reader, wait)
         wait=1 (default): the answer is the result.  wait=0: the answer is {"job": id} at once and
         GET /api/jobs/{id} reports progress - used by the dashboard, because the vision model needs
         2-3 minutes per page and a long silent request can time out behind a proxy.
    GET  /api/jobs/{id}                  progress / result of a reading started with wait=0
    GET  /api/saved                      results saved earlier with scripts/run_demo.py (e.g. on a Kaggle GPU)
    GET  /api/saved/{name}               open one saved result
    GET  /api/evaluation                 test-set scores of every reader (results/score24_*.json)
    GET  /api/results/{id}.csv|.xlsx|.json
    GET  /api/results/{id}/page.jpg      first page of the file that was read
"""
import json
import os
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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
SAVED_DIR = Path(os.environ.get("POCKETACC_SAVED", ROOT / "demo/results"))
DEMO = ROOT / "demo/samples"
DEMO_ORDER = ["A_salaried_statement.pdf", "A_page1_photo_light.jpg", "A_page1_photo_medium.jpg",
              "A_page1_photo_heavy.jpg", "B_shop_statement.pdf"]

app = FastAPI(title="PocketAcc", version="1.1",
              description="Bank statement photo/PDF -> verified, categorised transactions")
MODELS = pipeline.load_models()
RESULTS: "OrderedDict[str, dict]" = OrderedDict()
PREVIEWS: dict = {}
JOBS: "OrderedDict[str, dict]" = OrderedDict()
WORKER = ThreadPoolExecutor(max_workers=1)  # one statement at a time: the GPU holds one model
UPLOADS = Path(tempfile.mkdtemp(prefix="pocketacc_uploads_"))
SAMPLE_DIR = Path(tempfile.mkdtemp(prefix="pocketacc_samples_"))
SAMPLES = {
    "demo_a_pdf": dict(label="Demo statement · digital PDF",
                       detail="PocketAcc Demo Bank – a made-up layout the model never saw · 2 pages",
                       path=DEMO / "A_salaried_statement.pdf"),
    "demo_a_heavy": dict(label="Demo statement · phone photo, heavy damage",
                         detail="page 1 · tilted, blurred, stamped – Tesseract reads nothing here",
                         path=DEMO / "A_page1_photo_heavy.jpg"),
    "demo_b_pdf": dict(label="Demo statement · kirana shop", detail="PocketAcc Demo Bank · 3 pages · 102 transactions",
                       path=DEMO / "B_shop_statement.pdf"),
    "axis_pdf": dict(label="Digital PDF", detail="Axis Bank · 2 pages · a bank the model never trained on",
                     path=ROOT / "data/generated/pdf/G007_AXIS_Suresh-Ansari.pdf"),
    "pnb_pdf": dict(label="Digital PDF", detail="Punjab National Bank · kirana shop · 9 pages (~40 s)",
                    path=ROOT / "data/original/pdf/03_PNB_Rakesh-Agarwal_Indore.pdf"),
    "icici_photo_light": dict(label="Phone photo, light damage", detail="ICICI Bank · 1 page",
                              page=(ROOT / "data/generated/pdf/G005_ICICI_Rahul-Bhatt.pdf", "light")),
    "icici_photo_medium": dict(label="Phone photo, medium damage", detail="ICICI Bank · 1 page · shows the low-trust warning",
                               page=(ROOT / "data/generated/pdf/G005_ICICI_Rahul-Bhatt.pdf", "medium")),
}
SAMPLES = {k: v for k, v in SAMPLES.items() if Path(v.get("path") or v["page"][0]).exists()}


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


def keep(result, source=None):
    rid = uuid.uuid4().hex[:12]
    result["id"] = rid
    RESULTS[rid] = result
    if source is not None:
        try:
            preview = pipeline.page_preview(source)
            if preview:
                PREVIEWS[rid] = preview
        except Exception:                   # a preview is a nice-to-have, never a reason to fail
            pass
    result["has_page"] = rid in PREVIEWS
    while len(RESULTS) > KEEP:
        old, _ = RESULTS.popitem(last=False)
        PREVIEWS.pop(old, None)
    return result


def check(reader, bank):
    if reader not in ("auto", "tesseract", "vlm"):
        raise HTTPException(400, "reader must be auto, tesseract or vlm")
    if reader == "vlm" and not pipeline.vlm_available():
        raise HTTPException(400, "The vision model needs a GPU and a trained adapter (adapters/lora_r8).")
    bank = None if bank in (None, "", "auto") else bank
    if bank is not None and bank not in pipeline.BANK_LABEL:
        raise HTTPException(400, f"unknown bank {bank}")
    return bank


def analyse_file(path, reader, bank, name, progress=None):
    result = pipeline.analyse(path, reader, bank, MODELS, progress)
    result["file"] = name
    pipeline.attach_answer_key(result, name)
    return keep(result, path)


def respond(work, wait, cleanup=None):
    """work(progress) -> result. wait: answer with the result; else start a job and answer with its id."""
    if wait:
        try:
            return WORKER.submit(work, None).result()
        except ValueError as e:
            raise HTTPException(422, str(e))
        finally:
            if cleanup:
                cleanup()
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "status": "running", "message": "Waiting for the reader", "started": time.time()}
    JOBS[jid] = job
    while len(JOBS) > KEEP:
        JOBS.popitem(last=False)

    def progress(msg):
        job["message"] = msg

    def run():
        try:
            job["result"] = work(progress)
            job["status"] = "done"
        except ValueError as e:
            job["status"], job["error"] = "error", str(e)
        except Exception as e:  # noqa: BLE001 - reported to the user instead of a silent failure
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        finally:
            job["seconds"] = round(time.time() - job["started"], 1)
            if cleanup:
                cleanup()

    WORKER.submit(run)
    return JSONResponse({"job": jid, "status": "running"}, status_code=202)


if os.environ.get("POCKETACC_PRELOAD") == "1" and pipeline.vlm_available():
    WORKER.submit(pipeline.read_vlm, [])     # load the vision model now, not on the first upload


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "readers": {"tesseract": True, "vlm": pipeline.vlm_available()},
            "vlm_loaded": pipeline._EXTRACTOR is not None, "banks": pipeline.BANK_LABEL,
            "saved": len(list(SAVED_DIR.glob("*.json"))) if SAVED_DIR.exists() else 0}


@app.get("/api/samples")
def samples():
    return [{"name": k, "label": v["label"], "detail": v["detail"]} for k, v in SAMPLES.items()]


@app.post("/api/samples/{name}")
def run_sample(name: str, reader: str = Form("auto"), wait: str = Form("1")):
    if name not in SAMPLES:
        raise HTTPException(404, "no such sample")
    bank = check(reader, None)
    path = sample_path(name)
    return respond(lambda progress: analyse_file(path, reader, bank, Path(path).name, progress), wait != "0")


@app.post("/api/extract")
def extract(file: UploadFile = File(...), reader: str = Form("auto"), bank: str = Form("auto"), wait: str = Form("1")):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, "Upload a PDF or an image (JPG, PNG, WEBP, TIFF).")
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    bank = check(reader, bank)
    d = Path(tempfile.mkdtemp(dir=UPLOADS))
    path = d / f"upload{suffix}"
    path.write_bytes(data)
    name = Path(file.filename).name
    return respond(lambda progress: analyse_file(path, reader, bank, name, progress), wait != "0",
                   cleanup=lambda: shutil.rmtree(d, ignore_errors=True))


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "no such job")
    out = {"id": jid, "status": job["status"], "message": job["message"],
           "elapsed": round(time.time() - job["started"], 1)}
    if job["status"] == "done":
        out["result"] = job["result"]
    elif job["status"] == "error":
        out["error"] = job["error"]
    return out


def _saved_files():
    if not SAVED_DIR.exists():
        return []
    rank = {Path(f).stem: i for i, f in enumerate(DEMO_ORDER)}
    files = []
    for f in SAVED_DIR.glob("*.json"):
        stem, _, backend = f.stem.rpartition(".")
        files.append((rank.get(stem, 99), stem, 0 if backend == "vlm" else 1, f))
    out = []
    for f in sorted(files, key=lambda x: (x[0], x[1], x[2])):
        try:
            out.append((f[3], json.load(open(f[3]))))
        except (OSError, json.JSONDecodeError):
            continue
    return out


@app.get("/api/saved")
def saved():
    items = []
    for f, r in _saved_files():
        s = r.get("saved") or {}
        items.append({"name": f.stem, "file": r.get("file"), "label": s.get("label", r.get("file")),
                      "detail": s.get("detail", ""), "reader": s.get("reader", r.get("backend")),
                      "backend": r.get("backend"), "hardware": s.get("hardware"), "created": s.get("created"),
                      "rows": r["summary"]["transactions"], "trust": r["summary"]["trust_score"],
                      "answer_key": r.get("answer_key")})
    return items


@app.get("/api/saved/{name}")
def open_saved(name: str):
    f = SAVED_DIR / f"{name}.json"
    if not f.exists() or f.parent != SAVED_DIR:
        raise HTTPException(404, "no such saved result")
    r = json.load(open(f))
    src = ROOT / (r.get("saved") or {}).get("source", "")
    return keep(r, src if src.is_file() else None)


@app.get("/api/evaluation")
def evaluation():
    return pipeline.evaluation_summary()


@app.get("/api/evaluation/loss_curve.png", include_in_schema=False)
def loss_curve():
    f = ROOT / "results/finetune/loss_curve.png"
    if not f.exists():
        raise HTTPException(404, "no loss curve")
    return FileResponse(f, media_type="image/png")


@app.get("/api/results/{rid}/page.jpg")
def page(rid: str):
    if rid not in PREVIEWS:
        raise HTTPException(404, "no page image for this result")
    return Response(PREVIEWS[rid], media_type="image/jpeg")


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
