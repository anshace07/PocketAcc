"""
test_app.py - end-to-end test of the web app (step 10) through its HTTP API, on CPU.

    python scripts/test_app.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

from app import app  # noqa: E402

c = TestClient(app)

h = c.get("/api/health").json()
assert h["ok"] and h["readers"]["tesseract"], h
print("[1] OK health:", h["readers"])

assert c.get("/").status_code == 200 and "PocketAcc" in c.get("/").text
names = [s["name"] for s in c.get("/api/samples").json()]
print("[2] OK dashboard page served; samples:", names)

pdf = ROOT / "data/generated/pdf/G007_AXIS_Suresh-Ansari.pdf"
r = c.post("/api/extract", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
           data={"reader": "auto", "bank": "auto"})
assert r.status_code == 200, r.text
res = r.json()
s = res["summary"]
assert res["bank"] == "AXIS" and s["transactions"] > 0 and s["trust_score"] is not None, s
print(f"[3] OK PDF upload: {res['bank_label']}, {s['transactions']} rows, trust {s['trust_score']}%, "
      f"corrected {s['auto_corrected']}, review {s['needs_review']}, unusual {s['anomalies']}, closing matches {s['closing_matches']}")

csv = c.get(f"/api/results/{res['id']}.csv")
assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("date,narration,debit")
assert len(csv.text.strip().splitlines()) == s["transactions"] + 1
wb = load_workbook(io.BytesIO(c.get(f"/api/results/{res['id']}.xlsx").content))
assert wb.sheetnames == ["Transactions", "Summary"] and wb["Transactions"].max_row == s["transactions"] + 1
js = c.get(f"/api/results/{res['id']}.json").json()
assert js["id"] == res["id"] and "attachment" in c.get(f"/api/results/{res['id']}.csv").headers["content-disposition"]
print("[4] OK exports: CSV, Excel (2 sheets), JSON")

bad = c.post("/api/extract", files={"file": ("notes.txt", b"hello", "text/plain")})
assert bad.status_code == 415, bad.status_code
broken = c.post("/api/extract", files={"file": ("photo.jpg", b"not an image", "image/jpeg")})
assert broken.status_code == 422, broken.status_code
assert c.get("/api/results/nope.csv").status_code == 404
vlm = c.post("/api/extract", files={"file": (pdf.name, pdf.read_bytes())}, data={"reader": "vlm"})
assert vlm.status_code == (200 if h["readers"]["vlm"] else 400)
print("[5] OK errors: wrong file type 415, unreadable image 422, expired result 404, VLM without GPU 400")

p = c.post("/api/samples/icici_photo_light", data={"reader": "tesseract"}).json()
print(f"[6] OK phone-photo sample (light damage): {p['bank_label']}, {p['summary']['transactions']} rows, "
      f"trust {p['summary']['trust_score']}%, notes: {len(p['notes'])}")
print("\nALL APP TESTS PASSED")
