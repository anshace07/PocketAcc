"""
start_public_app.py - run the PocketAcc web app on this machine and give it a public https link.

Made for the live demo on Kaggle (GPU T4 x2, Internet On): the app runs on the second GPU, so uploads are
read by the FINE-TUNED vision model, and a free Cloudflare "quick tunnel" gives a link
(https://<random>.trycloudflare.com) that opens on any laptop or phone. No account is needed.

    import start_public_app; start_public_app.start()          # from the notebook (keeps running)
    python scripts/start_public_app.py                         # from a terminal (Ctrl+C to stop)

The link works only while this notebook session is running.
"""
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
PORT = 8000
PROCS = []


def n_gpus():
    out = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout
    return len([l for l in out.splitlines() if l.strip()])


def wait_for(url, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status == 200
        except Exception:
            time.sleep(2)
    return False


def start():
    (ROOT / "logs").mkdir(exist_ok=True)
    gpu = os.environ.get("APP_GPU") or ("1" if n_gpus() >= 2 else "0")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, POCKETACC_PRELOAD="1")
    print(f"starting the web app on GPU {gpu} ...", flush=True)
    PROCS.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0",
                                   "--port", str(PORT)], cwd=ROOT, env=env,
                                  stdout=open(ROOT / "logs/app.log", "w"), stderr=subprocess.STDOUT))
    if not wait_for(f"http://localhost:{PORT}/api/health", 240):
        print(open(ROOT / "logs/app.log").read()[-3000:])
        raise RuntimeError("the app did not start - see logs/app.log above")
    print("app is running locally", flush=True)

    cf = Path("/tmp/cloudflared")
    if not cf.exists():
        urllib.request.urlretrieve(CLOUDFLARED, cf)
        cf.chmod(0o755)
    log = ROOT / "logs/tunnel.log"
    PROCS.append(subprocess.Popen([str(cf), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
                                  stdout=open(log, "w"), stderr=subprocess.STDOUT))
    t0 = time.time()
    while time.time() - t0 < 90:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log.read_text(errors="ignore"))
        if m:
            print("\n" + "=" * 70 + f"\n  OPEN THIS LINK:  {m.group(0)}\n" + "=" * 70)
            print("The fine-tuned model is loading in the background (about 1-2 minutes after this).\n"
                  "Works while this notebook is running. Reading with the vision model: ~2-3 min per page.")
            return m.group(0)
        time.sleep(2)
    print(log.read_text(errors="ignore")[-2000:])
    raise RuntimeError("no public link from cloudflared - use Part B (live cell) or the results page instead")


def stop():
    for p in PROCS:
        p.terminate()
    PROCS.clear()


if __name__ == "__main__":
    start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        stop()
