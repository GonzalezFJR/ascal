"""Web app: upload an all-sky frame, give the site and the UTC time, get the calibration with an interactive viewer.

Run with ``ascal web`` or ``uvicorn ascal.web.app:app``. It can be served under a path prefix by a reverse proxy
that strips the prefix (all URLs of the page are relative).

Jobs run one at a time, each in its own process (``ascal.web.worker``), from a queue. Uploads and results live in
``ASCAL_WEB_DATA`` and are deleted after ``ASCAL_WEB_TTL_H`` hours. Environment variables:

ASCAL_WEB_DATA          job directory (default: <tmp>/ascal_web)
ASCAL_WEB_MAX_MB        largest upload in MB (150)
ASCAL_WEB_MAX_QUEUE     jobs waiting at most (8)
ASCAL_WEB_RATE          jobs per client IP and hour (20)
ASCAL_WEB_TTL_H         hours a job is kept (6)
ASCAL_WEB_MAX_TIME      largest search time a user may ask for, s (180)
ASCAL_WEB_TRUST_PROXY   "1": take the client IP from X-Real-IP / X-Forwarded-For (behind a proxy)
ASCAL_WEB_EXAMPLES      directory with examples.json and precomputed/<id>/ (deploy/precompute_examples.py)
"""
from __future__ import annotations

import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__

STATIC = Path(__file__).resolve().parent / "static"
DATA = Path(os.environ.get("ASCAL_WEB_DATA", Path(tempfile.gettempdir()) / "ascal_web"))
MAX_MB = float(os.environ.get("ASCAL_WEB_MAX_MB", "150"))
MAX_QUEUE = int(os.environ.get("ASCAL_WEB_MAX_QUEUE", "8"))
RATE = int(os.environ.get("ASCAL_WEB_RATE", "20"))
TTL = float(os.environ.get("ASCAL_WEB_TTL_H", "6")) * 3600
MAX_TIME = float(os.environ.get("ASCAL_WEB_MAX_TIME", "180"))
TRUST_PROXY = os.environ.get("ASCAL_WEB_TRUST_PROXY", "0") == "1"
_repo_examples = Path(__file__).resolve().parents[2] / "examples" / "web"
EXAMPLES = Path(os.environ.get("ASCAL_WEB_EXAMPLES", _repo_examples))
FORMATS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz",
           ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw")
EXAMPLE_FILES = {"display.jpg": "image/jpeg", "thumb.jpg": "image/jpeg", "calibration.json": "application/json",
                 "pairs.csv": "text/csv", "panel.png": "image/png", "result.json": "application/json", "log.txt": "text/plain"}
ELEV = (-450.0, 6000.0)
FWHM = (1.5, 15.0)
FILES = {"display.jpg": "image/jpeg", "calibration.json": "application/json", "pairs.csv": "text/csv",
         "panel.png": "image/png", "result.json": "application/json"}
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{16,40}$")

app = FastAPI(title="ascal", version=__version__, docs_url=None, redoc_url=None)
DATA.mkdir(parents=True, exist_ok=True)

_jobs: Dict[str, Dict[str, Any]] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_order: deque = deque()                  # ids waiting, for the queue position
_rate: Dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


# ---------------------------------------------------------------------- helpers
def _suffix(name: str) -> str:
    n = name.lower()
    for s in sorted(FORMATS, key=len, reverse=True):
        if n.endswith(s):
            return s
    return ""


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        for h in ("x-real-ip", "x-forwarded-for"):
            v = request.headers.get(h)
            if v:
                return v.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _examples() -> list:
    """Examples with precomputed results (precomputed/<id>/result.json)."""
    try:
        items = json.loads((EXAMPLES / "examples.json").read_text())
    except (OSError, ValueError):
        return []
    return [e for e in items if (EXAMPLES / "precomputed" / e["id"] / "result.json").exists()]


def _job(job_id: str) -> Dict[str, Any]:
    if not JOB_ID.match(job_id or ""):
        raise HTTPException(404, "unknown job")
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This result has expired or does not exist.")
    return job


def _purge() -> None:
    now = time.time()
    with _lock:
        old = [k for k, j in _jobs.items() if now - j["created"] > TTL and j["state"] in ("done", "error")]
        for k in old:
            _jobs.pop(k, None)
    for k in old:
        shutil.rmtree(DATA / k, ignore_errors=True)
    for d in DATA.glob("*"):                        # leftovers of a previous run
        if d.is_dir() and d.name not in _jobs and now - d.stat().st_mtime > TTL:
            shutil.rmtree(d, ignore_errors=True)


def _runner() -> None:
    while True:
        job_id = _queue.get()
        with _lock:
            job = _jobs.get(job_id)
            if job_id in _order:
                _order.remove(job_id)
        if job is None:
            continue
        job.update(state="running", started=time.time())
        d = DATA / job_id
        timeout = job["max_time"] + 300
        try:
            proc = subprocess.run([sys.executable, "-m", "ascal.web.worker", str(d)], timeout=timeout,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = proc.returncode == 0 and (d / "result.json").exists()
        except subprocess.TimeoutExpired:
            ok = False
            (d / "error.json").write_text(json.dumps({"kind": "internal", "message": "The job took too long and was stopped."}))
        if not ok and not (d / "error.json").exists():
            (d / "error.json").write_text(json.dumps({"kind": "internal", "message": "The calibration process ended unexpectedly."}))
        job.update(state="done" if ok else "error", finished=time.time())
        src = d / job["file"]
        if src.exists():
            src.unlink()                           # keep no uploaded image once the job is over
        _purge()


def _recover() -> None:
    """Jobs left on disk by a previous run of the server: finished ones are served again, unfinished ones failed."""
    for d in DATA.glob("*"):
        if not (d.is_dir() and JOB_ID.match(d.name) and (d / "params.json").exists()):
            continue
        try:
            params = json.loads((d / "params.json").read_text())
        except ValueError:
            continue
        if not (d / "result.json").exists() and not (d / "error.json").exists():
            (d / "error.json").write_text(json.dumps({"kind": "internal", "message": "The server restarted during the job; please submit it again."}))
        _jobs[d.name] = {"state": "done" if (d / "result.json").exists() else "error", "created": (d / "params.json").stat().st_mtime,
                         "file": params["file"], "max_time": params.get("max_time", 40), "name": params["original_name"]}


_recover()
threading.Thread(target=_runner, daemon=True, name="ascal-web-runner").start()


# ---------------------------------------------------------------------- pages and API
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def api_config() -> dict:
    return {"version": __version__, "max_mb": MAX_MB, "max_time": MAX_TIME, "default_time": 40, "ttl_h": TTL / 3600,
            "formats": [s for s in FORMATS if not s.endswith(".gz")],
            "elev_range": ELEV, "fwhm_range": FWHM,
            "examples": [{k: v for k, v in e.items() if k != "file"} for e in _examples()]}


@app.post("/api/exif")
async def api_exif(image: UploadFile = File(...)) -> dict:
    """Time stamp and exposure found in the file (EXIF or FITS header), so the form can be pre-filled."""
    from ..detect import FILENAME_TIME, _kind, fits_time, read_exif
    suffix = _suffix(image.filename or "") or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(await image.read())
        path = Path(fh.name)
    try:
        out: Dict[str, Any] = {"source": None}
        if _kind(path) == "fits":
            t, e = fits_time(path)
            out.update(datetime=t.isoformat() if t else None, exposure_s=e, source="FITS header (UTC)" if t else None)
        else:
            info = read_exif(path)
            if "datetime" in info:
                out.update(datetime=info["datetime"].isoformat(), source="EXIF (camera clock, often local time)")
            out["exposure_s"] = info.get("exposure_s")
        m = FILENAME_TIME.search(image.filename or "")
        if m and not out.get("datetime"):
            out.update(datetime=datetime(*(int(v) for v in m.groups())).isoformat(), source="file name (time zone unknown)")
        return out
    except Exception:
        return {"source": None}
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/jobs")
async def api_submit(request: Request, image: Optional[UploadFile] = File(None),
                     lat: float = Form(...), lon: float = Form(...), elev: float = Form(0.0), utc: str = Form(...),
                     exposure: Optional[float] = Form(None), max_time: float = Form(40.0), parity: str = Form("auto"),
                     fwhm: str = Form("auto")) -> dict:
    _purge()
    if not (-90 <= lat <= 90 and -180 <= lon <= 360 and ELEV[0] <= elev <= ELEV[1]):
        raise HTTPException(422, "Latitude, longitude or elevation out of range.")
    try:
        t = datetime.fromisoformat(utc.strip().replace("Z", "").replace(" ", "T"))
    except ValueError:
        raise HTTPException(422, "The time must be ISO 8601 UTC, e.g. 2026-08-09T01:00:46.")
    if exposure is not None and not (0 <= exposure <= 3600):
        raise HTTPException(422, "Exposure out of range (0-3600 s).")
    if parity not in ("auto", "direct", "mirror"):
        raise HTTPException(422, "parity must be auto, direct or mirror")
    if fwhm != "auto":
        try:
            if not FWHM[0] <= float(fwhm) <= FWHM[1]:
                raise ValueError
        except ValueError:
            raise HTTPException(422, f"The kernel FWHM must be 'auto' or between {FWHM[0]:g} and {FWHM[1]:g} px.")
    max_time = min(max(float(max_time), 10.0), MAX_TIME)

    ip = _client_ip(request)
    now = time.time()
    with _lock:
        hits = _rate[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if len(hits) >= RATE:
            raise HTTPException(429, f"Limit of {RATE} calibrations per hour reached; please try again later.")
        if len(_order) >= MAX_QUEUE:
            raise HTTPException(503, "The server is busy (queue full); please try again in a few minutes.")

    job_id = secrets.token_urlsafe(16)
    d = DATA / job_id
    d.mkdir(parents=True)
    if image is None:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, "No image.")
    name = Path(image.filename or "frame.jpg").name
    suffix = _suffix(name)
    if not suffix:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(415, "Unsupported format. Use JPEG, PNG, TIFF, FITS or a camera raw file.")
    size = 0
    with open(d / ("frame" + suffix), "wb") as fh:
        while chunk := await image.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB * 1e6:
                fh.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"The file is larger than {MAX_MB:.0f} MB.")
            fh.write(chunk)
    params = {"file": "frame" + _suffix(name), "original_name": name, "lat": lat, "lon": lon, "elev": elev,
              "utc": t.isoformat(), "exposure": exposure, "max_time": max_time, "parity": parity, "fwhm": fwhm}
    (d / "params.json").write_text(json.dumps(params))
    with _lock:
        _rate[ip].append(now)
        _jobs[job_id] = {"state": "queued", "created": now, "file": params["file"], "max_time": max_time,
                         "name": name}
        _order.append(job_id)
    _queue.put(job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def api_status(job_id: str) -> dict:
    job = _job(job_id)
    d = DATA / job_id
    with _lock:
        pos = list(_order).index(job_id) + 1 if job_id in _order else 0
    try:
        log = (d / "log.txt").read_text().splitlines()[-80:]
    except OSError:
        log = []
    out = {"state": job["state"], "position": pos, "name": job["name"], "log": log,
           "elapsed_s": round(time.time() - job["started"], 1) if job.get("started") else None,
           "expires_in_h": round(max(0.0, TTL - (time.time() - job["created"])) / 3600, 2)}
    if job["state"] == "error":
        try:
            out["error"] = json.loads((d / "error.json").read_text())
        except (OSError, ValueError):
            out["error"] = {"kind": "internal", "message": "unknown error"}
    return out


@app.get("/api/jobs/{job_id}/{name}")
def api_file(job_id: str, name: str) -> Response:
    job = _job(job_id)
    if name not in FILES or job["state"] != "done":
        raise HTTPException(404, "not available")
    path = DATA / job_id / name
    if not path.exists():
        raise HTTPException(404, "not available")
    stem = Path(job["name"]).stem.replace('"', "")
    headers = {"Cache-Control": "private, max-age=3600"}
    if name != "display.jpg" and name != "result.json":
        headers["Content-Disposition"] = f'attachment; filename="{stem}_{name}"'
    return FileResponse(path, media_type=FILES[name], headers=headers)


@app.get("/api/examples/{example_id}/{name}")
def api_example_file(example_id: str, name: str) -> Response:
    ex = next((e for e in _examples() if e["id"] == example_id), None)
    if ex is None or name not in EXAMPLE_FILES:
        raise HTTPException(404, "not available")
    headers = {"Cache-Control": "public, max-age=86400"}
    if name not in ("display.jpg", "thumb.jpg", "result.json", "log.txt"):
        headers["Content-Disposition"] = f'attachment; filename="{example_id}_{name}"'
    return FileResponse(EXAMPLES / "precomputed" / example_id / name, media_type=EXAMPLE_FILES[name], headers=headers)


@app.get("/api/version")
def api_version() -> dict:
    return {"version": __version__}


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
