"""Drag-and-drop web demo: upload an all-sky frame, get a calibration and diagnostic figures.

Run with ``allskycal web`` or ``uvicorn allskycal.web.app:app``.  Everything runs in the server
process (detection + fit take 10-60 s depending on the machine); one job at a time.
"""
from __future__ import annotations

import base64
import json
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..bootstrap import CalibrationError, Site, calibrate
from ..detect import load_frame, read_exif
from .. import plots

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="allskycal", version=__version__)
_lock = threading.Lock()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/version")
def version() -> dict:
    return {"version": __version__}


@app.post("/api/exif")
async def exif(image: UploadFile = File(...)) -> dict:
    """Exposure and timestamp found in the file, so the form can be pre-filled."""
    with tempfile.NamedTemporaryFile(suffix=Path(image.filename or "img.jpg").suffix, delete=False) as fh:
        fh.write(await image.read())
        path = Path(fh.name)
    try:
        info = read_exif(path)
        from ..detect import FILENAME_TIME
        m = FILENAME_TIME.search(image.filename or "")
        return {"exposure_s": info.get("exposure_s"), "datetime": info["datetime"].isoformat() if "datetime" in info else None,
                "filename_time": datetime(*(int(v) for v in m.groups())).isoformat() if m else None}
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/calibrate")
async def api_calibrate(image: UploadFile = File(...), lat: float = Form(...), lon: float = Form(...), elev: float = Form(0.0),
                        time: Optional[str] = Form(None), tz: Optional[str] = Form(None), exposure: Optional[float] = Form(None),
                        decentering: bool = Form(False), tiles: int = Form(1)) -> JSONResponse:
    if not _lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="A calibration is already running; try again in a minute.")
    suffix = Path(image.filename or "img.jpg").suffix or ".jpg"
    tmpdir = Path(tempfile.mkdtemp(prefix="allskycal_"))
    path = tmpdir / (Path(image.filename or "frame").stem + suffix)
    try:
        path.write_bytes(await image.read())
        t = datetime.fromisoformat(time) if time else None
        frame = load_frame(path, time=t, tz=tz or None, exposure_s=exposure, tiles=max(1, tiles), keep_image=True)
        site = Site(lat, lon, elev)
        log: list = []
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = calibrate([frame], site, decentering=decentering, verbose=True)
        log = buf.getvalue().splitlines()
        summary = result.summary()
        payload = {
            "ok": True,
            "frame": {"name": path.name, "utc_mid": frame.utc.isoformat(), "exposure_s": frame.exposure_s, "n_detections": len(frame.detections),
                      "width": frame.width, "height": frame.height},
            "calibration": result.model.to_dict(),
            "summary": {k: v for k, v in summary.items() if k not in ("pose_candidates",)},
            "log": log,
            "figures": {
                "overlay": base64.b64encode(plots.overlay(frame, result.model, site)).decode(),
                "cutouts": base64.b64encode(plots.cutouts(frame, result.model, site)).decode(),
                "residuals": base64.b64encode(plots.residual_plots(result)).decode(),
                "radial": base64.b64encode(plots.radial_plot(result.model)).decode(),
            },
        }
        return JSONResponse(json.loads(json.dumps(payload, default=float)))
    except CalibrationError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)
    except Exception as exc:  # pragma: no cover
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}, status_code=500)
    finally:
        _lock.release()
        for p in tmpdir.glob("*"):
            p.unlink(missing_ok=True)
        tmpdir.rmdir()


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
