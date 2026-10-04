"""One calibration job of the web app, run in its own process: ``python -m ascal.web.worker JOB_DIR``.

Reads ``JOB_DIR/params.json`` and the uploaded image, writes the progress to ``log.txt`` and, on success,
``result.json`` (model, statistics, cascade, image layers and chart data, all in pixels of the original image),
``display.jpg`` (the frame for the browser, at most ``ASCAL_WEB_DISPLAY_PX`` on its long side), ``calibration.json``,
``pairs.csv`` and ``panel.png``. On failure it writes ``error.json``.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DISPLAY_PX = int(os.environ.get("ASCAL_WEB_DISPLAY_PX", "4096"))


def _r(v, d=2):
    """Round for JSON; NaN/inf -> None."""
    if v is None:
        return None
    v = float(v)
    return round(v, d) if math.isfinite(v) else None


def _line(model, alt, az, W, H, d=1):
    """Polyline in pixels as [[x, y] | None, ...]; None breaks the line (below the horizon or off the sensor)."""
    x, y = model.project(np.asarray(alt, float), np.asarray(az, float))
    out = []
    for xi, yi, ai in zip(x, y, np.broadcast_to(alt, x.shape)):
        ok = np.isfinite(xi) and np.isfinite(yi) and -0.02 * W <= xi <= 1.02 * W and -0.02 * H <= yi <= 1.02 * H and ai >= -0.01
        out.append([round(float(xi), d), round(float(yi), d)] if ok else None)
    while out and out[0] is None:
        out.pop(0)
    while out and out[-1] is None:
        out.pop()
    return out if sum(p is not None for p in out) >= 2 else None


def _display_image(path: Path, gray: np.ndarray, out: Path) -> float:
    """Write the frame for the browser; colour JPEG/PNG kept in colour, everything else stretched. Returns the scale."""
    import cv2
    img = None
    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img is not None and img.dtype != np.uint8:
            f = img.astype(np.float32)
            lo, hi = np.percentile(f[::4, ::4], [1.0, 99.8])
            img = (np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.7 * 255).astype(np.uint8)
    if img is not None and img.dtype == np.uint8 and img.shape[:2] == gray.shape:
        # gentle common stretch of dark 8-bit frames (most all-sky JPEGs are dark); colours are kept
        lum = img if img.ndim == 2 else img.mean(axis=2)
        lo, hi = np.percentile(lum[::4, ::4], [0.5, 99.8])
        if hi < 235 and hi - lo > 5:
            f = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) ** 0.8
            img = (f * 255).astype(np.uint8)
    if img is None or img.shape[:2] != gray.shape:
        f = gray.astype(np.float32)
        lo, hi = np.percentile(f[::4, ::4], [1.0, 99.7])
        img = (np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.6 * 255).astype(np.uint8)
    h, w = img.shape[:2]
    k = min(1.0, DISPLAY_PX / max(h, w))
    if k < 1.0:
        img = cv2.resize(img, (int(round(w * k)), int(round(h * k))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return k


def build_result(frame, result, site, *, timings: dict) -> dict:
    from ..catalog import sky_stars
    from ..plots import _radec_polyline_altaz, constellation_lines

    m = result.model
    W, H = m.width, m.height
    p, res, inl = result.pairs, result.residual_px, result.inliers
    pred_x, pred_y = m.project(p.alt, p.az)
    xu = (W - 1) - p.x if m.mirror else p.x
    r_det = np.hypot(xu - m.cx, p.y - m.cy)
    alt_c, az_c = m.sky_to_camera(p.alt, p.az)

    stars = sky_stars(site.lat, site.lon, frame.utc, min_alt=0.0, max_mag=6.5, model=m)
    names = {int(h): n for h, n in zip(stars.hip, stars.names)}
    matched_hip = {int(h) for h, k in zip(p.hip, inl) if k}
    s = np.flatnonzero(stars.in_frame)
    catalog = [{"hip": int(stars.hip[i]), "name": stars.names[i] if not stars.names[i].startswith("HIP ") else None,
                "mag": _r(stars.mag[i]), "alt": _r(stars.alt[i], 3), "az": _r(stars.az[i], 3),
                "x": _r(stars.x[i], 1), "y": _r(stars.y[i], 1), "matched": int(stars.hip[i]) in matched_hip} for i in s]

    pairs = []
    for i in range(len(p)):
        nm = names.get(int(p.hip[i]), "")
        pairs.append({"hip": int(p.hip[i]), "name": None if not nm or nm.startswith("HIP ") else nm, "mag": _r(p.mag[i]),
                      "alt": _r(p.alt[i], 3), "az": _r(p.az[i], 3), "x": _r(p.x[i]), "y": _r(p.y[i]),
                      "xp": _r(pred_x[i]), "yp": _r(pred_y[i]), "dx": _r(p.x[i] - pred_x[i], 3), "dy": _r(p.y[i] - pred_y[i], 3),
                      "res": _r(res[i], 3), "inlier": bool(inl[i]), "r": _r(r_det[i], 1), "alt_cam": _r(alt_c[i], 3)})

    det = frame.detections
    order = det.order[:50000]
    detections = {"x": [round(float(v), 1) for v in det.x[order]], "y": [round(float(v), 1) for v in det.y[order]],
                  "flux": [round(float(v), 1) for v in det.flux[order]]}

    azg = np.linspace(0, 360, 721)
    grid = {"alt": [], "az": [], "labels": []}
    for a0 in range(0, 90, 10):
        ln = _line(m, np.full_like(azg, a0 + (0.05 if a0 == 0 else 0.0)), azg, W, H)
        if ln:
            grid["alt"].append({"alt": a0, "line": ln})
    altg = np.linspace(0.05, 90, 180)
    for z0 in range(0, 360, 30):
        ln = _line(m, altg, np.full_like(altg, float(z0)), W, H)
        if ln:
            grid["az"].append({"az": z0, "line": ln})
    for a0 in (10, 30, 50, 70):
        x, y = m.project(np.array([a0 + 1.0]), np.array([45.0]))
        if np.isfinite(x[0]) and 0 <= x[0] < W and 0 <= y[0] < H:
            grid["labels"].append({"text": f"{a0}°", "x": _r(x[0], 1), "y": _r(y[0], 1)})
    cardinal = []
    for z0, name in ((0, "N"), (45, "NE"), (90, "E"), (135, "SE"), (180, "S"), (225, "SW"), (270, "W"), (315, "NW")):
        x, y = m.project(np.array([4.0]), np.array([float(z0)]))
        if np.isfinite(x[0]) and 0 <= x[0] < W and 0 <= y[0] < H:
            cardinal.append({"text": name, "x": _r(x[0], 1), "y": _r(y[0], 1)})

    constellations = []
    for abbr, polylines in constellation_lines().items():
        lines, pts = [], []
        for line in polylines:
            ra, dec = np.array(line).T
            la, lz = _radec_polyline_altaz(ra, dec, site.lat, site.lon, frame.utc, n=12)
            if np.all(la < 0):
                continue
            la = np.where(la >= 0, la, -1.0)
            ln = _line(m, la, lz, W, H)
            if ln:
                lines.append(ln)
                pts += [q for q in ln if q is not None]
        if lines:
            c = np.mean(np.array(pts), axis=0) if pts else None
            constellations.append({"abbr": abbr, "lines": lines, "label": [round(float(c[0]), 1), round(float(c[1]), 1)] if c is not None else None})

    th = np.radians(np.linspace(0, 95, 191))
    f = m.f
    curves = {"theta_deg": [round(float(v), 2) for v in np.degrees(th)],
              "model": [_r(v, 2) for v in m.radius(th)],
              "equidistant": [_r(v, 2) for v in f * th],
              "equisolid": [_r(v, 2) for v in 2 * f * np.sin(th / 2)],
              "stereographic": [_r(v, 2) for v in 2 * f * np.tan(th / 2)],
              "orthographic": [_r(v, 2) if t <= math.pi / 2 else None for v, t in zip(f * np.sin(th), th)],
              # plate scales in arcmin per pixel: radial 1/(dr/dθ), tangential sinθ/r (θ in degrees)
              "scale_radial": [_r(60.0 / (v * math.pi / 180), 3) for v in m.dradius(th)],
              "scale_tangential": [_r(60.0 * math.sin(t) / (r * math.pi / 180), 3) if t > 1e-3 else _r(60.0 / (f * math.pi / 180), 3)
                                   for t, r in zip(th, m.radius(th))]}

    summary = result.summary()
    cas = summary.pop("cascade", None) or result.info.get("cascade") or {}
    zx, zy = m.zenith_pixel
    centre_x = (W - 1) - m.cx if m.mirror else m.cx
    return {
        "model": m.to_dict(),
        "summary": json.loads(json.dumps(summary, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))),
        "cascade": json.loads(json.dumps(cas, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))),
        "frame": {"name": frame.path.name, "width": W, "height": H, "utc_mid": frame.utc.isoformat() + "Z",
                  "exposure_s": frame.exposure_s, "n_detections": len(det), "fwhm_kernel": frame.info.get("fwhm"),
                  "fwhm_measured": frame.info.get("fwhm_measured"), "presmooth_sigma": frame.info.get("presmooth_sigma"),
                  "tiles": (frame.info.get("detect_kwargs") or {}).get("tiles"), "disc": [_r(v, 1) for v in frame.disc]},
        "site": {"lat": site.lat, "lon": site.lon, "elev": site.elev},
        "timings": timings,
        "markers": {"zenith": [_r(zx, 1), _r(zy, 1)], "centre": [_r(centre_x, 1), _r(m.cy, 1)]},
        "layers": {"detections": detections, "pairs": pairs, "catalog": catalog, "grid": grid, "cardinal": cardinal,
                   "constellations": constellations},
        "curves": curves,
    }


def main(job_dir: str) -> int:
    job = Path(job_dir)
    params = json.loads((job / "params.json").read_text())
    log = open(job / "log.txt", "a", buffering=1)
    sys.stdout = log
    sys.stderr = log
    t0 = time.perf_counter()
    try:
        from .. import config
        from ..bootstrap import CalibrationError, Site, calibrate
        from ..detect import load_frame
        from .. import plots

        path = job / params["file"]
        site = Site(float(params["lat"]), float(params["lon"]), float(params.get("elev") or 0.0))
        t = datetime.fromisoformat(params["utc"]).replace(tzinfo=timezone.utc) if params.get("utc") else None
        opts = {"max_time": float(params.get("max_time") or 40.0), "parity": params.get("parity") or "auto"}
        fwhm = params.get("fwhm") or "auto"
        print(f"ascal {__import__('ascal').__version__}: reading {params['original_name']} and detecting stars", flush=True)
        with config.options(**opts):
            frame = load_frame(path, time=t, exposure_s=params.get("exposure"), keep_image=True,
                               fwhm=fwhm if fwhm == "auto" else float(fwhm))
            frame.path = Path(params["original_name"])
            t_det = time.perf_counter() - t0
            print(f"searching the camera pose ({len(frame.detections)} detections, {t_det:.1f} s)", flush=True)
            result = calibrate([frame], site, verbose=True, max_time=opts["max_time"])   # budget for the search, detection excluded
        t_cal = time.perf_counter() - t0 - t_det
        print("preparing the layers and figures", flush=True)
        frame.path = path
        scale = _display_image(path, frame.gray, job / "display.jpg")
        frame.path = Path(params["original_name"])
        timings = {"detection_s": round(t_det, 2), "calibration_s": round(t_cal, 2), "total_s": round(t_det + t_cal, 2)}
        out = build_result(frame, result, site, timings=timings)
        out["display_scale"] = scale
        out["original_name"] = params["original_name"]
        result.model.save(job / "calibration.json")
        with open(job / "pairs.csv", "w") as fh:
            fh.write("hip,name,mag,alt,az,x,y,x_pred,y_pred,residual_px,inlier\n")
            for q in out["layers"]["pairs"]:
                fh.write(f"{q['hip']},{q['name'] or ''},{q['mag']},{q['alt']},{q['az']},{q['x']},{q['y']},{q['xp']},{q['yp']},{q['res']},{int(q['inlier'])}\n")
        try:
            plots.calibration_panel(frame, result, site, path=job / "panel.png",
                                    title=f"{params['original_name']}   {frame.utc:%Y-%m-%d %H:%M:%S} UTC")
        except Exception as exc:  # the panel is a convenience; never fail the job for it
            print(f"(panel not drawn: {exc})", flush=True)
        (job / "result.json").write_text(json.dumps(out, separators=(",", ":")))
        print(f"done in {time.perf_counter() - t0:.1f} s", flush=True)
        return 0
    except Exception as exc:
        from ..bootstrap import CalibrationError
        kind = "calibration" if isinstance(exc, CalibrationError) else ("input" if isinstance(exc, (ValueError, OSError)) else "internal")
        (job / "error.json").write_text(json.dumps({"kind": kind, "message": str(exc),
                                                     "trace": traceback.format_exc() if kind == "internal" else None}))
        print(f"failed: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
