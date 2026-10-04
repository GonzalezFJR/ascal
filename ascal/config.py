"""Run-time options of the calibration and a few helpers shared by the modules.

The defaults are the ones validated on the multi-camera test set (version 1.0). They can be changed for a block
of code with :func:`options`::

    with ascal.config.options(max_time=60, refraction=False):
        result = ascal.calibrate([frame], site)

Options
-------
max_time        seconds allowed for one calibration (detection excluded when called from the API); 40
refraction      compare detections with apparent (refracted) altitudes; True
scale_radii     scale every pixel radius and step with the plate scale (f / 1005 px/rad, the reference camera); True
presmooth       "auto": Gaussian pre-smoothing when the measured star FWHM is below 2.2 px (undersampled stars); None disables
noise           "mad": detection threshold from the MAD of the background-subtracted frame; "rms": photutils box rms (0.3.0)
box_scale       background box proportional to the plate scale; True
tiles           "auto" (3 x 3 tiles above 16 Mpx, processed in parallel) or an integer
workers         processes for tiled detection; None = min(tiles^2, CPU count)
min_pairs       quality gate inside the refinement; 30
stage_repeats   passes over the progressive association stages; 2
parity          "auto" (try both), "direct" or "mirror"
disc            fixed sky disc (cx, cy, r) in pixels, or None
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Iterator

import numpy as np

REF_F = 1005.0          # px/rad of the reference camera (ZRO); radii of the 0.3.0 procedure refer to it

DEFAULTS: Dict[str, Any] = {
    "max_time": 40.0,
    "refraction": True,
    "scale_radii": True,
    "presmooth": "auto",
    "noise": "mad",
    "box_scale": True,
    "tiles": "auto",
    "workers": None,
    "min_pairs": 30,
    "stage_repeats": 2,
    "parity": "auto",
    "disc": None,
}

CFG: Dict[str, Any] = dict(DEFAULTS)


def get(key: str, default: Any = None) -> Any:
    return CFG.get(key, default)


@contextlib.contextmanager
def options(**changes: Any) -> Iterator[Dict[str, Any]]:
    """Temporarily change options (unknown names raise KeyError)."""
    unknown = set(changes) - set(DEFAULTS) - {"focal_scales", "psi_step", "min_pose_pairs", "k_priors", "elev"}
    if unknown:
        raise KeyError(f"unknown ascal options: {sorted(unknown)}")
    saved = dict(CFG)
    CFG.update(changes)
    try:
        yield CFG
    finally:
        CFG.clear()
        CFG.update(saved)


def radius_scale(f: float) -> float:
    """Factor applied to the pixel radii of the 0.3.0 procedure for a camera of focal length ``f`` (px/rad)."""
    return float(np.clip(f / REF_F, 0.25, 4.0)) if CFG.get("scale_radii") else 1.0


def refraction_deg(alt_true_deg, elev_m: float = 0.0):
    """Atmospheric refraction (deg) for a true altitude, Saemundsson (1986), scaled to the pressure at ``elev_m``."""
    h = np.asarray(alt_true_deg, float)
    p = 1010.0 * np.exp(-float(elev_m or 0.0) / 8400.0)
    r_arcmin = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) * (p / 1010.0)
    return np.where(h > -1.0, r_arcmin / 60.0, 0.0)


def n_workers(n_tasks: int) -> int:
    w = CFG.get("workers")
    return max(1, min(n_tasks, int(w) if w else (os.cpu_count() or 1)))


def robust_disc_candidates(gray, n=4):
    """Candidate sky discs (cx, cy, r) from Hough circles on a downsampled, log-stretched frame,
    ranked by the brightness contrast across the circle (inside minus outside)."""
    import cv2

    h, w = gray.shape
    scale = max(1, int(round(min(h, w) / 480)))
    small = cv2.resize(gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(small, [1, 99.5])
    s = np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1)
    s = np.log1p(30 * s) / np.log1p(30)
    s8 = cv2.GaussianBlur((255 * s).astype(np.uint8), (0, 0), 2)
    sh, sw = s8.shape
    m = min(sh, sw)
    cands = []
    for p2 in (60, 45, 30, 20):
        c = cv2.HoughCircles(s8, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m * 0.05, param1=60, param2=p2,
                             minRadius=int(0.3 * m), maxRadius=int(0.75 * max(sh, sw)))
        if c is not None:
            cands += [tuple(v) for v in c[0][:40]]
        if len(cands) >= 20:
            break
    yy, xx = np.mgrid[0:sh, 0:sw]
    out = []
    for cx, cy, r in cands:
        if not (0.15 * sw < cx < 0.85 * sw and 0.15 * sh < cy < 0.85 * sh):
            continue
        d = np.hypot(xx - cx, yy - cy)
        inn = s[(d > 0.88 * r) & (d < 0.97 * r)]
        out_ = s[(d > 1.03 * r) & (d < 1.12 * r)]
        core = s[d < 0.5 * r]
        if inn.size < 50 or out_.size < 50:
            continue
        contrast = float(np.median(inn) - np.median(out_))
        sim = -abs(float(np.median(inn)) - float(np.median(core)))
        out.append((contrast + 0.5 * sim, cx * scale, cy * scale, r * scale))
    out.sort(key=lambda t: -t[0])
    res = []
    for sc, cx, cy, r in out:
        if all(abs(cx - a) + abs(cy - b) > 0.05 * r or abs(r - c) > 0.05 * r for a, b, c in res):
            res.append((float(cx), float(cy), float(r)))
        if len(res) >= n:
            break
    return res
