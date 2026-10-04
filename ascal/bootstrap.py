"""Zero-shot calibration: from one or more night frames, with no prior calibration, mask or labels.

Version 1.0 runs an incremental cascade (:mod:`ascal.fast`): the hypotheses about the image (sky disc, parity,
detection kernel, radial prior) are tried in order of prior likelihood and the search stops at the first one
that passes the quality gate, within a time budget (``config.max_time``, 40 s by default).  Each step is:

1. Blind pose: rotation, zenith displacement and focal scale found by counting bright catalogue stars that land
   on detections, using a distance transform of the detections (both parities at once).
2. Progressive association and fitting (this module, :func:`refine`): mag <= 4.5 / 25 px -> mag <= 5.5 /
   12 px -> 7 px, unique and mutual pairs, robust loss, per-band clipping.  Radii scale with the plate scale
   (they refer to the 1005 px/rad camera of the paper).
3. Gate: tight fit and either fair coverage of the stars bright enough to be seen in the frame, or a pose that
   clearly beats every rival.

With several frames, the frame with most detections is calibrated by the cascade and the model is then refined
on all of them together.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .catalog import Catalog, load_catalog, sky_stars
from .detect import Frame
from .match import Pairs, associate
from .model import CameraModel, band_statistics, fit, residuals, robust_fit


@dataclass
class Site:
    lat: float          # degrees north
    lon: float          # degrees east
    elev: float = 0.0   # metres (informational)


@dataclass
class CalibrationResult:
    model: CameraModel
    pairs: Pairs
    inliers: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def residual_px(self) -> np.ndarray:
        return self.pairs.residuals(self.model)

    def summary(self) -> Dict[str, Any]:
        d = self.residual_px[self.inliers]
        return {"n_pairs": int(self.inliers.sum()), "median_px": float(np.median(d)), "rms_px": float(np.sqrt(np.mean(d * d))),
                "p90_px": float(np.percentile(d, 90)), "within_1px": float(np.mean(d < 1.0)),
                "bands": band_statistics(self.pairs.alt[self.inliers], d), **self.info}


class CalibrationError(RuntimeError):
    pass


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


# ---------------------------------------------------------------------- progressive refinement
MIN_PAIRS = 80          # quality gate
MAX_MEDIAN_PX = 2.0

STAGES = (
    # max_mag, min_alt, radius_px, n_brightest_detections, f_scale
    (4.5, 15.0, 25.0, 1500, 3.0),
    (5.5, 5.0, 12.0, None, 3.0),
    (5.5, 3.0, 7.0, None, 1.5),
)


def refine(model: CameraModel, frames: Sequence[Frame], site: Site, catalog: Catalog, *, decentering: bool = False,
           stages: Sequence[Tuple] = STAGES, verbose: bool = True) -> CalibrationResult:
    """Progressive association and fitting from an approximate model; ends with per-band clipping."""
    t0 = time.perf_counter()
    S = config.radius_scale(model.f)
    stages = tuple(stages) * int(config.get("stage_repeats", 1))
    pairs = Pairs.concatenate([])
    for k, (max_mag, min_alt, radius, n_det, f_scale) in enumerate(stages):
        radius, f_scale = radius * S, f_scale * S
        pairs = Pairs.concatenate([associate(model, sky_stars(site.lat, site.lon, fr.utc, min_alt=min_alt, max_mag=max_mag, catalog=catalog, model=model),
                                             fr.detections, radius=radius, n_brightest=n_det, frame_index=i) for i, fr in enumerate(frames)])
        if len(pairs) < 12:
            raise CalibrationError(f"Stage {k}: only {len(pairs)} pairs; cannot fit")
        if decentering and k >= 1 and model.p is None:
            model = model.copy(p=np.zeros(2))
        model = fit(model, pairs.alt, pairs.az, pairs.x, pairs.y, loss="soft_l1", f_scale=f_scale)
        d = pairs.residuals(model)
        _log(verbose, f"  stage {k}: mag <= {max_mag}, radius {radius:.0f} px -> {len(pairs)} pairs, median {np.median(d):.2f} px, p90 {np.percentile(d, 90):.2f} px")
    model, keep = robust_fit(model, pairs.alt, pairs.az, pairs.x, pairs.y, floor_px=3.0 * S)
    d = pairs.residuals(model)[keep]
    info = {"refine_s": round(time.perf_counter() - t0, 1), "n_candidates": len(pairs), "n_inliers": int(keep.sum())}
    _log(verbose, f"  final: {keep.sum()} pairs, median {np.median(d):.2f} px, rms {np.sqrt(np.mean(d * d)):.2f} px")
    min_pairs, max_med = config.get("min_pairs", MIN_PAIRS), config.get("max_median_px", MAX_MEDIAN_PX)
    if keep.sum() < min_pairs or np.median(d) > max_med:
        raise CalibrationError(f"Calibration not reliable: {keep.sum()} pairs, median {np.median(d):.2f} px "
                               f"(gate: >= {min_pairs} pairs and median <= {max_med} px)")
    return CalibrationResult(model, pairs, keep, info)


# ---------------------------------------------------------------------- full pipeline

def _unmirror_result(result: CalibrationResult, width: int) -> CalibrationResult:
    """Express a result obtained on horizontally mirrored detections in the coordinates of the original image."""
    result.model = result.model.copy(mirror=not result.model.mirror)
    result.pairs.x = (width - 1) - result.pairs.x
    return result


def calibrate(frames: Sequence[Frame], site: Site, *, decentering: bool = False, catalog: Optional[Catalog] = None,
              initial: Optional[CameraModel] = None, verbose: bool = True, max_time: Optional[float] = None) -> CalibrationResult:
    """Zero-shot calibration from detected frames (see module docstring).

    ``initial`` (an approximate model, e.g. a previous calibration of the same camera) skips the cascade and
    runs only the progressive refinement.  ``max_time`` overrides ``config.max_time`` (seconds); the clock starts
    when this function is called.  Frames should be loaded with ``keep_image=True`` so that the cascade can try a
    wider detection kernel when needed.  The returned model maps sky to the pixels of the images as given (a
    mirrored camera is handled by ``CameraModel.mirror``).
    """
    from .fast import calibrate_fast, mirrored

    t_start = time.perf_counter()
    cat = catalog or load_catalog()
    h, w = frames[0].shape
    for fr in frames:
        if fr.shape != (h, w):
            raise CalibrationError("All frames must have the same size")
    with config.options(elev=site.elev):
        if initial is not None:
            m0 = initial.copy(width=w, height=h)
            if m0.mirror:
                res = refine(m0.copy(mirror=False), [mirrored(fr) for fr in frames], site, cat, decentering=decentering, verbose=verbose)
                res = _unmirror_result(res, w)
            else:
                res = refine(m0, frames, site, cat, decentering=decentering, verbose=verbose)
            info: Dict[str, Any] = {"cascade": None}
        else:
            lead = max(range(len(frames)), key=lambda i: len(frames[i].detections))
            budget = float(max_time if max_time is not None else config.get("max_time", 40.0))
            res, log = calibrate_fast(frames[lead], site, verbose=verbose, catalog=cat, deadline=t_start + budget)
            info = {"cascade": log}
            parity = log["parity"]
            if len(frames) > 1 or decentering:
                group = [fr if parity == "direct" else mirrored(fr) for fr in frames]
                res = refine(res.model, group, site, cat, decentering=decentering, verbose=verbose)
            if parity == "mirror":
                res = _unmirror_result(res, w)
    info.update({"n_frames": len(frames), "frames": [fr.path.name for fr in frames],
                 "n_detections": [len(fr.detections) for fr in frames], "fwhm": [fr.info.get("fwhm") for fr in frames],
                 "fwhm_measured": [fr.info.get("fwhm_measured") for fr in frames], "elapsed_s": round(time.perf_counter() - t_start, 1)})
    res.info.update(info)
    d = res.residual_px[res.inliers]
    res.model.meta.update({"calibration": "ascal 1.0 cascade", "frames": info["frames"], "site": vars(site),
                           "n_pairs": int(res.inliers.sum()), "median_px": round(float(np.median(d)), 3),
                           "detection_fwhm_px": info["fwhm"], "refraction": bool(config.get("refraction"))})
    return res


def evaluate(model: CameraModel, frames: Sequence[Frame], site: Site, *, max_mag: float = 5.5, min_alt: float = 3.0,
             radius: float = 10.0, photometric_gate: bool = True, catalog: Optional[Catalog] = None) -> Pairs:
    """Associate stars of ``frames`` with a fixed model (no fitting) and return the pairs for scoring.

    This is how a calibration is checked on frames it was not fitted on: unique, mutual pairs within
    ``radius`` pixels, optionally gated by a per-frame photometric zero point.
    """
    from .match import frame_zero_point

    cat = catalog or load_catalog()
    parts = []
    with config.options(elev=site.elev):
        for i, fr in enumerate(frames):
            stars = sky_stars(site.lat, site.lon, fr.utc, min_alt=min_alt, max_mag=max_mag, catalog=cat, model=model)
            zp = frame_zero_point(model, stars, fr.detections) if photometric_gate else None
            parts.append(associate(model, stars, fr.detections, radius=radius, zp=zp, frame_index=i))
    return Pairs.concatenate(parts)
