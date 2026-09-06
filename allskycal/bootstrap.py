"""Zero-shot calibration: from one or more night frames, with no prior calibration, mask or labels.

Stages
------
1. Sky disc -> initial optical centre and equidistant focal length ``f = 2 R_h / pi``.
2. Blind pose search: image rotation ``psi`` (0-360 deg in 3 deg steps) x displacement of the zenith
   with respect to the disc centre (+-210 px in 30 px steps; a tilted camera moves the zenith) x focal
   scale (0.88-1.20).  Score: number of catalogue stars brighter than magnitude 3 above 45 deg that
   fall within 25 px of one of the 200 brightest detections.  The best distinct candidates are refined
   (0.5 deg, 5 px) and validated by a first round of association.
3. Progressive association and fitting: mag <= 3.5 / 30 px -> mag <= 4.5 / 25 px -> mag <= 5.5 / 12 px
   -> 7 px, always with unique and mutual pairs and a robust loss; per-band clipping at the end.

Quality gate: at least 80 pairs and a median residual below 2 px (a fit that barely passes a looser
gate can hide a wrong pose; see the paper).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .catalog import Catalog, load_catalog, sky_stars
from .detect import Frame
from .match import Pairs, associate, unique_mutual
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


# ---------------------------------------------------------------------- stage 2: blind pose search

def blind_pose(frames: Sequence[Frame], site: Site, catalog: Catalog, *, model0: CameraModel, verbose: bool = True,
               psi_step: float = 3.0, shift_max: int = 210, shift_step: int = 30,
               focal_scales: Sequence[float] = (0.88, 0.96, 1.04, 1.12, 1.20), n_candidates: int = 4) -> Tuple[CameraModel, Dict[str, Any]]:
    """Find image rotation, zenith displacement and focal scale by counting bright-star coincidences."""
    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    cx, cy, f0 = model0.cx, model0.cy, model0.f
    w, h = model0.width, model0.height
    bright = [sky_stars(site.lat, site.lon, fr.utc, min_alt=45.0, max_mag=3.0, catalog=catalog) for fr in frames]
    trees = [cKDTree(fr.detections.xy[fr.detections.order[:200]]) for fr in frames]
    n_exp = sum(len(b) for b in bright)
    if n_exp < 4:
        raise CalibrationError("Fewer than 4 catalogue stars brighter than magnitude 3 above 45 deg: check site and time")
    psis = np.arange(0.0, 360.0, psi_step)
    cos_t, sin_t = np.cos(np.radians(psis)), np.sin(np.radians(psis))
    cands = []
    for fs in focal_scales:
        m = model0.copy(cx=0.0, cy=0.0, f=f0 * fs, psi=0.0, tau_x=0.0, tau_y=0.0)
        uv = []
        for b in bright:
            px, py = m.project(b.alt, b.az)
            ok = np.isfinite(px)
            uv.append((px[ok], py[ok]))
        for dx in range(-shift_max, shift_max + 1, shift_step):
            for dy in range(-shift_max, shift_max + 1, shift_step):
                counts = np.zeros(psis.size, int)
                for (u, v), tr in zip(uv, trees):
                    # psi rotates (u, v) about the centre: project once and rotate for all orientations
                    U = u[None, :] * cos_t[:, None] - v[None, :] * sin_t[:, None] + cx + dx
                    V = u[None, :] * sin_t[:, None] + v[None, :] * cos_t[:, None] + cy + dy
                    d, _ = tr.query(np.column_stack([U.ravel(), V.ravel()]), k=1)
                    counts += (d.reshape(U.shape) <= 25.0).sum(axis=1)
                k = int(np.argmax(counts))
                cands.append((int(counts[k]), float(psis[k]), dx, dy, fs))
    cands.sort(key=lambda c: -c[0])
    distinct: List[Tuple[int, float, int, int, float]] = []
    for c in cands:
        if all(abs(((c[1] - d[1] + 180) % 360) - 180) > 10 or abs(c[2] - d[2]) + abs(c[3] - d[3]) > 60 for d in distinct):
            distinct.append(c)
        if len(distinct) >= n_candidates:
            break

    def score(psi: float, dx: float, dy: float, fs: float, radius: float) -> int:
        m = model0.copy(cx=cx + dx, cy=cy + dy, f=f0 * fs, psi=psi, tau_x=0.0, tau_y=0.0)
        n = 0
        for b, tr in zip(bright, trees):
            px, py = m.project(b.alt, b.az)
            ok = np.isfinite(px)
            if ok.any():
                d, _ = tr.query(np.column_stack([px[ok], py[ok]]), k=1)
                n += int((d <= radius).sum())
        return n

    trials = []
    for n0, psi0, dx0, dy0, fs0 in distinct:
        best = (n0, psi0, dx0, dy0)
        for psi in np.arange(psi0 - 3.0, psi0 + 3.01, 0.5):
            for dx in range(dx0 - 15, dx0 + 16, 5):
                for dy in range(dy0 - 15, dy0 + 16, 5):
                    n = score(psi, dx, dy, fs0, 15.0)
                    if n > best[0]:
                        best = (n, psi, dx, dy)
        n0, psi0, dx0, dy0 = best
        m = model0.copy(cx=cx + dx0, cy=cy + dy0, f=f0 * fs0, psi=psi0, tau_x=0.0, tau_y=0.0)
        pairs = Pairs.concatenate([associate(m, sky_stars(site.lat, site.lon, fr.utc, min_alt=25.0, max_mag=3.5, catalog=catalog, model=m),
                                             fr.detections, radius=30.0, n_brightest=400, frame_index=i) for i, fr in enumerate(frames)])
        n1 = 0
        if len(pairs) >= 8:
            m = fit(m, pairs.alt, pairs.az, pairs.x, pairs.y, loss="soft_l1", f_scale=8.0, fixed=("k",))
            n1 = sum(len(associate(m, sky_stars(site.lat, site.lon, fr.utc, min_alt=15.0, max_mag=4.5, catalog=catalog, model=m),
                                   fr.detections, radius=12.0, n_brightest=1500)) for fr in frames)
        trials.append((n1, n0, m, (psi0, dx0, dy0, fs0)))
        _log(verbose, f"  candidate psi {psi0:.1f} deg, zenith shift ({dx0}, {dy0}) px, focal x{fs0:.2f}: "
                      f"{n0}/{n_exp} bright coincidences -> {n1} pairs of mag <= 4.5 within 12 px")
    trials.sort(key=lambda t: -t[0])
    n1, n0, model, pose = trials[0]
    info = {"pose_search_s": round(time.perf_counter() - t0, 1), "pose_candidates": [t[3] for t in trials], "pose_validation_pairs": int(n1),
            "bright_coincidences": f"{n0}/{n_exp}"}
    if n1 < 40:
        raise CalibrationError(f"Blind pose search found no reliable pose ({n1} validation pairs): few stars, clouds, or wrong site/time")
    _log(verbose, f"  initial pose: psi {model.psi:.1f} deg, {n1} validation pairs ({info['pose_search_s']} s)")
    return model, info


# ---------------------------------------------------------------------- stage 3: progressive refinement

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
    pairs = Pairs.concatenate([])
    for k, (max_mag, min_alt, radius, n_det, f_scale) in enumerate(stages):
        pairs = Pairs.concatenate([associate(model, sky_stars(site.lat, site.lon, fr.utc, min_alt=min_alt, max_mag=max_mag, catalog=catalog, model=model),
                                             fr.detections, radius=radius, n_brightest=n_det, frame_index=i) for i, fr in enumerate(frames)])
        if len(pairs) < 12:
            raise CalibrationError(f"Stage {k}: only {len(pairs)} pairs; cannot fit")
        if decentering and k >= 1 and model.p is None:
            model = model.copy(p=np.zeros(2))
        model = fit(model, pairs.alt, pairs.az, pairs.x, pairs.y, loss="soft_l1", f_scale=f_scale)
        d = pairs.residuals(model)
        _log(verbose, f"  stage {k}: mag <= {max_mag}, radius {radius:.0f} px -> {len(pairs)} pairs, median {np.median(d):.2f} px, p90 {np.percentile(d, 90):.2f} px")
    model, keep = robust_fit(model, pairs.alt, pairs.az, pairs.x, pairs.y)
    d = pairs.residuals(model)[keep]
    info = {"refine_s": round(time.perf_counter() - t0, 1), "n_candidates": len(pairs), "n_inliers": int(keep.sum())}
    _log(verbose, f"  final: {keep.sum()} pairs, median {np.median(d):.2f} px, rms {np.sqrt(np.mean(d * d)):.2f} px")
    if keep.sum() < MIN_PAIRS or np.median(d) > MAX_MEDIAN_PX:
        raise CalibrationError(f"Calibration not reliable: {keep.sum()} pairs, median {np.median(d):.2f} px "
                               f"(gate: >= {MIN_PAIRS} pairs and median <= {MAX_MEDIAN_PX} px)")
    return CalibrationResult(model, pairs, keep, info)


# ---------------------------------------------------------------------- full pipeline

def calibrate(frames: Sequence[Frame], site: Site, *, decentering: bool = False, catalog: Optional[Catalog] = None,
              initial: Optional[CameraModel] = None, verbose: bool = True) -> CalibrationResult:
    """Zero-shot calibration from detected frames (see module docstring).

    With ``initial`` (an approximate model, e.g. from a previous calibration) the blind pose search is
    skipped and only the progressive refinement runs.
    """
    t_start = time.perf_counter()
    cat = catalog or load_catalog()
    h, w = frames[0].shape
    for fr in frames:
        if fr.shape != (h, w):
            raise CalibrationError("All frames must have the same size")
    info: Dict[str, Any] = {"n_frames": len(frames), "frames": [fr.path.name for fr in frames],
                            "n_detections": [len(fr.detections) for fr in frames]}
    if initial is None:
        cx = float(np.median([fr.disc[0] for fr in frames]))
        cy = float(np.median([fr.disc[1] for fr in frames]))
        rh = float(np.median([fr.disc[2] for fr in frames]))
        info["sky_disc"] = [round(cx, 1), round(cy, 1), round(rh, 1)]
        _log(verbose, f"sky disc: centre ({cx:.0f}, {cy:.0f}), radius {rh:.0f} px -> f0 = {2 * rh / math.pi:.0f} px/rad")
        model0 = CameraModel.initial(w, h, cx, cy, rh)
        model, pose_info = blind_pose(frames, site, cat, model0=model0, verbose=verbose)
        info.update(pose_info)
    else:
        model = initial.copy(width=w, height=h)
    result = refine(model, frames, site, cat, decentering=decentering, verbose=verbose)
    result.info.update(info)
    result.info["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    result.model.meta.update({"calibration": "allskycal zero-shot", "frames": info["frames"], "site": vars(site),
                              "n_pairs": int(result.inliers.sum()), "median_px": round(float(np.median(result.residual_px[result.inliers])), 3)})
    return result


def evaluate(model: CameraModel, frames: Sequence[Frame], site: Site, *, max_mag: float = 5.5, min_alt: float = 3.0,
             radius: float = 10.0, photometric_gate: bool = True, catalog: Optional[Catalog] = None) -> Pairs:
    """Associate stars of ``frames`` with a fixed model (no fitting) and return the pairs for scoring.

    This is how a calibration is checked on frames it was not fitted on: unique, mutual pairs within
    ``radius`` pixels, optionally gated by a per-frame photometric zero point.
    """
    from .match import frame_zero_point

    cat = catalog or load_catalog()
    parts = []
    for i, fr in enumerate(frames):
        stars = sky_stars(site.lat, site.lon, fr.utc, min_alt=min_alt, max_mag=max_mag, catalog=cat, model=model)
        zp = frame_zero_point(model, stars, fr.detections) if photometric_gate else None
        parts.append(associate(model, stars, fr.detections, radius=radius, zp=zp, frame_index=i))
    return Pairs.concatenate(parts)
