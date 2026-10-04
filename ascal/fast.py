"""Incremental ("cascade") single-frame calibration with a raster pose search (referee-round experiment).

Instead of evaluating every hypothesis (sky disc x tolerance x parity x radial prior x detection kernel), the
hypotheses are tried in order of prior likelihood and the search stops at the first one whose calibration passes
the gate.  Each step is cheap:

* detection runs once (pre-smoothing for undersampled stars, MAD noise, background box scaled to the plate scale);
* the blind pose search counts coincidences on a distance transform of the detections, so all rotations, shifts and
  focal scales of one hypothesis cost about a second, for both parities at once;
* a candidate pose is accepted only after a quick refinement passes a gate that scales with the number of catalogue
  stars expected in the frame.

Order of the cascade (stop at the first accepted calibration):
  1. disc from ``sky_disc`` (as in 0.3.0), both parities ranked by pose score;
  2. other discs: inscribed and circumscribed circles centred on the sensor, then two Hough circles;
  3. a second detection with a 1.5x kernel, repeating 1-2.
Once a pose is accepted the three radial priors are refined and the best is kept.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import config
from .bootstrap import CalibrationError, CalibrationResult, Site, _log, refine
from .catalog import load_catalog, sky_stars
from .detect import Frame
from .match import associate
from .model import CameraModel, fit

RADIAL_PRIORS = ([-0.03, 0.0], [0.04, 0.0], [0.0, 0.0])


class DistMap:
    """Distance (px) from every (downsampled) pixel to the nearest of the brightest detections."""

    def __init__(self, xy: np.ndarray, shape: Tuple[int, int], q: int):
        import cv2
        h, w = shape
        self.q, self.H, self.W = q, h // q + 1, w // q + 1
        img = np.full((self.H, self.W), 255, np.uint8)
        xi = np.clip((xy[:, 0] / q).astype(int), 0, self.W - 1)
        yi = np.clip((xy[:, 1] / q).astype(int), 0, self.H - 1)
        img[yi, xi] = 0
        self.d = cv2.distanceTransform(img, cv2.DIST_L2, 3) * q

    def hits(self, X: np.ndarray, Y: np.ndarray, tol: float) -> np.ndarray:
        xi = np.rint(X / self.q).astype(np.int64)
        yi = np.rint(Y / self.q).astype(np.int64)
        ok = (xi >= 0) & (xi < self.W) & (yi >= 0) & (yi < self.H)
        out = np.zeros(X.shape, bool)
        out[ok] = self.d[yi[ok], xi[ok]] <= tol
        return out


def mirrored(fr: Frame) -> Frame:
    g = copy.copy(fr)
    w = fr.shape[1]
    g.detections = copy.copy(fr.detections)
    g.detections.x = (w - 1) - fr.detections.x
    g.disc = ((w - 1) - fr.disc[0], fr.disc[1], fr.disc[2])
    if fr.gray is not None:
        g.gray = np.ascontiguousarray(fr.gray[:, ::-1])
    g.info = dict(fr.info, parity="mirror")
    return g


def raster_pose(fr: Frame, site: Site, cat, disc, *, n_det: int = 250, top: int = 3) -> List[Tuple[int, CameraModel]]:
    """Exhaustive pose search (focal scale, rotation, zenith shift) on a distance map of the detections."""
    h, w = fr.shape
    cx0, cy0, rh = disc
    m0 = CameraModel.initial(w, h, cx0, cy0, rh)
    f0 = m0.f
    tol = max(2.0, 0.021 * f0)                       # ~1.2 deg
    q = max(1, int(tol // 3))
    dm = DistMap(fr.detections.xy[fr.detections.order[:n_det]], fr.shape, q)
    b = sky_stars(site.lat, site.lon, fr.utc, min_alt=40.0, max_mag=3.0, catalog=cat)
    if len(b.alt) < 6:
        b = sky_stars(site.lat, site.lon, fr.utc, min_alt=30.0, max_mag=3.5, catalog=cat)
    psis = np.radians(np.arange(0.0, 360.0, 1.5))
    c, s = np.cos(psis)[:, None], np.sin(psis)[:, None]
    shift_max, step = 0.22 * f0, tol
    shifts = np.arange(-shift_max, shift_max + 1e-9, step)
    cands = []
    for fs in np.arange(0.70, 1.36, 0.03):
        m = m0.copy(cx=0.0, cy=0.0, f=f0 * fs, psi=0.0, tau_x=0.0, tau_y=0.0)
        u, v = m.project(b.alt, b.az)
        ok = np.isfinite(u)
        u, v = u[ok], v[ok]
        U = u[None, :] * c - v[None, :] * s           # (P, N) rotated, relative to the centre
        V = u[None, :] * s + v[None, :] * c
        best = np.zeros((psis.size, shifts.size, shifts.size), np.int16)
        for a, dx in enumerate(shifts):
            X = U + cx0 + dx
            for bb, dy in enumerate(shifts):
                best[:, a, bb] = dm.hits(X, V + cy0 + dy, tol).sum(axis=1)
        k = np.unravel_index(np.argsort(best, axis=None)[::-1][:top], best.shape)
        for p_, a, bb in zip(*k):
            cands.append((int(best[p_, a, bb]), float(np.degrees(psis[p_])), float(shifts[a]), float(shifts[bb]), float(fs)))
    cands.sort(key=lambda t: -t[0])
    out, seen = [], []
    for n, psi, dx, dy, fs in cands:
        if any(abs(((psi - p + 180) % 360) - 180) < 4 and abs(fs - f) < 0.07 for p, f in seen):
            continue
        seen.append((psi, fs))
        out.append((n, m0.copy(cx=cx0 + dx, cy=cy0 + dy, f=f0 * fs, psi=psi, tau_x=0.0, tau_y=0.0)))
        if len(out) >= top:
            break
    return out


def quick_pose_fit(m: CameraModel, fr: Frame, site: Site, cat) -> Tuple[int, CameraModel]:
    """Associate bright stars generously, fit the pose (radial fixed), and score with a common 0.7 deg criterion."""
    S = m.f / config.REF_F
    pairs = associate(m, sky_stars(site.lat, site.lon, fr.utc, min_alt=20.0, max_mag=3.5, catalog=cat, model=m),
                      fr.detections, radius=max(4.0, 30.0 * S), n_brightest=400)
    if len(pairs) >= 8:
        m = fit(m, pairs.alt, pairs.az, pairs.x, pairs.y, loss="soft_l1", f_scale=max(2.0, 8.0 * S), fixed=("k",))
    n = len(associate(m, sky_stars(site.lat, site.lon, fr.utc, min_alt=15.0, max_mag=4.5, catalog=cat, model=m),
                      fr.detections, radius=max(2.0, 0.0122 * m.f), n_brightest=1500))
    return n, m


def expected_stars(m: CameraModel, fr: Frame, site: Site, cat, max_mag: float = 5.5) -> int:
    s = sky_stars(site.lat, site.lon, fr.utc, min_alt=10.0, max_mag=max_mag, catalog=cat, model=m)
    return int(np.sum(s.in_frame)) if s.in_frame is not None else len(s.alt)


def gate(r: CalibrationResult, fr: Frame, site: Site, cat, margin: float) -> Tuple[bool, str]:
    """Accept when the fit is tight and either covers a fair fraction of the stars bright enough to be seen in
    this frame (limit = 90th percentile of the matched magnitudes), or the pose clearly beat every competitor."""
    d = r.residual_px[r.inliers]
    n, med = int(r.inliers.sum()), float(np.median(d))
    m_lim = float(np.percentile(r.pairs.mag[r.inliers], 90)) if n else 0.0
    n_exp = expected_stars(r.model, fr, site, cat, max_mag=m_lim)
    need = max(30, int(0.15 * n_exp))
    max_med = 2.0 if (margin >= 3.0 and n >= 100) else 1.2
    ok = med <= max_med and n >= 30 and (n >= need or margin >= 2.0)
    return ok, f"{n} pairs (need {need} of {n_exp} expected to m<={m_lim:.1f}; pose margin x{margin:.1f}), median {med:.2f} px"


def disc_hypotheses(fr: Frame) -> List[Tuple[str, Tuple[float, float, float]]]:
    h, w = fr.shape
    hyps = [("sky_disc", tuple(fr.disc)), ("circumscribed", (w / 2.0, h / 2.0, 0.45 * math.hypot(w, h))),
            ("inscribed", (w / 2.0, h / 2.0, 0.5 * min(w, h)))]
    if fr.gray is not None:
        hyps += [(f"hough{k + 1}", c) for k, c in enumerate(config.robust_disc_candidates(fr.gray, n=2))]
    uniq = []
    for name, d in hyps:
        if all(math.hypot(d[0] - u[0], d[1] - u[1]) > 0.06 * u[2] or abs(d[2] - u[2]) > 0.08 * u[2] for _, u in uniq):
            uniq.append((name, d))
    return uniq


def _tight_but_short(msg: str) -> bool:
    """True for a refinement rejected only for lack of pairs while its fit was tight (median <= 1 px)."""
    import re
    m = re.search(r"(\d+) pairs, median ([\d.]+) px", msg)
    return bool(m) and float(m.group(2)) <= 1.0


class TimeLimit(CalibrationError):
    pass


def calibrate_fast(frame: Frame, site: Site, *, verbose: bool = True, catalog=None,
                   deadline: Optional[float] = None) -> Tuple[CalibrationResult, Dict[str, Any]]:
    """Cascade calibration of one frame.

    Returns the result, expressed in the coordinates of the chosen parity (mirrored detections when
    ``log["parity"] == "mirror"``; :func:`ascal.calibrate` converts it back), and a log of the cascade:
    every hypothesis tried, the step that was accepted and the time spent.  ``deadline`` is an absolute
    ``time.perf_counter()`` value; :class:`TimeLimit` is raised when it is reached.
    """
    t_start = time.perf_counter()
    cat = catalog or load_catalog()
    log: Dict[str, Any] = {"steps": []}
    frames = {"direct": frame}
    tried = set()                                   # (parity, f bin, psi bin) of poses already refined
    state = {"few_pairs": False, "margins": []}     # a clear pose failed only for lack of pairs -> redetect
    parities = ("direct", "mirror") if config.get("parity", "auto") == "auto" else (config.get("parity"),)

    def check_time(where: str) -> None:
        if deadline is not None and time.perf_counter() > deadline:
            log["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            raise TimeLimit(f"time limit reached ({where}); raise --max-time to search further")

    def frame_for(parity):
        if parity not in frames:
            frames[parity] = mirrored(frame)
        return frames[parity]

    def attempt(disc_name, disc):
        """Pose for the allowed parities on one disc; refine the best poses until one passes the gate."""
        t0 = time.perf_counter()
        scored = []
        for parity in parities:
            fr = frame_for(parity)
            for n_coarse, m in raster_pose(fr, site, cat, disc, top=2):
                n, mq = quick_pose_fit(m, fr, site, cat)
                scored.append((n, parity, mq, n_coarse))
        scored.sort(key=lambda t: -t[0])
        if not scored:
            return None
        top_n = scored[0][0]
        rival = [t[0] for t in scored[1:] if t[1] != scored[0][1] or abs(((t[2].psi - scored[0][2].psi + 180) % 360) - 180) > 4
                 or abs(t[2].f - scored[0][2].f) > 0.05 * scored[0][2].f]
        state["margins"].append(top_n / max(max(rival) if rival else 1, 1))
        step = {"disc": disc_name, "detection": state["level"], "pose_s": round(time.perf_counter() - t0, 2),
                "poses": [(p, n, round(m.f), round(m.psi, 1)) for n, p, m, _ in scored[:4]]}
        log["steps"].append(step)
        _log(verbose, f"  [{disc_name}] poses: " + ", ".join(f"{p} n={n} f={m.f:.0f} psi={m.psi:.0f}" for n, p, m, _ in scored[:4]))
        for n, parity, m, _ in scored[:2]:
            check_time("refinement")
            if n < 12:
                continue
            key = (parity, round(m.f / (0.03 * m.f)), round(m.psi / 3.0))
            if key in tried:
                continue
            tried.add(key)
            fr = frame_for(parity)
            rivals = [t[0] for t in scored if t[2] is not m and (t[1] != parity or abs(((t[2].psi - m.psi + 180) % 360) - 180) > 4
                                                                   or abs(t[2].f - m.f) > 0.05 * m.f)]
            margin = n / max(max(rivals) if rivals else 1, 1)
            # radial prior: equisolid-like first; a clear pose that fails is retried with the other priors
            # (a stereographic-like lens does not converge from the first one)
            priors = RADIAL_PRIORS if margin >= 1.5 else RADIAL_PRIORS[:1]
            r, ok, why = None, False, ""
            for kp in priors:
                try:
                    r_k = refine(m.copy(k=np.array(kp)), [fr], site, cat, verbose=False)
                except CalibrationError as e:
                    step.setdefault("rejected", []).append(f"{parity} n={n} prior {kp}: {str(e)[:80]}")
                    if margin >= 2.0 and _tight_but_short(str(e)):
                        state["few_pairs"] = True
                    continue
                ok_k, why_k = gate(r_k, fr, site, cat, margin)
                step.setdefault("refined", []).append(f"{parity} n={n} prior {kp}: {why_k} -> {'ACCEPT' if ok_k else 'reject'}")
                _log(verbose, f"    refine {parity} prior {kp}: {why_k} -> {'ACCEPT' if ok_k else 'reject'}")
                r, ok, why = r_k, ok_k, why_k
                if ok:
                    break
                check_time("radial priors")
            if r is None:
                continue
            if ok:
                step["accepted"] = True
                log["accepted"] = {"disc": disc_name, "detection": state["level"], "parity": parity, "margin": round(margin, 2),
                                   "pose_score": int(n), "gate": why, "first_prior": kp}
                return r, parity, m
            if margin >= 2.0 and r.inliers.sum() < 30 and np.median(r.residual_px[r.inliers]) <= 1.0:
                state["few_pairs"] = True
        return None

    found = None
    f_orig = float(frame.info.get("fwhm", 4.0))
    fixed_disc = config.get("disc")
    hyps = (lambda: [("given", tuple(float(v) for v in fixed_disc))]) if fixed_disc else (lambda: disc_hypotheses(frame))
    for k_try in range(3):
        state["level"], state["margins"] = k_try, []
        for k_disc, (name, disc) in enumerate(hyps()):
            check_time(f"disc hypothesis {name}")
            found = attempt(name, disc)
            if found or state["few_pairs"]:
                break
            # no clear pose on the two most informative discs: the detections are the problem, not the disc
            if k_disc == 1 and max(state["margins"]) < 1.5 and frame.gray is not None and k_try < 2:
                _log(verbose, "  no clear pose on the first two discs: skip to a new detection")
                log.setdefault("early_skips", 0)
                log["early_skips"] += 1
                break
        if found or frame.gray is None or k_try == 2:
            break
        check_time("before a new detection")
        # wider detection kernel (1.5x, then 2x the automatic one)
        f_new = round((1.5, 2.0)[k_try] * f_orig, 2)
        _log(verbose, f"  no hypothesis passed; detecting again with kernel {f_new} px")
        frame.redetect(f_new)
        frames = {"direct": frame}
        tried.clear()
        state["few_pairs"] = False
        log.setdefault("redetect", []).append(f_new)
    if not found:
        log["elapsed_s"] = round(time.perf_counter() - t_start, 1)
        raise CalibrationError("no hypothesis passed the quality gate: too few stars (clouds, Moon, dew), "
                               "not a fisheye all-sky image, or wrong site/time")
    r, parity, pose = found
    # the accepted pose: try the other radial priors (cheap) and keep the best
    t0 = time.perf_counter()
    best = (r.inliers.sum() / max(np.median(r.residual_px[r.inliers]), 0.1) ** 0.5, r, RADIAL_PRIORS[0])
    for kp in RADIAL_PRIORS[1:]:
        if deadline is not None and time.perf_counter() > deadline:
            break
        try:
            r2 = refine(pose.copy(k=np.array(kp)), [frame_for(parity)], site, cat, verbose=False)
        except CalibrationError:
            continue
        sc = r2.inliers.sum() / max(np.median(r2.residual_px[r2.inliers]), 0.1) ** 0.5
        if sc > best[0]:
            best = (sc, r2, kp)
    _, r, kp = best
    log.update(parity=parity, radial_prior=kp, priors_s=round(time.perf_counter() - t0, 2),
               presmooth_sigma=frame.info.get("presmooth_sigma"), elapsed_s=round(time.perf_counter() - t_start, 1))
    return r, log
