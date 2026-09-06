"""Association of catalogue stars with detections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .catalog import SkyStars
from .detect import Detections
from .model import CameraModel


def unique_mutual(pred_xy: np.ndarray, det_xy: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pairs (i_pred, j_det) such that detection j is the only one within ``radius`` of prediction i
    and prediction i is the nearest prediction to detection j."""
    from scipy.spatial import cKDTree

    if pred_xy.shape[0] == 0 or det_xy.shape[0] == 0:
        return np.zeros(0, int), np.zeros(0, int)
    tree = cKDTree(det_xy)
    d, j = tree.query(pred_xy, k=2 if det_xy.shape[0] > 1 else 1)
    if d.ndim == 1:
        d, j = d[:, None], j[:, None]
    ok = d[:, 0] <= radius
    if d.shape[1] > 1:
        ok &= d[:, 1] > radius
    back_tree = cKDTree(pred_xy)
    _, back = back_tree.query(det_xy[j[:, 0]], k=1)
    ok &= back == np.arange(pred_xy.shape[0])
    return np.flatnonzero(ok), j[ok, 0]


@dataclass
class ZeroPoint:
    """Photometric zero point of a frame: m_inst = a + m + k (X - 1)."""
    a: float
    k: float
    sigma: float
    n: int

    def predict(self, mag: np.ndarray, airmass: np.ndarray) -> np.ndarray:
        return self.a + mag + self.k * (airmass - 1.0)


def zero_point(m_inst: np.ndarray, mag: np.ndarray, airmass: np.ndarray, clip: float = 2.5, rounds: int = 4) -> ZeroPoint:
    keep = np.isfinite(m_inst)
    a = k = 0.0
    sigma = 1.0
    for _ in range(rounds):
        if keep.sum() < 3:
            break
        A = np.column_stack([np.ones(keep.sum()), airmass[keep] - 1.0])
        (a, k), *_ = np.linalg.lstsq(A, (m_inst - mag)[keep], rcond=None)
        res = m_inst - mag - a - k * (airmass - 1.0)
        sigma = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
        keep = np.isfinite(m_inst) & (np.abs(res) < clip * sigma)
    return ZeroPoint(float(a), float(k), float(sigma), int(keep.sum()))


@dataclass
class Pairs:
    """Star-detection associations of one or several frames."""
    alt: np.ndarray
    az: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mag: np.ndarray
    hip: np.ndarray
    flux: np.ndarray
    frame: np.ndarray          # frame index of each pair

    def __len__(self) -> int:
        return int(self.alt.size)

    def subset(self, s: np.ndarray) -> "Pairs":
        return Pairs(*(getattr(self, f)[s] for f in ("alt", "az", "x", "y", "mag", "hip", "flux", "frame")))

    @staticmethod
    def concatenate(parts: list) -> "Pairs":
        parts = [p for p in parts if len(p)]
        if not parts:
            e = np.zeros(0)
            return Pairs(e, e, e, e, e, e.astype(int), e, e.astype(int))
        return Pairs(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("alt", "az", "x", "y", "mag", "hip", "flux", "frame")))

    def residuals(self, model: CameraModel) -> np.ndarray:
        from .model import residuals
        return np.linalg.norm(residuals(model, self.alt, self.az, self.x, self.y), axis=1)


def associate(model: CameraModel, stars: SkyStars, det: Detections, *, radius: float, n_brightest: Optional[int] = None,
              zp: Optional[ZeroPoint] = None, mag_tolerance: float = 0.8, low_alt_tolerance: float = 1.2,
              low_alt_deg: float = 15.0, frame_index: int = 0) -> Pairs:
    """Associate stars projected with ``model`` to detections.

    A pair is accepted when the detection is the only one within ``radius`` pixels of the predicted
    position and the star is the nearest predicted star to that detection.  With a zero point, the
    instrumental magnitude of the detection must also agree with the catalogue magnitude to within
    ``mag_tolerance`` (``low_alt_tolerance`` below ``low_alt_deg`` of altitude).
    """
    x, y = model.project(stars.alt, stars.az)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < model.width) & (y >= 0) & (y < model.height)
    if stars.in_frame is not None:
        ok &= stars.in_frame
    idx = np.flatnonzero(ok)
    sel = det.order[:n_brightest] if n_brightest else np.arange(len(det))
    i, j = unique_mutual(np.column_stack([x[idx], y[idx]]), det.xy[sel], radius)
    si, dj = idx[i], sel[j]
    if zp is not None and si.size:
        dm = det.instrumental_mag[dj] - zp.predict(stars.mag[si], stars.airmass[si])
        tol = np.where(stars.alt[si] < low_alt_deg, low_alt_tolerance, mag_tolerance)
        good = np.abs(dm) <= tol
        si, dj = si[good], dj[good]
    return Pairs(stars.alt[si], stars.az[si], det.x[dj], det.y[dj], stars.mag[si], stars.hip[si], det.flux[dj], np.full(si.size, frame_index))


def frame_zero_point(model: CameraModel, stars: SkyStars, det: Detections, *, min_alt: float = 30.0, max_mag: float = 4.5,
                     radius: float = 12.0) -> Optional[ZeroPoint]:
    """Zero point from the bright, high stars that have a detection within ``radius`` of the prediction."""
    from scipy.spatial import cKDTree

    x, y = model.project(stars.alt, stars.az)
    s = np.isfinite(x) & np.isfinite(y) & (stars.alt >= min_alt) & (stars.mag <= max_mag)
    if stars.in_frame is not None:
        s &= stars.in_frame
    if s.sum() < 5 or len(det) == 0:
        return None
    d, j = cKDTree(det.xy).query(np.column_stack([x[s], y[s]]))
    ok = d <= radius
    if ok.sum() < 5:
        return None
    return zero_point(det.instrumental_mag[j[ok]], stars.mag[s][ok], stars.airmass[s][ok])
