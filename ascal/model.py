"""Camera model for all-sky (fisheye) cameras.

Projection chain (sky -> sensor), see docs/model.md:

1. rigid rotation of the optical axis (tilt): ``s_c = R(tau_x, tau_y) s``
2. radial function of Kannala & Brandt (2006): ``r = f (theta + k3 theta^3 + k5 theta^5 + ...)``
3. image rotation ``psi`` with respect to north: ``u0 = r sin(psi - az_c)``, ``v0 = -r cos(psi - az_c)``
4. optional Brown-Conrady decentering ``(p1, p2)``; ``x = cx + u``, ``y = cy + v``

Conventions: altitude/azimuth in degrees, azimuth from north through east; pixel
``x`` to the right and ``y`` downward with the origin at the top-left corner and
pixel centres at integer coordinates.  ``f`` is the focal length in pixels per
radian (plate scale on the optical axis).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ArrayLike = Any


def altaz_to_vec(alt_deg: ArrayLike, az_deg: ArrayLike) -> np.ndarray:
    """Unit vectors (east, north, up) of directions given in degrees."""
    a, z = np.radians(alt_deg), np.radians(az_deg)
    return np.stack([np.cos(a) * np.sin(z), np.cos(a) * np.cos(z), np.sin(a)], axis=-1)


def vec_to_altaz(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    alt = np.degrees(np.arcsin(np.clip(v[..., 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360.0
    return alt, az


def rotation_matrix(tau_x_deg: float, tau_y_deg: float) -> np.ndarray:
    """R = R_y(tau_y) R_x(tau_x): the rotation that takes the topocentric frame to the camera frame."""
    ax, ay = math.radians(tau_x_deg), math.radians(tau_y_deg)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(ax), math.sin(ax)], [0.0, -math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0.0, math.sin(ay)], [0.0, 1.0, 0.0], [-math.sin(ay), 0.0, math.cos(ay)]])
    return ry @ rx


@dataclass
class CameraModel:
    """All-sky camera model (model A of the paper; model B when ``p`` is set)."""

    cx: float
    cy: float
    f: float                                   # pixels per radian
    psi: float = 0.0                           # image rotation with respect to north (degrees)
    tau_x: float = 0.0                         # tilt about the east-west axis (degrees)
    tau_y: float = 0.0                         # tilt about the north-south axis (degrees)
    k: np.ndarray = field(default_factory=lambda: np.array([-0.03, 0.0]))   # k3, k5, ... (odd powers)
    p: Optional[np.ndarray] = None             # Brown-Conrady (p1, p2) in units of f, or None
    width: int = 0
    height: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    mirror: bool = False                       # image stored mirrored (x -> width - 1 - x), e.g. FITS rows as stored

    # ------------------------------------------------------------------ basics
    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        if self.p is not None:
            self.p = np.asarray(self.p, dtype=float)

    @property
    def decentering(self) -> bool:
        return self.p is not None

    @property
    def n_params(self) -> int:
        return 6 + self.k.size + (2 if self.decentering else 0)

    def copy(self, **changes: Any) -> "CameraModel":
        m = replace(self, **changes)
        m.k = np.array(self.k if "k" not in changes else changes["k"], dtype=float)
        if m.p is not None:
            m.p = np.array(m.p, dtype=float)
        m.meta = dict(self.meta if "meta" not in changes else changes["meta"])
        return m

    # ------------------------------------------------------------------ radial
    def radius(self, theta_rad: ArrayLike) -> np.ndarray:
        """r(theta) in pixels for zenith distance theta (radians) in the camera frame."""
        th = np.asarray(theta_rad, dtype=float)
        out = th.copy()
        for i, ki in enumerate(self.k, start=1):
            out = out + ki * th ** (2 * i + 1)
        return self.f * out

    def dradius(self, theta_rad: ArrayLike) -> np.ndarray:
        """dr/dtheta in pixels per radian (local plate scale)."""
        th = np.asarray(theta_rad, dtype=float)
        out = np.ones_like(th)
        for i, ki in enumerate(self.k, start=1):
            out = out + (2 * i + 1) * ki * th ** (2 * i)
        return self.f * out

    def theta(self, r_px: ArrayLike, iterations: int = 30) -> np.ndarray:
        """Zenith distance (radians) for a radius in pixels; Newton on the odd polynomial."""
        r = np.asarray(r_px, dtype=float)
        th = np.clip(r / self.f, 0.0, math.pi)
        for _ in range(iterations):
            g = self.radius(th) - r
            dg = self.dradius(th)
            th = np.clip(th - g / np.where(np.abs(dg) > 1e-9, dg, 1e-9), 0.0, math.pi)
        return th

    # ------------------------------------------------------------------ tilt
    def sky_to_camera(self, alt: ArrayLike, az: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        v = altaz_to_vec(alt, az) @ rotation_matrix(self.tau_x, self.tau_y).T
        return vec_to_altaz(v)

    def camera_to_sky(self, alt_c: ArrayLike, az_c: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        v = altaz_to_vec(alt_c, az_c) @ rotation_matrix(self.tau_x, self.tau_y)
        return vec_to_altaz(v)

    # ------------------------------------------------------------------ decentering
    def _decentering(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p1, p2 = self.p
        un, vn = u / self.f, v / self.f
        r2 = un * un + vn * vn
        du = (p1 * (r2 + 2 * un * un) + 2 * p2 * un * vn) * self.f
        dv = (2 * p1 * un * vn + p2 * (r2 + 2 * vn * vn)) * self.f
        return du, dv

    # ------------------------------------------------------------------ projections
    def project(self, alt: ArrayLike, az: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Sky (alt, az in degrees) -> pixel (x, y). Directions below the camera horizon give NaN."""
        alt_c, az_c = self.sky_to_camera(np.asarray(alt, float), np.asarray(az, float))
        th = np.radians(90.0 - alt_c)
        r = self.radius(th)
        r = np.where(th <= math.pi / 2 + 0.2, r, np.nan)
        ang = np.radians(self.psi - az_c)
        u, v = r * np.sin(ang), -r * np.cos(ang)
        if self.decentering:
            du, dv = self._decentering(u, v)
            u, v = u + du, v + dv
        x = self.cx + u
        if self.mirror:
            x = (self.width - 1) - x
        return x, self.cy + v

    def unproject(self, x: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel (x, y) -> sky (alt, az in degrees)."""
        x = np.asarray(x, float)
        if self.mirror:
            x = (self.width - 1) - x
        u = x - self.cx
        v = np.asarray(y, float) - self.cy
        if self.decentering:
            u0, v0 = u.copy(), v.copy()
            for _ in range(10):
                du, dv = self._decentering(u, v)
                u, v = u0 - du, v0 - dv
        r = np.hypot(u, v)
        alt_c = 90.0 - np.degrees(self.theta(r))
        az_c = (self.psi - np.degrees(np.arctan2(u, -v))) % 360.0
        return self.camera_to_sky(alt_c, az_c)

    # ------------------------------------------------------------------ derived quantities
    @property
    def total_tilt(self) -> float:
        """Angle between the optical axis and the zenith (degrees)."""
        return math.degrees(math.acos(math.cos(math.radians(self.tau_x)) * math.cos(math.radians(self.tau_y))))

    @property
    def zenith_pixel(self) -> Tuple[float, float]:
        x, y = self.project(np.array([90.0]), np.array([0.0]))
        return float(x[0]), float(y[0])

    @property
    def horizon_radius(self) -> float:
        return float(self.radius(np.array([math.pi / 2]))[0])

    def plate_scale(self, theta_deg: ArrayLike = 0.0) -> np.ndarray:
        """Pixels per degree at zenith distance theta (degrees, camera frame)."""
        return self.dradius(np.radians(theta_deg)) * math.pi / 180.0

    def solid_angle(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Solid angle seen by a pixel (steradian per pixel^2) at (x, y)."""
        u = np.asarray(x, float) - self.cx
        v = np.asarray(y, float) - self.cy
        r = np.maximum(np.hypot(u, v), 1e-6)
        th = self.theta(r)
        return np.sin(th) / (r * self.dradius(th))

    # ------------------------------------------------------------------ parameter vector
    def to_vector(self) -> np.ndarray:
        vec = [self.cx, self.cy, self.f, self.psi, self.tau_x, self.tau_y, *self.k]
        if self.decentering:
            vec += list(self.p)
        return np.asarray(vec, dtype=float)

    def with_vector(self, vec: np.ndarray) -> "CameraModel":
        n = self.k.size
        m = self.copy(cx=float(vec[0]), cy=float(vec[1]), f=float(vec[2]), psi=float(vec[3]) % 360.0,
                      tau_x=float(vec[4]), tau_y=float(vec[5]), k=np.asarray(vec[6:6 + n], float))
        if self.decentering:
            m.p = np.asarray(vec[6 + n:8 + n], float)
        return m

    def scales(self) -> np.ndarray:
        """Characteristic scale of each parameter (for the least-squares solver)."""
        s = np.ones(self.n_params)
        s[:3] = 10.0          # centre and focal length, pixels
        s[3:6] = 0.1          # angles, degrees
        s[6:6 + self.k.size] = 1.0
        if self.decentering:
            s[6 + self.k.size:] = 1e-3
        return s

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": "kannala-brandt + rigid rotation" + (" + brown-conrady" if self.decentering else ""),
            "cx": float(self.cx), "cy": float(self.cy), "f": float(self.f), "psi": float(self.psi),
            "tau_x": float(self.tau_x), "tau_y": float(self.tau_y), "k": [float(v) for v in self.k],
            "p": None if self.p is None else [float(v) for v in self.p],
            "width": int(self.width), "height": int(self.height), "mirror": bool(self.mirror),
            "derived": {
                "focal_length_px_per_deg": float(self.f * math.pi / 180.0),
                "total_tilt_deg": self.total_tilt,
                "zenith_pixel": list(self.zenith_pixel),
                "horizon_radius_px": self.horizon_radius,
                "plate_scale_axis_arcmin_per_px": 60.0 / float(self.plate_scale(0.0)),
                "plate_scale_horizon_arcmin_per_px": 60.0 / float(self.plate_scale(90.0)),
            },
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraModel":
        return cls(cx=d["cx"], cy=d["cy"], f=d["f"], psi=d.get("psi", 0.0), tau_x=d.get("tau_x", 0.0), tau_y=d.get("tau_y", 0.0),
                   k=np.asarray(d.get("k", [-0.03, 0.0]), float), p=None if d.get("p") is None else np.asarray(d["p"], float),
                   width=int(d.get("width", 0)), height=int(d.get("height", 0)), meta=dict(d.get("meta", {})),
                   mirror=bool(d.get("mirror", False)))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "CameraModel":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def initial(cls, width: int, height: int, cx: Optional[float] = None, cy: Optional[float] = None,
                horizon_radius: Optional[float] = None, psi: float = 0.0) -> "CameraModel":
        """Generic starting point: equidistant lens whose horizon is the illuminated disc."""
        cx = width / 2 if cx is None else cx
        cy = height / 2 if cy is None else cy
        rh = 0.475 * min(width, height) if horizon_radius is None else horizon_radius
        return cls(cx=cx, cy=cy, f=2 * rh / math.pi, psi=psi, k=np.array([-0.03, 0.0]), width=width, height=height)

    def __repr__(self) -> str:  # pragma: no cover
        p = "" if self.p is None else f", p=({self.p[0]:.2e}, {self.p[1]:.2e})"
        return (f"CameraModel(cx={self.cx:.2f}, cy={self.cy:.2f}, f={self.f:.2f} px/rad, psi={self.psi:.4f}°, "
                f"tilt=({self.tau_x:.4f}, {self.tau_y:.4f})°, k={np.round(self.k, 5).tolist()}{p})")


# ---------------------------------------------------------------------- fitting

def residuals(model: CameraModel, alt: ArrayLike, az: ArrayLike, x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Predicted minus detected pixel position, shape (N, 2). NaN predictions count as 50 px."""
    px, py = model.project(alt, az)
    r = np.column_stack([px - np.asarray(x, float), py - np.asarray(y, float)])
    return np.nan_to_num(r, nan=50.0)


def fit(model: CameraModel, alt: ArrayLike, az: ArrayLike, x: ArrayLike, y: ArrayLike, *,
        loss: str = "soft_l1", f_scale: float = 2.0, fixed: Sequence[str] = (), max_nfev: int = 400) -> CameraModel:
    """Least-squares fit in pixel space (scipy trust-region reflective).

    ``loss`` is ``"linear"`` (plain least squares) or ``"soft_l1"`` (robust; ``f_scale`` in pixels).
    ``fixed`` names parameters kept at their current value: any of cx, cy, f, psi, tau_x, tau_y, k, p.
    """
    from scipy.optimize import least_squares

    alt, az, x, y = (np.asarray(v, float) for v in (alt, az, x, y))
    p0 = model.to_vector()
    names = ["cx", "cy", "f", "psi", "tau_x", "tau_y"] + ["k"] * model.k.size + (["p", "p"] if model.decentering else [])
    free = np.array([n not in fixed for n in names])

    def full(q: np.ndarray) -> np.ndarray:
        vec = p0.copy()
        vec[free] = q
        return vec

    def fun(q: np.ndarray) -> np.ndarray:
        return residuals(model.with_vector(full(q)), alt, az, x, y).ravel()

    sol = least_squares(fun, p0[free], loss=loss, f_scale=f_scale, x_scale=model.scales()[free], max_nfev=max_nfev)
    out = model.with_vector(full(sol.x))
    out.meta.update({"fit_cost": float(sol.cost), "fit_nfev": int(sol.nfev), "fit_status": int(sol.status), "n_pairs": int(alt.size)})
    return out


def robust_fit(model: CameraModel, alt: ArrayLike, az: ArrayLike, x: ArrayLike, y: ArrayLike, *,
               bands: Sequence[float] = (3, 10, 20, 30, 50, 70, 90), n_sigma: float = 3.5, floor_px: float = 3.0,
               rounds: int = 4, fixed: Sequence[str] = ()) -> Tuple[CameraModel, np.ndarray]:
    """Soft-L1 fit, per-altitude-band clipping (median + n_sigma robust sigma, with a floor), final linear fit.

    Returns the fitted model and the boolean mask of retained pairs.
    """
    alt, az, x, y = (np.asarray(v, float) for v in (alt, az, x, y))
    keep = np.ones(alt.size, dtype=bool)
    for it in range(rounds):
        model = fit(model, alt[keep], az[keep], x[keep], y[keep], loss="soft_l1" if it == 0 else "linear", f_scale=3.0, fixed=fixed)
        d = np.linalg.norm(residuals(model, alt, az, x, y), axis=1)
        new = np.zeros_like(keep)
        for lo, hi in zip(bands[:-1], bands[1:]):
            band = (alt >= lo) & (alt < hi)
            ref = d[band & keep]
            if ref.size:
                med = np.median(ref)
                limit = max(floor_px, med + n_sigma * 1.4826 * np.median(np.abs(ref - med)))
                new[band] = d[band] < limit
        # stars outside the band range are kept if they pass the global floor
        outside = (alt < bands[0]) | (alt >= bands[-1])
        new[outside] = d[outside] < floor_px
        if np.array_equal(new, keep):
            break
        keep = new
    model = fit(model, alt[keep], az[keep], x[keep], y[keep], loss="linear", fixed=fixed)
    return model, keep


def band_statistics(alt: ArrayLike, d: ArrayLike, bands: Sequence[float] = (3, 10, 20, 30, 50, 70, 90)) -> list:
    """Median, rms, 90th percentile and fraction < 1 px of residuals per altitude band."""
    alt, d = np.asarray(alt, float), np.asarray(d, float)
    rows = []
    for lo, hi in list(zip(bands[:-1], bands[1:])) + [(bands[0], bands[-1])]:
        s = (alt >= lo) & (alt < hi)
        name = "all" if (lo, hi) == (bands[0], bands[-1]) else f"{lo:g}-{hi:g}"
        if not s.any():
            rows.append({"band": name, "n": 0})
            continue
        rows.append({"band": name, "n": int(s.sum()), "median": float(np.median(d[s])), "rms": float(np.sqrt(np.mean(d[s] ** 2))),
                     "p90": float(np.percentile(d[s], 90)), "within_1px": float(np.mean(d[s] < 1.0))})
    return rows
