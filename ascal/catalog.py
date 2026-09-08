"""Reference star positions: Hipparcos catalogue, precession, sidereal time, horizontal coordinates.

Pure numpy; no external ephemeris library.  Accuracy is a few arcseconds, far
below the pixel scale of any all-sky camera (nutation is included only through
the equation of the equinoxes; aberration is neglected).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

CATALOG_PATH = Path(__file__).resolve().parent / "data" / "hipparcos_mag65.json"
J2000_JD = 2451545.0


@dataclass
class Catalog:
    ra: np.ndarray        # degrees, J2000
    dec: np.ndarray       # degrees, J2000
    mag: np.ndarray       # Hipparcos V-like magnitude
    hip: np.ndarray
    names: List[str]

    def __len__(self) -> int:
        return int(self.ra.size)

    def brighter_than(self, max_mag: float) -> "Catalog":
        s = self.mag <= max_mag
        return Catalog(self.ra[s], self.dec[s], self.mag[s], self.hip[s], [n for n, k in zip(self.names, s) if k])


_CACHE: dict = {}


def load_catalog(path: Path | str = CATALOG_PATH, max_mag: float = 6.5) -> Catalog:
    """Hipparcos stars brighter than ``max_mag`` (8789 stars down to 6.5 in the bundled file)."""
    key = (str(path), max_mag)
    if key not in _CACHE:
        rows = [s for s in json.loads(Path(path).read_text()) if s.get("mag") is not None and s["mag"] <= max_mag]
        _CACHE[key] = Catalog(
            ra=np.array([s["ra"] for s in rows], float), dec=np.array([s["dec"] for s in rows], float),
            mag=np.array([s["mag"] for s in rows], float), hip=np.array([s.get("hip") or 0 for s in rows], int),
            names=[s.get("name") or f"HIP {s.get('hip')}" for s in rows])
    return _CACHE[key]


# ---------------------------------------------------------------------- time

def to_utc(t: datetime) -> datetime:
    """Naive datetimes are taken as UTC; aware ones are converted."""
    if t.tzinfo is None:
        return t
    return t.astimezone(timezone.utc).replace(tzinfo=None)


def julian_date(t: datetime) -> float:
    t = to_utc(t)
    y, m = t.year, t.month
    d = t.day + (t.hour + (t.minute + (t.second + t.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def equation_of_equinoxes_deg(jd: float) -> float:
    """Apparent minus mean sidereal time (degrees), from the two main nutation terms."""
    T = (jd - J2000_JD) / 36525.0
    omega = math.radians(125.04452 - 1934.136261 * T)             # longitude of the ascending node of the Moon
    L = math.radians(280.4665 + 36000.7698 * T)                   # mean longitude of the Sun
    Lp = math.radians(218.3165 + 481267.8813 * T)                 # mean longitude of the Moon
    dpsi_arcsec = -17.20 * math.sin(omega) - 1.32 * math.sin(2 * L) - 0.23 * math.sin(2 * Lp) + 0.21 * math.sin(2 * omega)
    eps = math.radians(23.439291 - 0.0130042 * T) + math.radians((9.20 * math.cos(omega) + 0.57 * math.cos(2 * L)) / 3600.0)
    return dpsi_arcsec * math.cos(eps) / 3600.0


def sidereal_time_deg(lon_deg: float, t: datetime, apparent: bool = True) -> float:
    """Local (apparent) sidereal time in degrees for east longitude ``lon_deg``."""
    jd = julian_date(t)
    T = (jd - J2000_JD) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - J2000_JD) + 0.000387933 * T * T - T ** 3 / 38710000.0
    if apparent:
        gmst += equation_of_equinoxes_deg(jd)
    return (gmst + lon_deg) % 360.0


# ---------------------------------------------------------------------- coordinates

def precess_j2000(ra_deg: np.ndarray, dec_deg: np.ndarray, t: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """IAU 1976 precession (Lieske et al. 1977) from J2000 to the date, vectorised."""
    T = (julian_date(t) - J2000_JD) / 36525.0
    arcsec = math.pi / 648000.0
    zeta = (2306.2181 * T + 0.30188 * T ** 2 + 0.017998 * T ** 3) * arcsec
    z = (2306.2181 * T + 1.09468 * T ** 2 + 0.018203 * T ** 3) * arcsec
    theta = (2004.3109 * T - 0.42665 * T ** 2 - 0.041833 * T ** 3) * arcsec
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    v = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    cz, sz, cth, sth, czz, szz = math.cos(zeta), math.sin(zeta), math.cos(theta), math.sin(theta), math.cos(z), math.sin(z)
    r3a = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    r2 = np.array([[cth, 0.0, -sth], [0.0, 1.0, 0.0], [sth, 0.0, cth]])
    r3b = np.array([[czz, -szz, 0.0], [szz, czz, 0.0], [0.0, 0.0, 1.0]])
    w = r3b @ (r2 @ (r3a @ v))
    return np.degrees(np.arctan2(w[1], w[0])) % 360.0, np.degrees(np.arcsin(np.clip(w[2], -1.0, 1.0)))


def radec_to_altaz(ra_deg: np.ndarray, dec_deg: np.ndarray, lat_deg: float, lst_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    ha = np.radians((lst_deg - ra_deg) % 360.0)
    dec, lat = np.radians(dec_deg), math.radians(lat_deg)
    sin_alt = np.sin(dec) * math.sin(lat) + np.cos(dec) * math.cos(lat) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    cos_alt = np.maximum(np.cos(alt), 1e-9)
    sin_az = -np.cos(dec) * np.sin(ha) / cos_alt
    cos_az = (np.sin(dec) - math.sin(lat) * sin_alt) / (math.cos(lat) * cos_alt)
    return np.degrees(alt), np.degrees(np.arctan2(sin_az, cos_az)) % 360.0


def airmass(alt_deg: np.ndarray) -> np.ndarray:
    """Kasten & Young (1989)."""
    a = np.clip(np.asarray(alt_deg, float), 0.1, 90.0)
    return 1.0 / (np.sin(np.radians(a)) + 0.50572 * (a + 6.07995) ** -1.6364)


@dataclass
class SkyStars:
    """Catalogue stars above the horizon at one instant, optionally projected with a model."""
    names: List[str]
    hip: np.ndarray
    mag: np.ndarray
    alt: np.ndarray
    az: np.ndarray
    airmass: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    in_frame: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.mag.size)

    def subset(self, s: np.ndarray) -> "SkyStars":
        return SkyStars([n for n, k in zip(self.names, s) if k], self.hip[s], self.mag[s], self.alt[s], self.az[s], self.airmass[s],
                        None if self.x is None else self.x[s], None if self.y is None else self.y[s],
                        None if self.in_frame is None else self.in_frame[s])


def sky_stars(lat: float, lon: float, t: datetime, *, min_alt: float = 0.0, max_mag: float = 6.5,
              catalog: Optional[Catalog] = None, model: Any = None, mask: Optional[np.ndarray] = None) -> SkyStars:
    """Stars above ``min_alt`` at instant ``t`` (UTC) seen from (lat, lon east, degrees).

    If ``model`` is given the stars are projected and ``in_frame`` marks those inside the sensor
    (and inside ``mask`` if provided, a boolean or 0/255 array of the sensor shape).
    """
    cat = (catalog or load_catalog()).brighter_than(max_mag)
    lst = sidereal_time_deg(lon, t)
    ra, dec = precess_j2000(cat.ra, cat.dec, t)
    alt, az = radec_to_altaz(ra, dec, lat, lst)
    keep = alt >= min_alt
    out = SkyStars([n for n, k in zip(cat.names, keep) if k], cat.hip[keep], cat.mag[keep], alt[keep], az[keep], airmass(alt[keep]))
    if model is not None:
        x, y = model.project(out.alt, out.az)
        w, h = model.width, model.height
        ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if mask is not None:
            xi = np.clip(np.nan_to_num(x).astype(int), 0, w - 1)
            yi = np.clip(np.nan_to_num(y).astype(int), 0, h - 1)
            ok &= np.asarray(mask)[yi, xi] > 0
        out.x, out.y, out.in_frame = x, y, ok
    return out
