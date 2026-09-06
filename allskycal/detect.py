"""Image loading, sky-disc detection and star detection (DAOStarFinder)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

FILENAME_TIME = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_T](\d{2})[-_](\d{2})[-_](\d{2})")


@dataclass
class Detections:
    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    peak: np.ndarray
    background: float = 0.0
    noise: float = 0.0

    def __len__(self) -> int:
        return int(self.x.size)

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    @property
    def order(self) -> np.ndarray:
        """Indices sorted by decreasing flux."""
        return np.argsort(-self.flux)

    @property
    def instrumental_mag(self) -> np.ndarray:
        return -2.5 * np.log10(np.clip(self.flux, 1e-3, None))


def read_image(path: Path | str) -> np.ndarray:
    """Load an image (any format OpenCV reads) as float32 luminance."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return to_gray(img)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]          # OpenCV is BGR
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)


def read_exif(path: Path | str) -> Dict[str, Any]:
    """Exposure time (s) and DateTimeOriginal (naive, as written) from EXIF, when present."""
    out: Dict[str, Any] = {}
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            ifd = exif.get_ifd(0x8769)
            exp = ifd.get(33434) or exif.get(33434)
            if exp:
                out["exposure_s"] = float(exp)
            dt = ifd.get(36867) or exif.get(306)
            if dt:
                out["datetime"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")
            iso = ifd.get(34855)
            if iso:
                out["iso"] = int(iso)
    except Exception:
        pass
    return out


def frame_time(path: Path | str, *, time: Optional[datetime] = None, tz: Optional[str] = None,
               exposure_s: Optional[float] = None, mid_exposure: bool = True) -> Tuple[datetime, float]:
    """Mid-exposure instant (naive UTC) of a frame.

    Priority: explicit ``time`` (aware, or naive in ``tz``/UTC) > EXIF DateTimeOriginal (in ``tz``/UTC)
    > timestamp in the file name (YYYY_MM_DD_HH_MM_SS, in ``tz``/UTC). The exposure (EXIF or argument)
    is added as half its length when ``mid_exposure`` is true and the time refers to the start.
    """
    exif = read_exif(path)
    exposure = float(exposure_s if exposure_s is not None else exif.get("exposure_s", 0.0))
    zone = ZoneInfo(tz) if tz else timezone.utc
    t = time
    if t is None and "datetime" in exif:
        t = exif["datetime"]
    if t is None:
        m = FILENAME_TIME.search(Path(path).name)
        if not m:
            raise ValueError(f"No time given, no EXIF DateTimeOriginal and no timestamp in the file name: {path}")
        t = datetime(*(int(v) for v in m.groups()))
    if t.tzinfo is None:
        t = t.replace(tzinfo=zone)
    t = t.astimezone(timezone.utc).replace(tzinfo=None)
    if mid_exposure:
        t = t + timedelta(seconds=exposure / 2.0)
    return t, exposure


def sky_disc(gray: np.ndarray) -> Tuple[float, float, float]:
    """Centre and radius (pixels) of the illuminated disc of a fisheye frame (the lens horizon).

    Threshold halfway between the centre and the corners of a blurred, downsampled copy, then the
    minimum enclosing circle of the largest connected component.
    """
    import cv2

    scale = 8
    small = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    h, w = small.shape
    centre = float(np.median(small[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20]))
    corners = float(np.median(np.concatenate([small[:15, :15].ravel(), small[:15, -15:].ravel(), small[-15:, :15].ravel(), small[-15:, -15:].ravel()])))
    thr = 0.5 * (centre + corners)
    mask = (small > thr).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray.shape[1] / 2.0, gray.shape[0] / 2.0, 0.475 * min(gray.shape)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return float(cx * scale), float(cy * scale), float(r * scale)


def disc_mask(shape: Tuple[int, int], cx: float, cy: float, radius: float, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of the pixels inside the sky disc (radius reduced by ``margin`` pixels)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius - margin) ** 2


def detect_stars(gray: np.ndarray, *, mask: Optional[np.ndarray] = None, fwhm: float = 4.0, threshold_sigma: float = 4.0,
                 box_size: int = 128, sharpness: Tuple[float, float] = (0.2, 1.0), roundness: Tuple[float, float] = (-0.7, 0.7),
                 tiles: int = 1, overlap: int = 96) -> Detections:
    """DAOStarFinder on a background-subtracted frame.

    The background is a 2-D median in boxes of ``box_size`` pixels (photutils ``Background2D``).
    Pixels outside ``mask`` (boolean or 0/255) are set to zero so nothing is detected there.
    ``tiles`` > 1 processes the frame in tiles x tiles overlapping pieces to bound memory (useful on a
    Raspberry Pi with a 12-Mpx frame).
    """
    if tiles > 1:
        return _detect_tiled(gray, mask=mask, fwhm=fwhm, threshold_sigma=threshold_sigma, box_size=box_size,
                             sharpness=sharpness, roundness=roundness, tiles=tiles, overlap=overlap)
    from photutils.background import Background2D, MedianBackground
    from photutils.detection import DAOStarFinder

    data = gray.astype(np.float32, copy=True)
    inside = np.ones(data.shape, bool) if mask is None else np.asarray(mask) > 0
    bkg = Background2D(data, box_size=box_size, bkg_estimator=MedianBackground(), mask=~inside)
    data -= bkg.background
    noise = float(bkg.background_rms_median)
    background = float(np.median(bkg.background[inside])) if inside.any() else 0.0
    data[~inside] = 0.0
    kwargs = dict(fwhm=fwhm, threshold=threshold_sigma * max(noise, 1e-3))
    try:  # photutils >= 2
        finder = DAOStarFinder(sharpness_range=sharpness, roundness_range=roundness, **kwargs)
    except TypeError:  # photutils 1.x
        finder = DAOStarFinder(sharplo=sharpness[0], sharphi=sharpness[1], roundlo=roundness[0], roundhi=roundness[1], **kwargs)
    table = finder(data)
    if table is None or len(table) == 0:
        e = np.zeros(0)
        return Detections(e, e, e, e, background, noise)
    xcol = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
    return Detections(np.asarray(table[xcol], float), np.asarray(table[ycol], float), np.asarray(table["flux"], float),
                      np.asarray(table["peak"], float), background, noise)


def _detect_tiled(gray: np.ndarray, *, mask, tiles: int, overlap: int, **kwargs) -> Detections:
    h, w = gray.shape
    xs = np.linspace(0, w, tiles + 1).astype(int)
    ys = np.linspace(0, h, tiles + 1).astype(int)
    parts, bg, noise = [], [], []
    for j in range(tiles):
        for i in range(tiles):
            x0, x1 = max(0, xs[i] - overlap), min(w, xs[i + 1] + overlap)
            y0, y1 = max(0, ys[j] - overlap), min(h, ys[j + 1] + overlap)
            sub = detect_stars(gray[y0:y1, x0:x1], mask=None if mask is None else np.asarray(mask)[y0:y1, x0:x1], tiles=1, **kwargs)
            if len(sub) == 0:
                continue
            gx, gy = sub.x + x0, sub.y + y0
            core = (gx >= xs[i]) & (gx < xs[i + 1]) & (gy >= ys[j]) & (gy < ys[j + 1])
            parts.append((gx[core], gy[core], sub.flux[core], sub.peak[core]))
            bg.append(sub.background)
            noise.append(sub.noise)
    if not parts:
        e = np.zeros(0)
        return Detections(e, e, e, e)
    cols = [np.concatenate(c) for c in zip(*parts)]
    return Detections(*cols, float(np.median(bg)), float(np.median(noise)))


@dataclass
class Frame:
    """One image ready for calibration: detections, sky disc and mid-exposure instant (UTC)."""
    path: Path
    utc: datetime
    exposure_s: float
    shape: Tuple[int, int]
    detections: Detections
    disc: Tuple[float, float, float]
    gray: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]


def load_frame(path: Path | str, *, time: Optional[datetime] = None, tz: Optional[str] = None, exposure_s: Optional[float] = None,
               tiles: int = 1, keep_image: bool = False, mask: Optional[np.ndarray] = None, **detect_kwargs: Any) -> Frame:
    """Read, time-stamp and detect stars in one frame.

    Without ``mask`` the detections are restricted to the illuminated sky disc (found automatically),
    shrunk by 2% to stay clear of its edge.
    """
    import time as _time

    t0 = _time.perf_counter()
    gray = read_image(path)
    utc, exposure = frame_time(path, time=time, tz=tz, exposure_s=exposure_s)
    disc = sky_disc(gray)
    if mask is None:
        mask = disc_mask(gray.shape, *disc, margin=0.02 * disc[2])
    det = detect_stars(gray, mask=mask, tiles=tiles, **detect_kwargs)
    info = {"n_detections": len(det), "detect_s": round(_time.perf_counter() - t0, 2), "exif": {k: v for k, v in read_exif(path).items() if k != "datetime"}}
    return Frame(Path(path), utc, exposure, gray.shape, det, disc, gray if keep_image else None, info)
