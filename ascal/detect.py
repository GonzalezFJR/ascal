"""Image loading, sky-disc detection and star detection (DAOStarFinder).

Formats: anything OpenCV reads (JPEG, PNG, TIFF, 8 or 16 bit), FITS (``.fits``, ``.fit``, ``.fts``, optionally
gzipped; rows as stored, colour cubes averaged; time from ``DATE-OBS`` and exposure from ``EXPTIME``/``EXPOSURE``)
and camera raw files (``.nef``, ``.cr2``, ``.cr3``, ``.arw``, ``.dng``, ``.raf``, ``.orf``, ``.rw2``; needs the
optional ``rawpy``; the linear green plane is used).  A mirrored image (e.g. FITS with the origin at the bottom)
needs no flipping: the calibration finds the parity.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np

# Detection kernel.  ``fwhm="auto"`` measures the star FWHM and sets the DAOStarFinder kernel to
# FWHM_FACTOR times it, never below DEFAULT_FWHM (the value of the paper, right for the ZRO camera,
# whose stars measure 2.8 px) nor above FWHM_MAX.
DEFAULT_FWHM = 4.0
FWHM_FACTOR = 1.3
FWHM_MAX = 12.0

FITS_EXT = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")
RAW_EXT = (".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw")
AUTO_TILES_PIXELS = 16e6     # above this, detection runs in 3 x 3 tiles processed in parallel

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


def _kind(path: Path | str) -> str:
    name = str(path).lower()
    if name.endswith(FITS_EXT):
        return "fits"
    if name.endswith(RAW_EXT):
        return "raw"
    return "image"


def read_image(path: Path | str) -> np.ndarray:
    """Load an image as float32 luminance (FITS and camera raw files included, see module docstring)."""
    kind = _kind(path)
    if kind == "fits":
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = next(h.data for h in hdul if h.data is not None)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 3:                          # colour cube (3, H, W) or (H, W, 3)
            data = data.mean(axis=0) if data.shape[0] in (3, 4) else data.mean(axis=-1)
        return np.nan_to_num(data)
    if kind == "raw":
        try:
            import rawpy
        except ImportError as exc:  # pragma: no cover
            raise ImportError("reading camera raw files needs rawpy: pip install rawpy") from exc
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=False, user_wb=[1, 1, 1, 1],
                                  demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, output_color=rawpy.ColorSpace.raw)
        return rgb[..., 1].astype(np.float32)
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return to_gray(img)


def fits_time(path: Path | str) -> Tuple[Optional[datetime], Optional[float]]:
    """(start time as naive UTC, exposure s) from a FITS header, when present."""
    from astropy.io import fits
    try:
        with fits.open(path) as hdul:
            h = next(x.header for x in hdul if x.data is not None)
    except Exception:
        return None, None
    t = None
    for key in ("DATE-OBS", "DATE_OBS"):
        if key in h:
            v = str(h[key]).strip().replace("Z", "")
            if "T" not in v and "TIME-OBS" in h:
                v = f"{v}T{h['TIME-OBS']}"
            try:
                t = datetime.fromisoformat(v[:26])
            except ValueError:
                t = None
            break
    exp = None
    for key in ("EXPTIME", "EXPOSURE", "ONTIME"):
        if key in h:
            try:
                exp = float(h[key])
                break
            except (TypeError, ValueError):
                pass
    return t, exp


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
    exif = read_exif(path) if _kind(path) != "fits" else {}
    if _kind(path) == "fits":
        t_fits, exp_fits = fits_time(path)
        if t_fits is not None:
            exif["datetime"] = t_fits
        if exp_fits is not None:
            exif["exposure_s"] = exp_fits
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

    from . import config
    if config.get("disc"):
        return tuple(float(v) for v in config.get("disc"))
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
    from . import config
    if config.get("noise") == "mad" and inside.any():
        # Background2D's rms includes the large-scale gradient inside each box (Moon, vignetting): use the residual
        r = data[inside]
        r = r[np.isfinite(r)]
        mad = 1.4826 * float(np.median(np.abs(r - np.median(r)))) if r.size else noise
        noise = max(mad, 1e-3)
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


def estimate_fwhm(gray: np.ndarray, disc: Tuple[float, float, float], *, n_stars: int = 200,
                  half: int = 8) -> Tuple[float, int]:
    """Median FWHM (pixels) of unsaturated stars, from 2-D Gaussian fits in the centre of the sky disc.

    A first DAOStarFinder pass (6 px kernel, 5 sigma, no shape cuts) runs on the central square of the
    disc (side 1.2 R); the brightest detections whose cut-out has no pixel within 3 % of the image
    maximum are fitted with a circular Gaussian plus constant.  Returns (FWHM, number of stars fitted);
    (nan, 0) when nothing could be fitted.
    """
    from scipy.optimize import curve_fit

    cx, cy, r = disc
    s = int(0.6 * r)
    x0, y0 = max(0, int(cx) - s), max(0, int(cy) - s)
    sub = gray[y0:int(cy) + s, x0:int(cx) + s]
    mask = disc_mask(sub.shape, cx - x0, cy - y0, r, margin=0.02 * r)
    det = detect_stars(sub, mask=mask, fwhm=6.0, threshold_sigma=5.0, sharpness=(0.0, 2.0), roundness=(-2.0, 2.0),
                       box_size=min(128, max(16, min(sub.shape) // 4)))
    saturation = 0.97 * float(gray.max())
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)

    def gauss(c, a, xc, yc, sigma, b):
        return a * np.exp(-((c[0] - xc) ** 2 + (c[1] - yc) ** 2) / (2 * sigma * sigma)) + b

    fwhm = []
    for i in det.order:
        xi, yi = int(round(det.x[i])), int(round(det.y[i]))
        c = sub[yi - half:yi + half + 1, xi - half:xi + half + 1]
        if c.shape != xx.shape or c.max() >= saturation:
            continue
        b0 = float(np.median(np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])))
        try:
            p, _ = curve_fit(gauss, (xx.ravel(), yy.ravel()), c.ravel().astype(float), p0=[c.max() - b0, 0.0, 0.0, 1.5, b0], maxfev=400)
        except Exception:
            continue
        if p[0] > 0 and 0.4 < abs(p[3]) < half / 2 and math.hypot(p[1], p[2]) < 2:
            fwhm.append(2.3548 * abs(p[3]))
        if len(fwhm) >= n_stars:
            break
    return (float(np.median(fwhm)), len(fwhm)) if fwhm else (float("nan"), 0)


def detection_fwhm(measured: float) -> float:
    """DAOStarFinder kernel FWHM for stars of the measured FWHM (see DEFAULT_FWHM)."""
    if not np.isfinite(measured):
        return DEFAULT_FWHM
    return round(float(np.clip(FWHM_FACTOR * measured, DEFAULT_FWHM, FWHM_MAX)), 2)


def _detect_tile(args):
    sub, sub_mask, kwargs = args
    return detect_stars(sub, mask=sub_mask, tiles=1, **kwargs)


def _detect_tiled(gray: np.ndarray, *, mask, tiles: int, overlap: int, **kwargs) -> Detections:
    """Detection in tiles x tiles overlapping pieces, processed in parallel (``config.workers`` processes)."""
    from . import config

    h, w = gray.shape
    xs = np.linspace(0, w, tiles + 1).astype(int)
    ys = np.linspace(0, h, tiles + 1).astype(int)
    jobs, boxes = [], []
    for j in range(tiles):
        for i in range(tiles):
            x0, x1 = max(0, xs[i] - overlap), min(w, xs[i + 1] + overlap)
            y0, y1 = max(0, ys[j] - overlap), min(h, ys[j + 1] + overlap)
            sub_mask = None if mask is None else np.asarray(mask)[y0:y1, x0:x1]
            if sub_mask is not None and not np.any(sub_mask):
                continue
            jobs.append((np.ascontiguousarray(gray[y0:y1, x0:x1]), sub_mask, kwargs))
            boxes.append((x0, y0, xs[i], xs[i + 1], ys[j], ys[j + 1]))
    nw = config.n_workers(len(jobs))
    if nw > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(nw) as ex:
            results = list(ex.map(_detect_tile, jobs))
    else:
        results = [_detect_tile(job) for job in jobs]
    parts, bg, noise = [], [], []
    for sub, (x0, y0, ax, bx, ay, by) in zip(results, boxes):
        if len(sub) == 0:
            continue
        gx, gy = sub.x + x0, sub.y + y0
        core = (gx >= ax) & (gx < bx) & (gy >= ay) & (gy < by)
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
    mask: Optional[np.ndarray] = None          # custom detection mask, when one was given

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    def redetect(self, fwhm: float) -> None:
        """Run the star detection again with another kernel FWHM (needs the image: ``keep_image=True``)."""
        if self.gray is None:
            raise ValueError("redetect needs the image (load the frame with keep_image=True)")
        kw = dict(self.info.get("detect_kwargs", {}), fwhm=float(fwhm))
        mask = self.mask if self.mask is not None else disc_mask(self.gray.shape, *self.disc, margin=0.02 * self.disc[2])
        self.detections = detect_stars(self.gray, mask=mask, **kw)
        self.info.update(fwhm=float(fwhm), n_detections=len(self.detections), detect_kwargs=kw)


def load_frame(path: Path | str, *, time: Optional[datetime] = None, tz: Optional[str] = None, exposure_s: Optional[float] = None,
               tiles: Union[int, str, None] = None, keep_image: bool = False, mask: Optional[np.ndarray] = None,
               fwhm: Union[float, str] = "auto", **detect_kwargs: Any) -> Frame:
    """Read, time-stamp and detect stars in one frame.

    Without ``mask`` the detections are restricted to the illuminated sky disc (found automatically),
    shrunk by 2% to stay clear of its edge.  ``fwhm`` is the DAOStarFinder kernel in pixels, or
    ``"auto"`` to derive it from the measured star FWHM (:func:`estimate_fwhm`, :func:`detection_fwhm`).
    Other keyword arguments go to :func:`detect_stars` (``threshold_sigma``, ``roundness``...).
    ``tiles``: ``"auto"`` (default, from ``config.tiles``) uses 3 x 3 tiles in parallel above 16 Mpx.
    """
    import time as _time

    t0 = _time.perf_counter()
    from . import config
    gray = read_image(path)
    utc, exposure = frame_time(path, time=time, tz=tz, exposure_s=exposure_s)
    disc = sky_disc(gray)
    info: Dict[str, Any] = {"fwhm_mode": "auto" if fwhm == "auto" else "fixed"}
    ps = config.get("presmooth")
    if ps:
        # undersampled stars (FWHM < ~2.2 px) fail DAOStarFinder's shape cuts: smooth them to a resolvable width
        import cv2
        measured0, _ = estimate_fwhm(gray, disc)
        if ps == "auto":
            sig = math.sqrt(max(0.0, (2.5 / 2.3548) ** 2 - (measured0 / 2.3548) ** 2)) if np.isfinite(measured0) and measured0 < 2.2 else 0.0
        else:
            sig = float(ps)
        if sig > 0.3:
            gray = cv2.GaussianBlur(gray, (0, 0), sig)
        info.update(presmooth_sigma=round(sig, 2), fwhm_raw=round(measured0, 2) if np.isfinite(measured0) else None)
    if fwhm == "auto":
        measured, n = estimate_fwhm(gray, disc)
        fwhm = detection_fwhm(measured)
        info.update(fwhm_measured=round(measured, 2), fwhm_measured_n=n)
    fwhm = float(fwhm)
    custom_mask = mask
    if mask is None:
        mask = disc_mask(gray.shape, *disc, margin=0.02 * disc[2])
    if tiles is None:
        tiles = config.get("tiles", "auto")
    if tiles == "auto":
        tiles = 3 if gray.size > AUTO_TILES_PIXELS else 1
    kw = dict(detect_kwargs, tiles=int(tiles), fwhm=fwhm)
    if config.get("box_scale") and "box_size" not in kw:
        kw["box_size"] = int(np.clip(round(128 * (2 * disc[2] / math.pi) / config.REF_F), 24, 256))
    det = detect_stars(gray, mask=mask, **kw)
    info.update({"fwhm": fwhm, "detect_kwargs": kw, "n_detections": len(det), "detect_s": round(_time.perf_counter() - t0, 2),
                 "exif": {k: v for k, v in read_exif(path).items() if k != "datetime"}})
    return Frame(Path(path), utc, exposure, gray.shape, det, disc, gray if keep_image else None, info,
                 custom_mask if keep_image else None)
