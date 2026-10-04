"""Diagnostic figures (matplotlib): overlay, residuals, radial function."""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .bootstrap import CalibrationResult, Site
from .catalog import sky_stars
from .detect import Frame
from .model import CameraModel

BLUE, ORANGE, GREY, CYAN = "#0072B2", "#D55E00", "#7F7F7F", "#56B4E9"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    return plt


def _output(fig, path: Optional[Path | str], dpi: int = 150):
    if path is None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        _plt().close(fig)
        return buf.getvalue()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _plt().close(fig)
    return Path(path)


def overlay(frame: Frame, model: CameraModel, site: Site, *, max_mag: float = 4.5, min_alt: float = 3.0, downsample: int = 4,
            show_detections: bool = False, grid: bool = True, path: Optional[Path | str] = None, title: Optional[str] = None):
    """Frame with the catalogue stars projected by ``model`` (and optionally the detections)."""
    plt = _plt()
    if frame.gray is None:
        from .detect import read_image
        gray = read_image(frame.path)
    else:
        gray = frame.gray
    h, w = gray.shape
    stars = sky_stars(site.lat, site.lon, frame.utc, min_alt=min_alt, max_mag=max_mag, model=model)
    small = gray[::downsample, ::downsample]
    inside = np.hypot(np.arange(small.shape[1])[None, :] * downsample - model.cx, np.arange(small.shape[0])[:, None] * downsample - model.cy) < model.horizon_radius
    lo, hi = np.percentile(small[inside], [5, 99.7]) if inside.any() else np.percentile(small, [5, 99.7])
    fig, ax = plt.subplots(figsize=(9, 9 * h / w))
    ax.imshow(small, cmap="gray", vmin=lo, vmax=hi, extent=(0, w, h, 0))
    if show_detections:
        ax.scatter(frame.detections.x, frame.detections.y, s=6, facecolors="none", edgecolors=CYAN, linewidths=0.4, label=f"detections ({len(frame.detections)})")
    s = stars.in_frame
    ax.scatter(stars.x[s], stars.y[s], s=40 * 10 ** (-0.25 * stars.mag[s]) + 4, facecolors="none", edgecolors=ORANGE, linewidths=0.7,
               label=f"catalogue m <= {max_mag:g} ({int(s.sum())})")
    bright = s & (stars.mag <= 1.6)
    for n, x, y in zip(np.array(stars.names)[bright], stars.x[bright], stars.y[bright]):
        ax.annotate(n, (x, y), xytext=(4, 4), textcoords="offset points", color=ORANGE, fontsize=7)
    if grid:
        azg = np.linspace(0, 360, 361)
        for a in (0, 30, 60):
            px, py = model.project(np.full_like(azg, float(a)), azg)
            ax.plot(px, py, color="white", lw=0.5, alpha=0.6, ls="--" if a == 0 else "-")
        altg = np.linspace(0, 89, 90)
        for a0, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            px, py = model.project(altg, np.full_like(altg, float(a0)))
            ax.plot(px, py, color="white", lw=0.5, alpha=0.6)
            ax.text(px[0], py[0], name, color="white", fontsize=9, fontweight="bold", ha="center", va="center")
    zx, zy = model.zenith_pixel
    ax.plot(zx, zy, "*", color=BLUE, ms=10, label="zenith")
    ax.plot(model.cx, model.cy, "+", color=ORANGE, ms=10, mew=1.5, label="optical centre")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, labelcolor="white", facecolor="black", framealpha=0.5, frameon=True)
    ax.set_title(title or f"{frame.path.name}   {frame.utc:%Y-%m-%d %H:%M:%S} UTC", fontsize=9)
    return _output(fig, path)


def cutouts(frame: Frame, model: CameraModel, site: Site, *, n: int = 6, half: int = 40, path: Optional[Path | str] = None):
    """Cut-outs around the brightest stars at low and high altitude with detections and predictions."""
    plt = _plt()
    gray = frame.gray if frame.gray is not None else __import__("ascal.detect", fromlist=["read_image"]).read_image(frame.path)
    stars = sky_stars(site.lat, site.lon, frame.utc, min_alt=8.0, max_mag=3.0, model=model)
    s = np.flatnonzero(stars.in_frame)
    low = s[(stars.alt[s] < 30)][np.argsort(stars.mag[s][stars.alt[s] < 30])][: n // 2]
    high = s[(stars.alt[s] >= 30)][np.argsort(stars.mag[s][stars.alt[s] >= 30])][: n - len(low)]
    picks = list(low) + list(high)
    cols = max(1, min(3, len(picks)))
    rows = max(1, math.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, j in zip(axes.ravel(), picks):
        cx0, cy0 = int(round(stars.x[j])), int(round(stars.y[j]))
        x0, y0 = max(cx0 - half, 0), max(cy0 - half, 0)
        crop = gray[y0:y0 + 2 * half, x0:x0 + 2 * half]
        v0, v1 = np.percentile(crop, [10, 99.9])
        ax.imshow(crop, cmap="gray", vmin=v0, vmax=v1, extent=(x0, x0 + 2 * half, y0 + 2 * half, y0))
        near = (np.abs(frame.detections.x - cx0) < half) & (np.abs(frame.detections.y - cy0) < half)
        ax.scatter(frame.detections.x[near], frame.detections.y[near], s=80, facecolors="none", edgecolors=CYAN, linewidths=0.9)
        ax.plot(stars.x[j], stars.y[j], "+", color=ORANGE, ms=10, mew=1.2)
        ax.set_title(f"{stars.names[j]}  m={stars.mag[j]:.1f}  alt {stars.alt[j]:.0f}°", fontsize=8)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("circles: detections;  +: predicted position", fontsize=8)
    return _output(fig, path)


def residual_plots(result: CalibrationResult, path: Optional[Path | str] = None):
    """Residual against altitude, residual vectors on the sensor and residual histogram."""
    plt = _plt()
    pairs, keep, model = result.pairs, result.inliers, result.model
    from .model import residuals as _res
    r = _res(model, pairs.alt, pairs.az, pairs.x, pairs.y)
    d = np.hypot(r[:, 0], r[:, 1])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.scatter(pairs.alt[~keep], d[~keep], s=4, color=GREY, alpha=0.4, label="clipped")
    ax.scatter(pairs.alt[keep], d[keep], s=4, color=BLUE, alpha=0.5, label="retained")
    edges = np.arange(0, 91, 10)
    med = [np.median(d[keep][(pairs.alt[keep] >= lo) & (pairs.alt[keep] < hi)]) if np.any((pairs.alt[keep] >= lo) & (pairs.alt[keep] < hi)) else np.nan
           for lo, hi in zip(edges[:-1], edges[1:])]
    ax.plot(0.5 * (edges[1:] + edges[:-1]), med, "o-", color=ORANGE, ms=4, label="median per 10°")
    ax.set_ylim(0, min(10, max(3, np.percentile(d, 99))))
    ax.set_xlabel("altitude (°)")
    ax.set_ylabel("residual |predicted − detected| (px)")
    ax.legend(fontsize=7)
    ax = axes[1]
    scale = 60.0
    ax.quiver(pairs.x[keep], pairs.y[keep], r[keep, 0] * scale, r[keep, 1] * scale, d[keep], cmap="viridis", angles="xy", scale_units="xy", scale=1, width=0.003)
    ax.add_patch(plt.Rectangle((0, 0), model.width, model.height, fill=False, lw=0.8, color="k"))
    ax.set_xlim(-50, model.width + 50)
    ax.set_ylim(model.height + 50, -50)
    ax.set_aspect("equal")
    ax.set_title(f"residual vectors ×{scale:.0f}", fontsize=8)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax = axes[2]
    ax.hist(np.clip(d[keep], 0, 3), bins=np.arange(0, 3.01, 0.1), color=BLUE, alpha=0.8)
    ax.axvline(np.median(d[keep]), color=ORANGE, ls="--", label=f"median {np.median(d[keep]):.2f} px")
    ax.set_xlabel("residual (px)")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=7)
    fig.suptitle(f"{keep.sum()} pairs, median {np.median(d[keep]):.2f} px, {100 * np.mean(d[keep] < 1):.0f}% within 1 px", fontsize=9)
    fig.tight_layout()
    return _output(fig, path)


def radial_plot(model: CameraModel, path: Optional[Path | str] = None):
    """Zenith distance against radius and local plate scale, compared with the ideal projections."""
    plt = _plt()
    th = np.radians(np.linspace(0, 90, 300))
    f = model.f
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    ax = axes[0]
    for name, r, ls in (("equidistant", f * th, "--"), ("equisolid", 2 * f * np.sin(th / 2), "-."), ("stereographic", 2 * f * np.tan(th / 2), ":"), ("orthographic", f * np.sin(th), (0, (1, 1)))):
        ax.plot(r, np.degrees(th), color=GREY, ls=ls, lw=0.9, label=name)
    ax.plot(model.radius(th), np.degrees(th), color=BLUE, lw=1.8, label="fitted")
    ax.set_xlabel("radius from optical centre (px)")
    ax.set_ylabel("zenith distance (°)")
    ax.set_xlim(0, 1.2 * model.horizon_radius)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(np.degrees(th), model.plate_scale(np.degrees(th)), color=BLUE, lw=1.8, label="fitted")
    ax.axhline(f * math.pi / 180, color=GREY, ls="--", lw=0.9, label="equidistant")
    ax.set_xlabel("zenith distance (°)")
    ax.set_ylabel("plate scale (px per °)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _output(fig, path)


# ---------------------------------------------------------------------- summary panel
_CONSTELLATIONS: Optional[dict] = None


def constellation_lines() -> dict:
    """Constellation figures as J2000 polylines {abbreviation: [[[ra, dec], ...], ...]} (d3-celestial, BSD 3-Clause)."""
    global _CONSTELLATIONS
    if _CONSTELLATIONS is None:
        import json
        _CONSTELLATIONS = json.loads((Path(__file__).resolve().parent / "data" / "constellation_lines.json").read_text())["lines"]
    return _CONSTELLATIONS


def _radec_polyline_altaz(ra, dec, lat, lon, t, n: int = 16):
    """Alt/az (apparent if refraction is on) of a J2000 polyline, densified along great circles."""
    from . import config
    from .catalog import precess_j2000, radec_to_altaz, sidereal_time_deg
    from .model import altaz_to_vec, vec_to_altaz
    ra_d, dec_d = precess_j2000(np.asarray(ra, float), np.asarray(dec, float), t)
    alt, az = radec_to_altaz(ra_d, dec_d, lat, sidereal_time_deg(lon, t))
    v = altaz_to_vec(alt, az)
    out = [v[:1]]
    for v0, v1 in zip(v[:-1], v[1:]):
        w = float(np.arccos(np.clip(v0 @ v1, -1, 1)))
        s = np.linspace(0, 1, n)[1:, None]
        out.append(v1[None] if w < 1e-6 else (np.sin((1 - s) * w) * v0 + np.sin(s * w) * v1) / np.sin(w))
    a, z = vec_to_altaz(np.vstack(out))
    if config.get("refraction"):
        a = a + config.refraction_deg(a, config.get("elev", 0.0))
    return a, z


def _display_image(gray: np.ndarray, max_side: int = 1600) -> np.ndarray:
    step = max(1, int(math.ceil(max(gray.shape) / max_side)))
    small = gray[::step, ::step].astype(np.float32)
    lo, hi = np.percentile(small, [1.0, 99.7])
    return np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.6


def calibration_panel(frame: Frame, result: CalibrationResult, site: Site, *, frame_index: int = 0,
                      path: Optional[Path | str] = None, title: Optional[str] = None):
    """Four-panel summary of a calibration on one frame:
    (a) the frame as recorded; (b) matched stars with the altitude circles (0, 30, 60 deg), the N-S and E-W lines and
    the constellation figures projected by the calibration; (c) altitude in the camera frame against distance to the
    optical centre (model, equidistant projection of the same focal length, matched stars); (d) residual against altitude."""
    plt = _plt()
    from .detect import read_image
    model = result.model
    gray = frame.gray if frame.gray is not None else read_image(frame.path)
    W, H = model.width, model.height
    sel = (result.pairs.frame == frame_index) & result.inliers
    p = result.pairs
    alt, az, x, y, res = p.alt[sel], p.az[sel], p.x[sel], p.y[sel], result.residual_px[sel]

    def clip(px, py):
        px, py = np.array(px, float), np.array(py, float)
        bad = ~((px >= 0) & (px < W) & (py >= 0) & (py < H))
        px[bad], py[bad] = np.nan, np.nan
        return px, py

    def square(ax):
        s = max(W, H) / 2
        ax.set_xlim(W / 2 - s, W / 2 + s)
        ax.set_ylim(H / 2 + s, H / 2 - s)
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])

    img = _display_image(gray)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 7.4))
    (a, b), (c, e) = axs
    for ax in (a, b):
        ax.imshow(img, cmap="gray", extent=(0, W, H, 0), interpolation="lanczos", vmin=0, vmax=1)
        square(ax)
        for sp in ax.spines.values():
            sp.set_visible(True)
    a.set_title("(a) frame as recorded")
    b.set_title("(b) matched stars and calibration")
    guide = dict(color="#d8d8d8", lw=0.6, alpha=0.75)
    azg = np.linspace(0, 360, 721)
    for a0 in (0, 30, 60):
        b.plot(*clip(*model.project(np.full_like(azg, a0 + (0.2 if a0 == 0 else 0.0)), azg)), ls="--" if a0 == 0 else "-", **guide)
    altg = np.linspace(0.2, 90, 120)
    for z0 in (0, 90, 180, 270):
        b.plot(*clip(*model.project(altg, np.full_like(altg, float(z0)))), **guide)
    for z0, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        px, py = model.project(np.array([6.0]), np.array([float(z0)]))
        if np.isfinite(px[0]) and 0 <= px[0] < W and 0 <= py[0] < H:
            b.text(px[0], py[0], name, color="white", fontsize=8, fontweight="bold", ha="center", va="center")
    for polylines in constellation_lines().values():
        for line in polylines:
            ra, dec = np.array(line).T
            la, lz = _radec_polyline_altaz(ra, dec, site.lat, site.lon, frame.utc)
            if np.all(la < 0.5):
                continue
            la = np.where(la > 0.5, la, np.nan)
            b.plot(*clip(*model.project(la, lz)), color="#ff9f1c", lw=0.6, alpha=0.5)
    b.scatter(x, y, s=12, facecolors="none", edgecolors="#4da3ff", linewidths=0.6, alpha=0.7, label=f"matched stars ({sel.sum()})")
    b.legend(loc="lower left", fontsize=6.5, labelcolor="white", facecolor="black", framealpha=0.55, frameon=True, markerscale=1.6,
             borderpad=0.3, handletextpad=0.2)

    xu = (W - 1) - x if model.mirror else x
    r_det = np.hypot(xu - model.cx, y - model.cy)
    alt_c, _ = model.sky_to_camera(alt, az)
    th = np.radians(np.linspace(0, 90, 181))
    c.scatter(r_det, alt_c, s=7, color=BLUE, edgecolors="none", alpha=0.55, label="matched stars", zorder=2)
    c.plot(model.radius(th), 90 - np.degrees(th), color="#c0392b", lw=0.9, label="model", zorder=3)
    c.plot(model.f * th, 90 - np.degrees(th), color="0.45", lw=0.8, ls="--", label="equidistant, same f", zorder=1)
    c.set_xlabel("distance to the optical centre (px)")
    c.set_ylabel("altitude in the camera frame (°)")
    c.set_ylim(0, 90)
    c.set_xlim(0, model.horizon_radius * 1.03)
    c.grid(alpha=0.3)
    c.legend(fontsize=7, loc="upper right", frameon=True)
    c.set_title("(c) projection: altitude vs radius")
    c.set_box_aspect(1)

    top = max(2.5, 1.25 * float(np.percentile(res, 98))) if res.size else 2.5
    e.scatter(alt, res, s=4, color=BLUE, edgecolors="none", alpha=0.7, label="matched")
    mid, med = [], []
    for lo in range(0, 90, 10):
        s = (alt >= lo) & (alt < lo + 10)
        if s.sum() >= 5:
            mid.append(lo + 5)
            med.append(float(np.median(res[s])))
    e.plot(mid, med, color="k", lw=1.1, marker="o", ms=3, label="median per 10°")
    if res.size:
        e.axhline(float(np.median(res)), color="0.5", ls="--", lw=0.8, label=f"overall median {np.median(res):.2f} px")
    e.set_xlim(0, 90)
    e.set_ylim(0, top * 1.3)
    e.grid(alpha=0.3)
    e.set_xlabel("altitude (°)")
    e.set_ylabel("residual (px)")
    e.legend(fontsize=6.5, loc="upper center", ncol=2, frameon=True, framealpha=0.9)
    e.set_title("(d) residual vs altitude")
    e.set_box_aspect(1)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93 if title else 0.96, bottom=0.07, wspace=0.22, hspace=0.2)
    return _output(fig, path, dpi=200)
