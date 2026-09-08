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
