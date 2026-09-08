"""Command-line interface.

    ascal calibrate IMG [IMG ...] --lat LAT --lon LON [--elev M] [--time ISO | --tz ZONE] [--decentering] [--out calib.json] [--report DIR]
    ascal check calib.json IMG [IMG ...] --lat LAT --lon LON [--tz ZONE] [--report DIR]
    ascal project calib.json --alt A --az Z | --x X --y Y
    ascal web [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import __version__
from .bootstrap import Site, calibrate, evaluate
from .detect import load_frame
from .model import CameraModel, band_statistics


def _site(args) -> Site:
    return Site(args.lat, args.lon, args.elev)


def _times(args, n: int) -> List[Optional[datetime]]:
    if not args.time:
        return [None] * n
    if len(args.time) not in (1, n):
        raise SystemExit("--time must be given once or once per image")
    ts = [datetime.fromisoformat(t) for t in args.time]
    return ts if len(ts) == n else ts * n


def _load_frames(args, keep_image: bool = False):
    frames = []
    for path, t in zip(args.images, _times(args, len(args.images))):
        fr = load_frame(path, time=t, tz=args.tz, exposure_s=args.exposure, tiles=args.tiles, keep_image=keep_image)
        print(f"{Path(path).name}: {len(fr.detections)} detections, mid-exposure {fr.utc:%Y-%m-%d %H:%M:%S} UTC, "
              f"disc centre ({fr.disc[0]:.0f}, {fr.disc[1]:.0f}) radius {fr.disc[2]:.0f} px", flush=True)
        frames.append(fr)
    return frames


def _report(result, frames, site, out_dir: Path) -> None:
    from . import plots
    out_dir.mkdir(parents=True, exist_ok=True)
    result.model.save(out_dir / "calibration.json")
    (out_dir / "summary.json").write_text(json.dumps(result.summary(), indent=1, default=float))
    plots.residual_plots(result, out_dir / "residuals.png")
    plots.radial_plot(result.model, out_dir / "radial.png")
    for fr in frames[:3]:
        plots.overlay(fr, result.model, site, path=out_dir / f"overlay_{fr.path.stem}.png")
        plots.cutouts(fr, result.model, site, path=out_dir / f"cutouts_{fr.path.stem}.png")
    p = result.pairs
    with open(out_dir / "pairs.csv", "w") as fh:
        fh.write("frame,hip,mag,alt,az,x,y,flux,residual_px,inlier\n")
        d = result.residual_px
        for i in range(len(p)):
            fh.write(f"{p.frame[i]},{p.hip[i]},{p.mag[i]:.2f},{p.alt[i]:.4f},{p.az[i]:.4f},{p.x[i]:.3f},{p.y[i]:.3f},{p.flux[i]:.1f},{d[i]:.3f},{int(result.inliers[i])}\n")
    print(f"report written to {out_dir}/")


def cmd_calibrate(args) -> int:
    site = _site(args)
    frames = _load_frames(args, keep_image=bool(args.report))
    result = calibrate(frames, site, decentering=args.decentering, verbose=not args.quiet)
    s = result.summary()
    print(f"\n{result.model}")
    print(f"total tilt {result.model.total_tilt:.2f} deg, zenith at pixel ({result.model.zenith_pixel[0]:.0f}, {result.model.zenith_pixel[1]:.0f}), "
          f"horizon radius {result.model.horizon_radius:.0f} px, {60 / float(result.model.plate_scale(0)):.2f} arcmin/px on axis")
    print(f"{s['n_pairs']} pairs, median {s['median_px']:.2f} px, rms {s['rms_px']:.2f} px, p90 {s['p90_px']:.2f} px, "
          f"{100 * s['within_1px']:.0f}% within 1 px, {s['elapsed_s']} s")
    for b in s["bands"]:
        if b["n"]:
            print(f"  alt {b['band']:>6}: n {b['n']:5d}  median {b['median']:.2f}  p90 {b['p90']:.2f}")
    out = Path(args.out) if args.out else None
    if out:
        result.model.save(out)
        print(f"calibration written to {out}")
    if args.report:
        _report(result, frames, site, Path(args.report))
    return 0


def cmd_check(args) -> int:
    site = _site(args)
    model = CameraModel.load(args.calibration)
    frames = _load_frames(args, keep_image=bool(args.report))
    pairs = evaluate(model, frames, site, max_mag=args.max_mag, radius=args.radius)
    d = pairs.residuals(model)
    print(f"\n{len(pairs)} associations (m <= {args.max_mag}, radius {args.radius} px): median {np.median(d):.2f} px, "
          f"p90 {np.percentile(d, 90):.2f} px, {100 * np.mean(d < 1):.0f}% within 1 px")
    for b in band_statistics(pairs.alt, d):
        if b["n"]:
            print(f"  alt {b['band']:>6}: n {b['n']:5d}  median {b['median']:.2f}  p90 {b['p90']:.2f}")
    if args.report:
        from . import plots
        out = Path(args.report)
        out.mkdir(parents=True, exist_ok=True)
        for fr in frames:
            plots.overlay(fr, model, site, path=out / f"check_{fr.path.stem}.png")
        print(f"overlays written to {out}/")
    return 0


def cmd_project(args) -> int:
    model = CameraModel.load(args.calibration)
    if args.alt is not None and args.az is not None:
        x, y = model.project(np.array([args.alt]), np.array([args.az]))
        print(f"alt {args.alt} az {args.az} -> x {x[0]:.2f} y {y[0]:.2f}")
    elif args.x is not None and args.y is not None:
        alt, az = model.unproject(np.array([args.x]), np.array([args.y]))
        print(f"x {args.x} y {args.y} -> alt {alt[0]:.4f} az {az[0]:.4f}")
    else:
        raise SystemExit("give --alt/--az or --x/--y")
    return 0


def cmd_web(args) -> int:
    import uvicorn
    uvicorn.run("ascal.web.app:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="ascal", description="Geometric calibration of all-sky cameras from star positions.")
    ap.add_argument("--version", action="version", version=f"ascal {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def site_args(p):
        p.add_argument("--lat", type=float, required=True, help="latitude, degrees north")
        p.add_argument("--lon", type=float, required=True, help="longitude, degrees east")
        p.add_argument("--elev", type=float, default=0.0, help="elevation, metres (informational)")

    def frame_args(p):
        p.add_argument("images", nargs="+")
        p.add_argument("--time", nargs="*", help="exposure start (ISO 8601, aware or in --tz); default: EXIF or file name")
        p.add_argument("--tz", default=None, help="time zone of naive times / file names (e.g. Europe/Madrid); default UTC")
        p.add_argument("--exposure", type=float, default=None, help="exposure in seconds if not in EXIF")
        p.add_argument("--tiles", type=int, default=1, help="detect in N x N tiles to limit memory (e.g. 2 on a Raspberry Pi)")
        p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("calibrate", help="zero-shot calibration from one or more frames")
    frame_args(p)
    site_args(p)
    p.add_argument("--decentering", action="store_true", help="fit the Brown-Conrady decentering term (model B)")
    p.add_argument("--out", help="write the calibration JSON here")
    p.add_argument("--report", help="write figures, pairs and summary to this directory")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("check", help="score a calibration on other frames (no fitting)")
    p.add_argument("calibration")
    frame_args(p)
    site_args(p)
    p.add_argument("--max-mag", type=float, default=5.5)
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--report", help="write overlays to this directory")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("project", help="convert between sky and pixel coordinates")
    p.add_argument("calibration")
    p.add_argument("--alt", type=float)
    p.add_argument("--az", type=float)
    p.add_argument("--x", type=float)
    p.add_argument("--y", type=float)
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("web", help="run the drag-and-drop web demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_web)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
