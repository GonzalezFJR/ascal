"""Precompute the example results offered by the web app.

    python deploy/precompute_examples.py [--sources DIR ...] [--only ID,ID]

For every entry of examples/web/examples.json, the image (``file``, looked up next to examples.json and then in each
``--sources`` directory) is calibrated with the web worker, exactly as an upload would be, and the outputs are written
to examples/web/precomputed/<id>/ (result.json, display.jpg, thumb.jpg, calibration.json, pairs.csv, panel.png,
log.txt). The source images are not copied. precomputed/ is not tracked by git; deploy it with the rest of the tree.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "web"
OUT = EX / "precomputed"
KEEP = ("result.json", "display.jpg", "calibration.json", "pairs.csv", "panel.png", "log.txt")


def suffix(name: str) -> str:
    n = name.lower()
    return ".fits.gz" if n.endswith(".fits.gz") else ".fit.gz" if n.endswith(".fit.gz") else Path(n).suffix


def thumbnail(d: Path, size: int = 360) -> None:
    from PIL import Image
    res = json.loads((d / "result.json").read_text())
    k = res["display_scale"]
    cx, cy = (v * k for v in res["markers"]["centre"])           # optical centre, in the pixels of the file
    r = res["model"]["derived"]["horizon_radius_px"] * k
    im = Image.open(d / "display.jpg").convert("RGB")
    half = min(r * 1.0, max(im.size) / 2)
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    sq = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), "black")
    sq.paste(im.crop((max(box[0], 0), max(box[1], 0), min(box[2], im.width), min(box[3], im.height))),
             (max(-box[0], 0), max(-box[1], 0)))
    sq.resize((size, size), Image.LANCZOS).save(d / "thumb.jpg", quality=86)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=[], help="directories with the example images")
    ap.add_argument("--only", default="", help="comma-separated example ids")
    ap.add_argument("--display-px", default="3072", help="long side of the stored display image")
    ap.add_argument("--thumbs-only", action="store_true", help="only redraw the thumbnails of existing results")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    for e in json.loads((EX / "examples.json").read_text()):
        if only and e["id"] not in only:
            continue
        if a.thumbs_only:
            if (OUT / e["id"] / "result.json").exists():
                thumbnail(OUT / e["id"])
            continue
        src = next((p for p in [EX / e["file"]] + [Path(s) / e["file"] for s in a.sources] if p.exists()), None)
        if src is None:
            print(f"{e['id']}: image {e['file']} not found, skipped")
            continue
        job = Path(tempfile.mkdtemp(prefix="ascal_ex_"))
        name = "frame" + suffix(src.name)
        os.symlink(src.resolve(), job / name)
        (job / "params.json").write_text(json.dumps({"file": name, "original_name": src.name, "lat": e["lat"], "lon": e["lon"],
                                                     "elev": e.get("elev", 0), "utc": e["utc"], "exposure": e.get("exposure"),
                                                     "max_time": 90, "parity": "auto", "fwhm": "auto"}))
        env = dict(os.environ, ASCAL_WEB_DISPLAY_PX=a.display_px, PYTHONPATH=str(ROOT))
        rc = subprocess.run([sys.executable, "-m", "ascal.web.worker", str(job)], env=env).returncode
        if rc != 0 or not (job / "result.json").exists():
            print(f"{e['id']}: FAILED\n" + (job / "log.txt").read_text()[-2000:])
            continue
        d = OUT / e["id"]
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for f in KEEP:
            if (job / f).exists():
                shutil.copy(job / f, d / f)
        thumbnail(d)
        shutil.rmtree(job, ignore_errors=True)
        s = json.loads((d / "result.json").read_text())["summary"]
        print(f"{e['id']}: {s['n_pairs']} stars, median {s['median_px']:.2f} px, "
              f"{sum(f.stat().st_size for f in d.iterdir()) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
