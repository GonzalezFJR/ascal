"""Version 1.0: mirrored images, FITS input, options and time limit."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import ascal
from ascal import CameraModel, Site, calibrate, config, evaluate, load_frame
from ascal.bootstrap import CalibrationError

IMAGES = Path(__file__).resolve().parents[1] / "examples" / "images"
SITE = Site(43.259147, -6.60345, 650)
FRAME = IMAGES / "2026_08_09_03_00_46.jpg"


def test_version():
    assert ascal.__version__.startswith("1.")


def test_mirror_model_round_trip(tmp_path):
    m = CameraModel(cx=1950.0, cy=1470.0, f=1005.0, psi=161.0, tau_x=-1.8, tau_y=-3.3, width=4056, height=3040, mirror=True)
    alt, az = np.array([20.0, 45.0, 80.0]), np.array([10.0, 200.0, 300.0])
    x, y = m.project(alt, az)
    a2, z2 = m.unproject(x, y)
    assert np.allclose(a2, alt, atol=1e-6) and np.allclose((z2 - az + 180) % 360 - 180, 0, atol=1e-6)
    direct = m.copy(mirror=False)
    xd, yd = direct.project(alt, az)
    assert np.allclose(x, 4055 - xd) and np.allclose(y, yd)
    m.save(tmp_path / "m.json")
    assert CameraModel.load(tmp_path / "m.json").mirror


def test_options_context():
    assert config.get("max_time") == 40.0
    with config.options(max_time=5, refraction=False):
        assert config.get("max_time") == 5 and not config.get("refraction")
    assert config.get("refraction")
    with pytest.raises(KeyError):
        with config.options(not_an_option=1):
            pass


@pytest.mark.slow
def test_mirrored_fits_frame_is_calibrated_in_its_own_pixels(tmp_path):
    """A left-right mirrored copy, written as FITS with the time in the header: parity found, pixels consistent."""
    import cv2
    from astropy.io import fits

    from ascal.detect import read_image
    gray = read_image(FRAME)[:, ::-1]
    hdu = fits.PrimaryHDU(gray.astype(np.float32))
    hdu.header["DATE-OBS"] = "2026-08-09T01:00:46"
    hdu.header["EXPTIME"] = 20.0
    path = tmp_path / "mirrored.fits"
    hdu.writeto(path)
    frame = load_frame(path, keep_image=True)
    assert frame.utc == datetime(2026, 8, 9, 1, 0, 56)          # mid-exposure from the header
    result = calibrate([frame], SITE, verbose=False)
    assert result.model.mirror
    assert result.info["cascade"]["parity"] == "mirror"
    assert result.summary()["median_px"] < 1.0
    # the mirrored model predicts the stars of the mirrored image
    pairs = evaluate(result.model, [frame], SITE)
    assert len(pairs) > 200 and np.median(pairs.residuals(result.model)) < 1.0


@pytest.mark.slow
def test_time_limit():
    frame = load_frame(FRAME, tz="Europe/Madrid", keep_image=True)
    with pytest.raises(CalibrationError):
        calibrate([frame], SITE, verbose=False, max_time=0.0)


def test_constellation_lines():
    from ascal import plots
    lines = plots.constellation_lines()
    assert len(lines) == 88
    ra, dec = np.array(lines["UMa"][0]).T
    assert np.all((ra >= 0) & (ra < 360)) and np.all(dec > 40)


@pytest.mark.slow
def test_cli_writes_report_with_panel_by_default(tmp_path, monkeypatch):
    from ascal.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["calibrate", str(FRAME), "--lat", "43.259147", "--lon", "-6.60345", "--elev", "650", "--tz", "Europe/Madrid", "--quiet"]) == 0
    out = tmp_path / f"{FRAME.stem}_ascal"
    for name in ("calibration.json", "summary.json", "pairs.csv", "panel.png", "residuals.png", "radial.png"):
        assert (out / name).stat().st_size > 0
    assert main(["calibrate", str(FRAME), "--lat", "43.259147", "--lon", "-6.60345", "--tz", "Europe/Madrid", "--quiet",
                 "--no-report", "--report", str(tmp_path / "none")]) == 0
    assert not (tmp_path / "none").exists()
