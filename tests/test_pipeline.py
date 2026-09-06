"""End-to-end test on one bundled frame (about 30 s)."""
from pathlib import Path

import numpy as np
import pytest

from allskycal import Site, calibrate, evaluate, load_frame

IMAGES = Path(__file__).resolve().parents[1] / "examples" / "images"
SITE = Site(43.259147, -6.60345, 650)


@pytest.mark.slow
def test_zero_shot_calibration_of_one_frame():
    frame = load_frame(IMAGES / "2026_08_09_03_00_46.jpg", tz="Europe/Madrid")
    assert len(frame.detections) > 1000
    result = calibrate([frame], SITE, verbose=False)
    s = result.summary()
    assert s["n_pairs"] > 200
    assert s["median_px"] < 1.0
    m = result.model
    assert 950 < m.f < 1060                       # 1.55 mm lens on 1.55 um pixels
    assert 2.5 < m.total_tilt < 5.0               # this camera is known to be ~3.7 deg off the zenith
    # the calibration transfers to another night
    other = load_frame(IMAGES / "2026_07_08_01_01_06.jpg", tz="Europe/Madrid")
    pairs = evaluate(m, [other], SITE)
    d = pairs.residuals(m)
    assert len(pairs) > 200
    assert np.median(d) < 1.2
