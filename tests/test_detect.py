"""Star-FWHM estimate and choice of the detection kernel (synthetic frames)."""
import numpy as np
import pytest

from ascal.cli import _fwhm_arg
from ascal.detect import DEFAULT_FWHM, FWHM_MAX, detection_fwhm, estimate_fwhm


def synthetic_frame(fwhm: float, n: int = 400, size: int = 1200, seed: int = 1) -> tuple:
    rng = np.random.default_rng(seed)
    img = rng.normal(40.0, 3.0, (size, size))
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[-15:16, -15:16]
    for x, y, a in zip(rng.uniform(40, size - 40, n), rng.uniform(40, size - 40, n), rng.uniform(40, 150, n)):
        xi, yi = int(x), int(y)
        img[yi - 15:yi + 16, xi - 15:xi + 16] += a * np.exp(-((xx - (x - xi)) ** 2 + (yy - (y - yi)) ** 2) / (2 * sigma ** 2))
    img[0, 0] = 255.0                                        # saturation level of the "camera"
    return img.astype(np.float32), (size / 2, size / 2, size / 2)


@pytest.mark.parametrize("fwhm", [2.8, 4.7, 7.0])
def test_estimate_fwhm_recovers_the_star_width(fwhm):
    img, disc = synthetic_frame(fwhm)
    measured, n = estimate_fwhm(img, disc)
    assert n >= 50
    assert measured == pytest.approx(fwhm, rel=0.1)


def test_detection_fwhm_keeps_the_paper_kernel_for_sharp_stars():
    assert detection_fwhm(2.8) == DEFAULT_FWHM              # ZRO camera: unchanged 4 px kernel
    assert detection_fwhm(4.7) == pytest.approx(6.11)       # DSLR frame with wide stars
    assert detection_fwhm(40.0) == FWHM_MAX
    assert detection_fwhm(float("nan")) == DEFAULT_FWHM


def test_fwhm_argument():
    assert _fwhm_arg("auto") == "auto"
    assert _fwhm_arg("6") == 6.0
    with pytest.raises(Exception):
        _fwhm_arg("wide")
