import math

import numpy as np

from ascal.model import CameraModel, fit, residuals, rotation_matrix


def make_model(dec=False):
    return CameraModel(cx=1948.3, cy=1467.9, f=1005.4, psi=161.06, tau_x=-1.8, tau_y=-3.25, k=[-0.0213, -0.005],
                       p=[-2.1e-4, -2.2e-4] if dec else None, width=4056, height=3040)


def test_rotation_is_orthonormal():
    R = rotation_matrix(-1.8, -3.25)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.allclose(rotation_matrix(0, 0), np.eye(3))


def test_project_unproject_roundtrip():
    for dec in (False, True):
        m = make_model(dec)
        rng = np.random.default_rng(1)
        alt = rng.uniform(3, 89, 500)
        az = rng.uniform(0, 360, 500)
        x, y = m.project(alt, az)
        alt2, az2 = m.unproject(x, y)
        assert np.all(np.isfinite(x))
        assert np.max(np.abs(alt2 - alt)) < 1e-6
        daz = (az2 - az + 180) % 360 - 180
        assert np.max(np.abs(daz * np.cos(np.radians(alt)))) < 1e-6


def test_zenith_and_tilt():
    m = make_model()
    assert abs(m.total_tilt - math.degrees(math.acos(math.cos(math.radians(-1.8)) * math.cos(math.radians(-3.25))))) < 1e-9
    zx, zy = m.zenith_pixel
    assert 50 < math.hypot(zx - m.cx, zy - m.cy) < 80          # ~3.7 deg * 1005 px/rad
    alt, az = m.unproject(np.array([zx]), np.array([zy]))
    assert alt[0] > 89.999


def test_equidistant_special_case():
    m = CameraModel(cx=0, cy=0, f=1000, k=[0.0, 0.0], width=10, height=10)
    th = np.radians(np.array([10, 45, 80]))
    assert np.allclose(m.radius(th), 1000 * th)
    assert np.allclose(m.plate_scale(np.array([0, 45, 90])), 1000 * math.pi / 180)


def test_fit_recovers_parameters_from_synthetic_stars():
    truth = make_model()
    rng = np.random.default_rng(3)
    alt = rng.uniform(5, 89, 3000)
    az = rng.uniform(0, 360, 3000)
    x, y = truth.project(alt, az)
    x += rng.normal(0, 0.5, x.size)
    y += rng.normal(0, 0.5, y.size)
    start = CameraModel(cx=2028, cy=1520, f=1000, psi=160, k=[-0.03, 0.0], width=4056, height=3040)
    m = fit(start, alt, az, x, y, loss="linear")
    assert abs(m.cx - truth.cx) < 0.2 and abs(m.cy - truth.cy) < 0.2
    assert abs(m.f - truth.f) < 0.3
    assert abs(m.psi - truth.psi) < 0.002
    assert abs(m.tau_x - truth.tau_x) < 0.005 and abs(m.tau_y - truth.tau_y) < 0.005
    d = np.linalg.norm(residuals(m, alt, az, x, y), axis=1)
    assert 0.5 < np.median(d) < 0.7                 # noise level, not model error


def test_serialisation_roundtrip(tmp_path):
    m = make_model(True)
    m.save(tmp_path / "c.json")
    m2 = CameraModel.load(tmp_path / "c.json")
    assert np.allclose(m.to_vector(), m2.to_vector())
    assert m2.width == 4056
