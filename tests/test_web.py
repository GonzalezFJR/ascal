"""Web app: configuration, input validation and one example job end to end."""
import importlib
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCAL_WEB_DATA", str(tmp_path))
    import ascal.web.app as webapp
    importlib.reload(webapp)
    return TestClient(webapp.app)


def test_config_and_page(client):
    c = client.get("/api/config").json()
    assert c["version"] and c["max_mb"] > 0 and c["max_time"] == 180
    assert "static/app.js" in client.get("/").text
    for e in c["examples"]:                     # only examples with precomputed results are listed
        assert client.get(f"/api/examples/{e['id']}/result.json").status_code == 200
        assert client.get(f"/api/examples/{e['id']}/thumb.jpg").status_code == 200
    assert client.get("/api/examples/nope/result.json").status_code == 404


def test_validation(client):
    base = {"lat": "43", "lon": "-6", "utc": "2026-08-09T01:00:46"}
    img = {"image": ("frame.jpg", b"\xff\xd8\xff", "image/jpeg")}
    assert client.post("/api/jobs", data={**base, "lat": "95"}, files=img).status_code == 422
    assert client.post("/api/jobs", data={**base, "elev": "7000"}, files=img).status_code == 422
    assert client.post("/api/jobs", data={**base, "fwhm": "40"}, files=img).status_code == 422
    assert client.post("/api/jobs", data={**base, "utc": "yesterday"}, files=img).status_code == 422
    r = client.post("/api/jobs", data=base, files={"image": ("frame.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415
    assert client.get("/api/jobs/doesnotexist0000000").status_code == 404


@pytest.mark.slow
def test_upload_job(client):
    from pathlib import Path
    frame = Path(__file__).resolve().parents[1] / "examples" / "images" / "2026_08_09_03_00_46.jpg"
    r = client.post("/api/jobs", data={"lat": "43.259147", "lon": "-6.60345", "elev": "650", "utc": "2026-08-09T01:00:46",
                                       "exposure": "20"}, files={"image": (frame.name, frame.read_bytes(), "image/jpeg")})
    job = r.json()["id"]
    for _ in range(120):
        s = client.get(f"/api/jobs/{job}").json()
        if s["state"] in ("done", "error"):
            break
        time.sleep(1)
    assert s["state"] == "done", s
    res = client.get(f"/api/jobs/{job}/result.json").json()
    assert res["summary"]["n_pairs"] > 400 and res["summary"]["median_px"] < 0.8
    assert len(res["layers"]["constellations"]) > 20 and res["layers"]["grid"]["alt"]
    for name in ("display.jpg", "calibration.json", "pairs.csv", "panel.png"):
        assert client.get(f"/api/jobs/{job}/{name}").status_code == 200
