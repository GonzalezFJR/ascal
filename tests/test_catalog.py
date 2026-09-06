from datetime import datetime

import numpy as np

from allskycal.catalog import julian_date, load_catalog, precess_j2000, sidereal_time_deg, sky_stars


def test_julian_date_j2000():
    assert abs(julian_date(datetime(2000, 1, 1, 12)) - 2451545.0) < 1e-9


def test_sidereal_time_against_reference():
    # Greenwich apparent sidereal time on 2026-07-07 23:01:16 UTC, reference value from PyEphem
    lst = sidereal_time_deg(-6.60345, datetime(2026, 7, 7, 23, 1, 16))
    assert abs(lst - 264.63768) < 0.001            # < 4 arcsec


def test_precession_moves_stars_by_about_a_third_of_a_degree_in_2026():
    cat = load_catalog(max_mag=1.0)
    ra, dec = precess_j2000(cat.ra, cat.dec, datetime(2026, 7, 1))
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(dec)) * np.sin(np.radians(cat.dec)) +
                                       np.cos(np.radians(dec)) * np.cos(np.radians(cat.dec)) * np.cos(np.radians(ra - cat.ra)), -1, 1)))
    assert 0.2 < np.median(sep) < 0.45


def test_catalog_and_sky_stars():
    cat = load_catalog()
    assert len(cat) == 8789
    stars = sky_stars(43.259147, -6.60345, datetime(2026, 8, 9, 1, 0, 56), min_alt=0, max_mag=2.0, catalog=cat)
    names = set(stars.names)
    assert {"Vega", "Altair", "Deneb", "Capella"} <= names        # summer sky from northern Spain, 01 UTC
    assert "Sirius" not in names
    assert np.all(stars.alt >= 0)
