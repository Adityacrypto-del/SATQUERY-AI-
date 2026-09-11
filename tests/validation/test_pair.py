"""Tests for bi-temporal pair validation (build-order step 2).

This is the GeoTIFF demo path, not the benchmark path: CDVQA images are
ungeoreferenced PNGs, so georeferencing checks must degrade to "not
applicable" rather than failing a pair that is perfectly usable.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from tools.change_analysis.io import RSImage, load_rsimage
from tools.validation.pair import (
    CO_REGISTRATION_MIN_PSR,
    CO_REGISTRATION_WARN_PX,
    Severity,
    validate_pair,
)

TEST_CRS = CRS.from_epsg(32643)
TEST_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)


def _write(path, array, crs=TEST_CRS, transform=TEST_TRANSFORM, band_names=None):
    array = np.asarray(array, dtype=np.float32)
    count, height, width = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(array)
        if band_names:
            dst.descriptions = tuple(band_names)
    return str(path)


def _textured(height=64, width=64, shift_rows=0, shift_cols=0, seed=0):
    """A textured scene, optionally shifted, so phase correlation has signal."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (height * 2, width * 2)).astype(np.float32)
    base = np.cumsum(np.cumsum(base, axis=0), axis=1)  # low-frequency structure
    window = base[
        height // 2 + shift_rows: height // 2 + shift_rows + height,
        width // 2 + shift_cols: width // 2 + shift_cols + width,
    ]
    return window[None, :, :].astype(np.float32)


def _ungeoreferenced(array, modality="optical", band_names=None):
    return RSImage(
        array=np.asarray(array, dtype=np.float32),
        crs=None, transform=None, modality=modality,
        band_names=band_names, gsd_m=None,
    )


# --------------------------------------------------------------------------
# Structured report, never a bare bool
# --------------------------------------------------------------------------


def test_report_is_structured_and_lists_every_check(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured()))
    t2 = load_rsimage(_write(tmp_path / "b.tif", _textured()))

    report = validate_pair(t1, t2)

    assert report.ok is True
    assert not isinstance(report, bool)
    names = {check.name for check in report.checks}
    assert {"shape", "crs", "geotransform", "band_count", "modality",
            "gsd_ratio", "co_registration"} <= names


def test_matching_pair_passes_every_check(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured()))
    t2 = load_rsimage(_write(tmp_path / "b.tif", _textured()))

    report = validate_pair(t1, t2)

    assert report.ok
    assert all(c.severity is not Severity.FAIL for c in report.checks)


# --------------------------------------------------------------------------
# Hard failures
# --------------------------------------------------------------------------


def test_mismatched_shapes_fail(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured(64, 64)))
    t2 = load_rsimage(_write(tmp_path / "b.tif", _textured(32, 32)))

    report = validate_pair(t1, t2)

    assert not report.ok
    assert report.check("shape").severity is Severity.FAIL


def test_different_crs_fails_and_names_both(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured()))
    t2 = load_rsimage(
        _write(tmp_path / "b.tif", _textured(), crs=CRS.from_epsg(32644))
    )

    report = validate_pair(t1, t2)

    assert not report.ok
    detail = report.check("crs").detail
    assert "32643" in detail and "32644" in detail


def test_mismatched_band_count_fails(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", np.zeros((3, 32, 32))))
    t2 = load_rsimage(_write(tmp_path / "b.tif", np.zeros((4, 32, 32))))

    assert report_fails(validate_pair(t1, t2), "band_count")


def test_mismatched_modality_fails():
    t1 = _ungeoreferenced(np.zeros((1, 16, 16)), modality="optical")
    t2 = _ungeoreferenced(np.zeros((1, 16, 16)), modality="sar")

    assert report_fails(validate_pair(t1, t2), "modality")


def report_fails(report, name):
    return not report.ok and report.check(name).severity is Severity.FAIL


# --------------------------------------------------------------------------
# GSD and extent
# --------------------------------------------------------------------------


def test_large_gsd_ratio_fails(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured(32, 32)))
    t2 = load_rsimage(_write(
        tmp_path / "b.tif", _textured(32, 32),
        transform=Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 3000000.0),
    ))

    assert report_fails(validate_pair(t1, t2), "gsd_ratio")


def test_disjoint_extents_fail(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured(32, 32)))
    far = Affine(10.0, 0.0, 900000.0, 0.0, -10.0, 3900000.0)
    t2 = load_rsimage(_write(tmp_path / "b.tif", _textured(32, 32), transform=far))

    report = validate_pair(t1, t2)

    assert not report.ok
    assert report.check("geotransform").severity is Severity.FAIL


# --------------------------------------------------------------------------
# Co-registration (Townshend 1992; Dai & Khorram 1998)
# --------------------------------------------------------------------------


def test_aligned_pair_reports_subpixel_shift(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured(64, 64, seed=1)))
    t2 = load_rsimage(_write(tmp_path / "b.tif", _textured(64, 64, seed=1)))

    check = validate_pair(t1, t2).check("co_registration")

    assert check.severity is Severity.PASS
    assert check.value < CO_REGISTRATION_WARN_PX


def test_large_shift_warns_but_never_fails(tmp_path):
    """Advisory only. Phase correlation on a bi-temporal pair mixes true
    misregistration with real land-cover change and cannot separate them.
    Measured on 40 genuinely co-registered SECOND pairs: median apparent
    shift 6.20 px, 90% above 1 px -- a hard fail would reject nine in ten
    valid benchmark pairs."""
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured(64, 64, seed=1)))
    t2 = load_rsimage(
        _write(tmp_path / "b.tif", _textured(64, 64, shift_cols=5, seed=1))
    )

    report = validate_pair(t1, t2)
    check = report.check("co_registration")

    assert check.severity is Severity.WARN
    assert check.severity is not Severity.FAIL
    assert report.ok, "a large apparent shift must not reject the pair"


def test_structureless_pair_reports_no_estimate_rather_than_a_random_one(tmp_path):
    """With no shared structure the correlation argmax is noise. Reporting it
    as a confident shift is exactly the confident-wrong-answer failure."""
    rng = np.random.default_rng(0)
    flat = np.full((1, 64, 64), 0.04, dtype=np.float32)
    a = flat + rng.normal(0, 0.004, (1, 64, 64)).astype(np.float32)
    b = flat + rng.normal(0, 0.004, (1, 64, 64)).astype(np.float32)
    t1 = load_rsimage(_write(tmp_path / "a.tif", a))
    t2 = load_rsimage(_write(tmp_path / "b.tif", b))

    check = validate_pair(t1, t2).check("co_registration")

    assert check.severity is Severity.NOT_APPLICABLE
    assert check.value is None


# --------------------------------------------------------------------------
# The benchmark path: ungeoreferenced PNGs must not be failed
# --------------------------------------------------------------------------


def test_ungeoreferenced_pair_skips_geospatial_checks_without_failing():
    """CDVQA pairs are plain PNGs. They are valid input, not broken input."""
    t1 = _ungeoreferenced(_textured(32, 32, seed=2))
    t2 = _ungeoreferenced(_textured(32, 32, seed=2))

    report = validate_pair(t1, t2)

    assert report.ok
    for name in ("crs", "geotransform", "gsd_ratio"):
        assert report.check(name).severity is Severity.NOT_APPLICABLE
    # Co-registration needs no georeferencing, so it still runs.
    assert report.check("co_registration").severity is not Severity.NOT_APPLICABLE


def test_temporal_order_warns_when_reversed():
    t1 = _ungeoreferenced(np.zeros((1, 16, 16)))
    t2 = _ungeoreferenced(np.zeros((1, 16, 16)))

    report = validate_pair(
        t1, t2, t1_datetime="2024-06-01", t2_datetime="2023-01-15"
    )

    check = report.check("temporal_order")
    assert check.severity is Severity.FAIL
    assert report.ok is False


def test_temporal_order_is_not_applicable_without_metadata():
    t1 = _ungeoreferenced(np.zeros((1, 16, 16)))
    t2 = _ungeoreferenced(np.zeros((1, 16, 16)))

    report = validate_pair(t1, t2)

    assert report.check("temporal_order").severity is Severity.NOT_APPLICABLE


def test_report_serialises_for_the_trace(tmp_path):
    t1 = load_rsimage(_write(tmp_path / "a.tif", _textured()))
    t2 = load_rsimage(_write(tmp_path / "b.tif", _textured()))

    payload = validate_pair(t1, t2).to_dict()

    assert payload["ok"] is True
    assert isinstance(payload["checks"], list)
    assert all("severity" in c and isinstance(c["severity"], str)
               for c in payload["checks"])
