"""Tests for change detection and thresholding (build-order step 4).

Weighted toward the four failure modes that actually bite:

*   NaN reaching Otsu (one nodata pixel otherwise takes out the threshold),
*   a degenerate histogram with nothing to split,
*   SAR going through difference instead of ratio,
*   a threshold that is constant rather than derived from the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.detector import (
    detect_change,
    otsu_threshold,
)
from tools.change_analysis.io import RSImage


def _image(bands: dict, modality: str) -> RSImage:
    names = list(bands)
    stack = np.stack([np.asarray(bands[n], dtype=np.float32) for n in names])
    return RSImage(
        array=stack,
        crs=None,
        transform=None,
        modality=modality,
        band_names=names,
        gsd_m=None,
    )


def _bimodal(shape=(32, 32), low=0.0, high=1.0):
    """Half the pixels near ``low``, half near ``high``."""
    out = np.full(shape, low, dtype=np.float32)
    out[: shape[0] // 2] = high
    return out


# --------------------------------------------------------------------------
# Otsu
# --------------------------------------------------------------------------


def test_otsu_lands_between_the_two_modes():
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(10.0, 1.0, 2000), rng.normal(60.0, 2.0, 2000)]
    ).astype(np.float32)

    threshold = otsu_threshold(values)

    assert 12.0 < threshold < 58.0


def test_otsu_ignores_nan_and_still_returns_a_finite_threshold():
    """One nodata pixel must not take out the threshold for the whole tile."""
    rng = np.random.default_rng(0)
    clean = np.concatenate(
        [rng.normal(10.0, 1.0, 2000), rng.normal(60.0, 2.0, 2000)]
    ).astype(np.float32)
    poisoned = clean.copy()
    poisoned[0] = np.nan
    poisoned[1] = np.nan

    threshold = otsu_threshold(poisoned)

    assert np.isfinite(threshold)
    assert threshold == pytest.approx(otsu_threshold(clean[2:]), rel=0.05)


def test_otsu_on_a_constant_image_reports_no_threshold():
    """Nothing to split. None is honest; a number here would be invented."""
    assert otsu_threshold(np.full(100, 5.0, dtype=np.float32)) is None


def test_otsu_with_no_valid_pixels_reports_no_threshold():
    assert otsu_threshold(np.full(10, np.nan, dtype=np.float32)) is None


def test_otsu_threshold_moves_with_the_data():
    """Guards against a constant masquerading as a derived threshold."""
    low = np.concatenate([np.zeros(500), np.full(500, 10.0)]).astype(np.float32)
    high = np.concatenate([np.zeros(500), np.full(500, 100.0)]).astype(np.float32)

    assert otsu_threshold(low) != pytest.approx(otsu_threshold(high))


# --------------------------------------------------------------------------
# SAR: ratio, not difference
# --------------------------------------------------------------------------


def test_sar_uses_the_ratio_operator():
    t1 = _image({"VV": np.full((8, 8), 100.0, dtype=np.float32)}, "sar")
    t2 = _image({"VV": np.full((8, 8), 200.0, dtype=np.float32)}, "sar")

    result = detect_change(t1, t2)

    assert "ratio" in result.operator


def test_sar_ratio_is_invariant_to_backscatter_magnitude():
    """This is the whole reason ratio beats difference for SAR.

    Speckle is multiplicative, so a difference operator's false alarm rate
    climbs with backscatter magnitude. The same *relative* change must score
    the same whether it happens in a dark field or a bright one.
    """
    dim1 = _image({"VV": np.full((8, 8), 100.0, dtype=np.float32)}, "sar")
    dim2 = _image({"VV": np.full((8, 8), 200.0, dtype=np.float32)}, "sar")
    bright1 = _image({"VV": np.full((8, 8), 1000.0, dtype=np.float32)}, "sar")
    bright2 = _image({"VV": np.full((8, 8), 2000.0, dtype=np.float32)}, "sar")

    dim = detect_change(dim1, dim2).change_map
    bright = detect_change(bright1, bright2).change_map

    # Both are a doubling; the change magnitude must agree.
    assert float(np.nanmean(dim)) == pytest.approx(
        float(np.nanmean(bright)), rel=1e-4
    )
    # And a raw difference operator would emphatically not agree, which is
    # what makes this test meaningful rather than tautological.
    assert abs(200.0 - 100.0) != pytest.approx(abs(2000.0 - 1000.0))


def test_sar_ratio_guards_against_division_by_zero():
    t1 = _image({"VV": np.zeros((4, 4), dtype=np.float32)}, "sar")
    t2 = _image({"VV": np.full((4, 4), 5.0, dtype=np.float32)}, "sar")

    result = detect_change(t1, t2)

    assert not np.isinf(result.change_map).any()


# --------------------------------------------------------------------------
# Optical: difference the indices, not the raw DNs
# --------------------------------------------------------------------------


def test_optical_differences_the_index_and_names_it():
    red1 = _bimodal((16, 16), 0.05, 0.30)
    t1 = _image({"B04": red1, "B08": np.full((16, 16), 0.45, dtype=np.float32)}, "optical")
    t2 = _image(
        {"B04": np.full((16, 16), 0.05, dtype=np.float32),
         "B08": np.full((16, 16), 0.45, dtype=np.float32)},
        "optical",
    )

    result = detect_change(t1, t2, index="ndvi")

    assert result.index_name == "ndvi"
    assert result.operator == "index_difference"
    assert result.threshold is not None


def test_identical_optical_images_detect_no_change():
    bands = {
        "B04": np.full((16, 16), 0.10, dtype=np.float32),
        "B08": np.full((16, 16), 0.40, dtype=np.float32),
    }
    image = _image(bands, "optical")

    result = detect_change(image, image, index="ndvi")

    assert result.mask.sum() == 0
    assert result.threshold is None


def test_nodata_pixels_are_excluded_from_the_mask():
    a = _bimodal((16, 16), 0.05, 0.40)
    b = np.full((16, 16), 0.05, dtype=np.float32)
    nir = np.full((16, 16), 0.45, dtype=np.float32)
    nir[0, 0] = np.nan

    t1 = _image({"B04": a, "B08": nir}, "optical")
    t2 = _image({"B04": b, "B08": np.full((16, 16), 0.45, dtype=np.float32)}, "optical")

    result = detect_change(t1, t2, index="ndvi")

    assert not result.mask[0, 0]
    assert not result.valid_mask[0, 0]


def test_mismatched_shapes_raise():
    t1 = _image({"B04": np.zeros((8, 8), np.float32), "B08": np.zeros((8, 8), np.float32)}, "optical")
    t2 = _image({"B04": np.zeros((4, 4), np.float32), "B08": np.zeros((4, 4), np.float32)}, "optical")

    with pytest.raises(ValueError):
        detect_change(t1, t2, index="ndvi")


def test_modality_mismatch_between_dates_raises():
    """Comparing an optical scene against a SAR scene is not a change signal."""
    t1 = _image({"B04": np.zeros((8, 8), np.float32), "B08": np.zeros((8, 8), np.float32)}, "optical")
    t2 = _image({"VV": np.zeros((8, 8), np.float32)}, "sar")

    with pytest.raises(ValueError):
        detect_change(t1, t2)
