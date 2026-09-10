"""Tests for spectral indices (build-order step 4).

Ten tests, aimed at the failure modes that actually bite rather than at
formula restatement: divide-by-zero, NaN propagation, band resolution, and
orientation (a sign error in NDWI is invisible until it inverts an answer).
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.indices import (
    available_indices,
    ndbi,
    ndvi,
    ndwi,
    normalized_difference,
)
from tools.change_analysis.io import RSImage


def _image(bands: dict, modality: str = "optical") -> RSImage:
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


def _uniform(value: float, shape=(2, 2)) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


# --------------------------------------------------------------------------
# Core operator
# --------------------------------------------------------------------------


def test_normalized_difference_matches_the_definition():
    a = np.array([[0.6]], dtype=np.float32)
    b = np.array([[0.2]], dtype=np.float32)

    assert normalized_difference(a, b)[0, 0] == pytest.approx(0.5)


def test_zero_denominator_is_nan_not_inf():
    """(a-b)/(a+b) with both bands zero is undefined, and inf would survive
    Otsu as a finite-looking extreme."""
    zeros = np.zeros((2, 2), dtype=np.float32)

    out = normalized_difference(zeros, zeros)

    assert np.isnan(out).all()
    assert not np.isinf(out).any()


def test_opposite_signed_bands_summing_to_zero_are_nan():
    a = np.array([[5.0]], dtype=np.float32)
    b = np.array([[-5.0]], dtype=np.float32)

    assert np.isnan(normalized_difference(a, b)[0, 0])


def test_nan_input_propagates_without_raising():
    a = np.array([[np.nan, 0.6]], dtype=np.float32)
    b = np.array([[0.2, 0.2]], dtype=np.float32)

    out = normalized_difference(a, b)

    assert np.isnan(out[0, 0])
    assert out[0, 1] == pytest.approx(0.5)


def test_valid_output_stays_within_minus_one_to_one():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
    b = rng.uniform(0.0, 1.0, (32, 32)).astype(np.float32)

    out = normalized_difference(a, b)
    finite = out[np.isfinite(out)]

    assert finite.min() >= -1.0
    assert finite.max() <= 1.0


# --------------------------------------------------------------------------
# Orientation -- a sign error here silently inverts every answer downstream
# --------------------------------------------------------------------------


def test_ndvi_is_positive_over_vegetation():
    image = _image({"B04": _uniform(0.05), "B08": _uniform(0.45)})

    assert ndvi(image)[0, 0] == pytest.approx(0.8)


def test_ndwi_is_positive_over_water_and_ndvi_is_not():
    """Water: high green, low NIR. The two indices must disagree in sign."""
    image = _image(
        {"B03": _uniform(0.30), "B04": _uniform(0.10), "B08": _uniform(0.05)}
    )

    assert ndwi(image)[0, 0] > 0
    assert ndvi(image)[0, 0] < 0


def test_ndbi_is_positive_over_built_up():
    """Built-up: SWIR exceeds NIR."""
    image = _image({"B08": _uniform(0.20), "B11": _uniform(0.40)})

    assert ndbi(image)[0, 0] == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------
# Band resolution
# --------------------------------------------------------------------------


def test_plain_band_names_resolve_as_well_as_sentinel_codes():
    sentinel = _image({"B04": _uniform(0.05), "B08": _uniform(0.45)})
    plain = _image({"red": _uniform(0.05), "nir": _uniform(0.45)})

    assert ndvi(plain)[0, 0] == pytest.approx(ndvi(sentinel)[0, 0])


def test_missing_band_names_the_index_and_the_band():
    image = _image({"B03": _uniform(0.3), "B04": _uniform(0.1)})

    with pytest.raises(ValueError) as excinfo:
        ndvi(image)

    message = str(excinfo.value).lower()
    assert "ndvi" in message and "nir" in message


def test_available_indices_reports_only_what_the_bands_support():
    image = _image({"B03": _uniform(0.3), "B04": _uniform(0.1), "B08": _uniform(0.4)})

    available = available_indices(image)

    assert "ndvi" in available and "ndwi" in available
    # No SWIR band, so NDBI is not computable and must not be offered.
    assert "ndbi" not in available


def test_sar_modality_offers_no_optical_indices():
    """Indices are optical-only; SAR goes through the ratio operator."""
    image = _image({"VV": _uniform(0.4)}, modality="sar")

    assert available_indices(image) == []
