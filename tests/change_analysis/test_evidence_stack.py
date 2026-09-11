"""Tests for multi-index evidence stacking and region spectral signatures."""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.detector import detect_change_stack, index_stack
from tools.change_analysis.io import RSImage
from tools.change_analysis.regions import extract_regions


def _img(bands: dict, size=32):
    names = list(bands)
    stack = np.stack([np.full((size, size), v, dtype=np.float32) for v in bands.values()])
    return RSImage(array=stack, crs=None, transform=None, modality="optical",
                   band_names=names, gsd_m=None)


def _full_bands(size=32, **over):
    base = {"B03": 0.10, "B04": 0.10, "B08": 0.40, "B11": 0.20}
    base.update(over)
    return _img(base, size)


# -- stacking -------------------------------------------------------------


def test_stack_computes_every_index_the_bands_allow():
    stack = index_stack(_full_bands())
    assert set(stack) == {"ndvi", "ndwi", "ndbi"}


def test_stack_only_offers_what_bands_support():
    stack = index_stack(_img({"B04": 0.1, "B08": 0.4}))
    assert set(stack) == {"ndvi"}


def test_rgb_only_yields_an_empty_stack():
    assert index_stack(_img({"red": 0.1, "green": 0.1, "blue": 0.1})) == {}


# -- routing --------------------------------------------------------------


@pytest.mark.parametrize("target,expected", [
    ("water", "ndwi"),
    ("trees", "ndvi"),
    ("low_vegetation", "ndvi"),
    ("buildings", "ndbi"),
    ("NVG_surface", "ndbi"),
    ("playgrounds", "ndbi"),
])
def test_named_target_class_routes_to_its_diagnostic_index(target, expected):
    t1 = _full_bands()
    t2 = _full_bands(B08=0.20)
    result = detect_change_stack(t1, t2, target_class=target)
    assert result.index_name == expected
    assert result.operator == "index_difference_routed_by_class"


def test_unnamed_target_takes_max_magnitude_across_indices():
    t1 = _full_bands()
    t2 = _full_bands(B08=0.20)
    result = detect_change_stack(t1, t2)
    assert result.operator == "index_difference_max_across_stack"
    assert result.index_name is None


def test_max_across_stack_is_at_least_each_single_index():
    """Max-combining must dominate any one index, or it is not a max."""
    t1 = _full_bands()
    t2 = _full_bands(B08=0.20, B03=0.30)
    combined = detect_change_stack(t1, t2).change_map
    for name in ("ndvi", "ndwi", "ndbi"):
        single = detect_change_stack(t1, t2, index=name).change_map
        assert np.nanmax(combined) >= np.nanmax(single) - 1e-6


def test_target_class_with_no_supporting_index_falls_back_not_crashes():
    t1 = _img({"B04": 0.1, "B08": 0.4})
    t2 = _img({"B04": 0.1, "B08": 0.2})
    result = detect_change_stack(t1, t2, target_class="water")  # needs NDWI
    assert result.index_name == "ndvi"
    assert "fallback" in (result.note or "")


def test_stack_records_what_it_considered_in_the_trace():
    params = detect_change_stack(_full_bands(), _full_bands(B08=0.2)).as_trace_params()
    assert set(params["indices_available"]) == {"ndvi", "ndwi", "ndbi"}


# -- region spectral signature -------------------------------------------


def test_regions_carry_before_and_after_index_values():
    t1 = _full_bands()
    t2 = _full_bands(B08=0.20)
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:12] = True

    regions = extract_regions(
        mask, t1, index_stacks=(index_stack(t1), index_stack(t2))
    )

    sig1 = regions[0].signature_t1
    sig2 = regions[0].signature_t2
    assert set(sig1) == {"ndvi", "ndwi", "ndbi"}
    # NIR fell 0.40 -> 0.20, so NDVI must drop and NDWI must rise.
    assert sig2["ndvi"] < sig1["ndvi"]
    assert sig2["ndwi"] > sig1["ndwi"]


def test_signature_is_none_when_no_stack_supplied():
    """Existing callers must keep working unchanged."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:12] = True
    region = extract_regions(mask, _full_bands())[0]

    assert region.signature_t1 is None
    assert region.signature_t2 is None
    assert "signature_t1" in region.to_dict()
