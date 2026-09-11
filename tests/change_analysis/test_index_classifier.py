"""Producer B unit tests (originally from aditya/producer-b).

Migrated to the IndexClassification result object. classify_pair returned a
bare (s_t1, s_t2, trace) tuple; it now returns a structured result, because a
SAR pair has no semantic maps to put in those slots and zero-filled arrays
would assert "nothing changed" rather than "cannot say". Each test keeps its
original intent; only the accessor changed.

The contract tests for the four review defects live in
test_index_classifier_contract.py.
"""

import numpy as np
import pytest

from tools.change_analysis.io import RSImage
from tools.change_analysis.index_classifier import (
    classify_pair,
)


def make_optical_image(
    band_names=None,
    values=None,
):
    if band_names is None:
        band_names = ["red", "green", "blue", "nir", "swir"]

    if values is None:
        values = np.ones(
            (len(band_names), 8, 8),
            dtype=np.float32,
        )

    return RSImage(
        array=values,
        crs=None,
        transform=None,
        modality="optical",
        band_names=band_names,
        gsd_m=None,
    )


def make_sar_image(values):
    return RSImage(
        array=values.astype(np.float32),
        crs=None,
        transform=None,
        modality="sar",
        band_names=["sar"],
        gsd_m=None,
    )


def test_output_shape_and_dtype():
    t1 = make_optical_image()
    t2 = make_optical_image()

    result = classify_pair(t1, t2)
    s_t1, s_t2 = result.s_t1, result.s_t2

    assert s_t1.shape == (8, 8)
    assert s_t2.shape == (8, 8)

    assert np.issubdtype(s_t1.dtype, np.integer)
    assert np.issubdtype(s_t2.dtype, np.integer)

    assert np.all((s_t1 >= 0) & (s_t1 <= 6))
    assert np.all((s_t2 >= 0) & (s_t2 <= 6))


def test_missing_swir_skips_ndbi():
    band_names = ["red", "green", "blue", "nir"]

    t1 = make_optical_image(band_names=band_names)
    t2 = make_optical_image(band_names=band_names)

    trace = classify_pair(t1, t2).trace

    assert "ndvi" in trace["indices"]
    assert "ndwi" in trace["indices"]
    assert "ndbi" not in trace["indices"]


def test_missing_nir_skips_ndvi_and_ndwi():
    band_names = ["red", "green", "blue", "swir"]

    t1 = make_optical_image(band_names=band_names)
    t2 = make_optical_image(band_names=band_names)

    trace = classify_pair(t1, t2).trace

    assert "ndvi" not in trace["indices"]
    assert "ndwi" not in trace["indices"]
    assert "ndbi" not in trace["indices"]


def test_sar_only_has_no_semantic_class_assignment():
    values = np.ones(
        (1, 8, 8),
        dtype=np.float32,
    )

    t1 = make_sar_image(values)
    t2 = make_sar_image(values * 2)

    result = classify_pair(t1, t2)
    trace = result.trace

    # Was: all-zero maps. Zero means "unchanged" to every downstream rule,
    # so zero-filling here asserted that nothing changed on a pair where
    # change was in fact detected. The maps are now absent instead.
    assert result.semantic_available is False
    assert result.s_t1 is None
    assert result.s_t2 is None
    with pytest.raises(ValueError):
        result.require_semantic()

    assert trace["producer"] == "index_classifier"
    assert trace["note"] == (
        "physics-based approximation, not a trained classifier"
    )


def test_playgrounds_are_never_predicted():
    t1_values = np.zeros(
        (5, 8, 8),
        dtype=np.float32,
    )

    t2_values = np.zeros(
        (5, 8, 8),
        dtype=np.float32,
    )

    # Strong red signature intended to make sure no rule
    # accidentally maps anything to playgrounds.
    t1_values[0] = 1.0
    t2_values[0] = 1.0

    t1 = make_optical_image(values=t1_values)
    t2 = make_optical_image(values=t2_values)

    result = classify_pair(t1, t2)
    s_t1, s_t2 = result.s_t1, result.s_t2

    assert not np.any(s_t1 == 6)
    assert not np.any(s_t2 == 6)


def test_confidence_is_below_producer_a_reference():
    t1 = make_optical_image()
    t2 = make_optical_image()

    result = classify_pair(t1, t2)

    assert result.confidence < 0.90


def test_identical_pair_returns_all_zero_maps():
    values = np.zeros(
        (5, 8, 8),
        dtype=np.float32,
    )

    values[0] = 0.2
    values[1] = 0.3
    values[2] = 0.1
    values[3] = 0.8
    values[4] = 0.4

    t1 = make_optical_image(values=values)
    t2 = make_optical_image(values=values.copy())

    result = classify_pair(t1, t2)
    s_t1, s_t2 = result.s_t1, result.s_t2

    assert np.all(s_t1 == 0)
    assert np.all(s_t2 == 0)


def test_trace_records_available_indices():
    t1 = make_optical_image()
    t2 = make_optical_image()

    trace = classify_pair(t1, t2).trace

    assert trace["producer"] == "index_classifier"
    assert "indices" in trace
    assert isinstance(trace["indices"], list)


def test_ndvi_classifies_elevated_vegetation():
    band_names = ["red", "green", "blue", "nir"]

    t1_values = np.zeros((4, 8, 8), dtype=np.float32)
    t2_values = np.zeros((4, 8, 8), dtype=np.float32)

    # Background: low NDVI.
    t1_values[0] = 0.8  # red
    t1_values[3] = 0.2  # nir

    # Changed region: two vegetation levels.
    t1_values[0, :2, :4] = 0.1
    t1_values[3, :2, :4] = 0.9

    t1_values[0, 2:4, :4] = 0.2
    t1_values[3, 2:4, :4] = 0.8

    t2_values[:] = t1_values

    t1 = make_optical_image(
        band_names=band_names,
        values=t1_values,
    )
    t2 = make_optical_image(
        band_names=band_names,
        values=t2_values,
    )

    trace = classify_pair(t1, t2).trace

    assert "ndvi" in trace["indices"]
    assert trace["t1"]["class_splits"]["ndvi_trees_vs_low_vegetation"] is not None


def test_ndwi_classifies_elevated_water():
    band_names = ["green", "red", "blue", "nir"]

    t1_values = np.zeros((4, 8, 8), dtype=np.float32)
    t2_values = np.zeros((4, 8, 8), dtype=np.float32)

    # Background: low NDWI.
    t1_values[0] = 0.2  # green
    t1_values[3] = 0.8  # nir

    # Changed region: high NDWI.
    t1_values[0, :4, :4] = 0.9
    t1_values[3, :4, :4] = 0.1

    t2_values[:] = t1_values

    t1 = make_optical_image(
        band_names=band_names,
        values=t1_values,
    )
    t2 = make_optical_image(
        band_names=band_names,
        values=t2_values,
    )

    trace = classify_pair(t1, t2).trace

    assert "ndwi" in trace["indices"]
    assert trace["t1"]["thresholds"]["ndwi"] is not None


def test_trace_records_ambiguous_index_overlap():
    band_names = ["red", "green", "blue", "nir", "swir"]

    t1_values = np.zeros((5, 8, 8), dtype=np.float32)
    t2_values = np.zeros((5, 8, 8), dtype=np.float32)

    # Background: moderate values that yield low index scores.
    t1_values[0] = 0.5  # red
    t1_values[1] = 0.5  # green
    t1_values[2] = 0.5  # blue
    t1_values[3] = 0.5  # nir
    t1_values[4] = 0.5  # swir

    # Overlap region (top-left quadrant):
    # High NIR → high NDVI  (NIR >> RED)
    # High SWIR → high NDBI (SWIR >> NIR in a relative sense)
    #
    # NDVI = (nir - red)/(nir + red) = (0.9 - 0.1)/(0.9 + 0.1) = 0.8
    # NDBI = (swir - nir)/(swir + nir) = (0.95 - 0.9)/(0.95 + 0.9) ≈ 0.027
    #
    # We need NDBI to also be elevated. Use very high SWIR with
    # lower NIR so both NDVI and NDBI are above Otsu thresholds.
    #
    # Adjusted: NIR=0.6, RED=0.05 → NDVI ≈ 0.846
    #           SWIR=0.95, NIR=0.6 → NDBI ≈ 0.226
    # Background: NIR=0.5, RED=0.5 → NDVI = 0.0
    #             SWIR=0.5, NIR=0.5 → NDBI = 0.0
    # Otsu will split around these, making both elevated in the region.
    t1_values[0, :4, :4] = 0.05  # red  → pushes NDVI high
    t1_values[3, :4, :4] = 0.6   # nir
    t1_values[4, :4, :4] = 0.95  # swir → pushes NDBI high

    t2_values[:] = t1_values

    t1 = make_optical_image(
        band_names=band_names,
        values=t1_values,
    )
    t2 = make_optical_image(
        band_names=band_names,
        values=t2_values,
    )

    trace = classify_pair(t1, t2).trace

    # At least one date must record the overlap ambiguity.
    ambiguity_t1 = trace["t1"]["ambiguity"]
    ambiguity_t2 = trace["t2"]["ambiguity"]

    all_ambiguity = ambiguity_t1 + ambiguity_t2

    has_overlap_note = any(
        "multiple" in entry.lower() for entry in all_ambiguity
    )

    assert has_overlap_note, (
        f"Expected an ambiguity entry mentioning 'multiple', "
        f"got: {all_ambiguity}"
    )

    assert trace["t1"]["precedence"] == ["ndwi", "ndbi", "ndvi"]


def test_confidence_is_marked_as_conservative():
    t1 = make_optical_image()
    t2 = make_optical_image()

    result = classify_pair(t1, t2)

    assert isinstance(result.confidence, (int, float))
    assert 0 <= result.confidence <= 1
    # Was 0.50 with confidence_type "conservative_heuristic". A producer that
    # has never been scored has no accuracy to report, so it reports zero and
    # says why -- the same measured-or-zero rule the pipeline follows.
    assert result.confidence == 0.0
    assert "not calibrated" in result.confidence_basis

    assert result.trace["producer"] == "index_classifier"
    assert "physics-based approximation" in result.trace["note"]
