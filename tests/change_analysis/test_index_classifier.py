import numpy as np

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

    s_t1, s_t2, trace = classify_pair(t1, t2)

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

    _, _, trace = classify_pair(t1, t2)

    assert "ndvi" in trace["indices"]
    assert "ndwi" in trace["indices"]
    assert "ndbi" not in trace["indices"]


def test_missing_nir_skips_ndvi_and_ndwi():
    band_names = ["red", "green", "blue", "swir"]

    t1 = make_optical_image(band_names=band_names)
    t2 = make_optical_image(band_names=band_names)

    _, _, trace = classify_pair(t1, t2)

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

    s_t1, s_t2, trace = classify_pair(t1, t2)

    assert np.all(s_t1 == 0)
    assert np.all(s_t2 == 0)

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

    s_t1, s_t2, _ = classify_pair(t1, t2)

    assert not np.any(s_t1 == 6)
    assert not np.any(s_t2 == 6)


def test_confidence_is_below_producer_a_reference():
    t1 = make_optical_image()
    t2 = make_optical_image()

    _, _, trace = classify_pair(t1, t2)

    assert trace["confidence"] < 0.90


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

    s_t1, s_t2, _ = classify_pair(t1, t2)

    assert np.all(s_t1 == 0)
    assert np.all(s_t2 == 0)


def test_trace_records_available_indices():
    t1 = make_optical_image()
    t2 = make_optical_image()

    _, _, trace = classify_pair(t1, t2)

    assert trace["producer"] == "index_classifier"
    assert "indices" in trace
    assert isinstance(trace["indices"], list)

    