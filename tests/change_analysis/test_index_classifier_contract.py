"""Acceptance tests for Producer B's four review defects.

Producer B is the training-free path for inputs the SECOND-trained model has
no coverage for: multispectral with arbitrary band counts, and SAR. That is
the path the ISRO/SAC evaluation data takes, so its failure modes matter more
than its accuracy.

Each test here corresponds to a defect found by running the merged producer
against this branch's modules. They are contract tests rather than unit
tests: they assert the invariants this branch enforces everywhere else --
measured-or-zero confidence, no fabricated values, no bare tuples, and class 0
meaning "unchanged" rather than "unknown".
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.index_classifier import classify_pair
from tools.change_analysis.io import RSImage

BANDS = ["red", "green", "blue", "nir", "swir"]


def _optical(red, green, blue, nir, swir):
    return RSImage(
        array=np.stack([red, green, blue, nir, swir]).astype(np.float32),
        crs=None, transform=None, modality="optical",
        band_names=list(BANDS), gsd_m=None,
    )


def _flat(value, size=16):
    return np.full((size, size), value, dtype=np.float32)


def _sar(plane):
    return RSImage(
        array=np.asarray(plane, dtype=np.float32)[None], crs=None, transform=None,
        modality="sar", band_names=["vv"], gsd_m=None,
    )


# --------------------------------------------------------------------------
# B1 -- a SAR pair must not assert "nothing changed"
# --------------------------------------------------------------------------


def test_sar_returns_no_semantic_maps_rather_than_empty_ones():
    """All-zero maps are a claim that nothing changed. The producer detects
    SAR change but cannot name land-cover classes for it, so the honest
    output is "no semantic maps", not maps full of zeros."""
    base = _flat(0.2)
    moved = base.copy()
    moved[4:12, 4:12] = 0.9

    result = classify_pair(_sar(base), _sar(moved))

    assert result.semantic_available is False
    assert result.s_t1 is None and result.s_t2 is None
    # The binary change it *can* measure is still reported.
    assert result.change_mask is not None
    assert int(result.change_mask.sum()) > 0


def test_the_rules_cannot_be_fed_a_sar_result_by_accident():
    """The failure this prevents: zero-filled maps reaching the CDVQA rules,
    which answer "no" and "0" with full confidence while the producer's own
    trace records detected change."""
    base = _flat(0.2)
    moved = base.copy()
    moved[4:12, 4:12] = 0.9

    result = classify_pair(_sar(base), _sar(moved))

    with pytest.raises(ValueError, match="semantic"):
        result.require_semantic()


# --------------------------------------------------------------------------
# B2 -- declared precedence must be the precedence actually applied
# --------------------------------------------------------------------------


def test_water_wins_over_built_up_where_both_indices_fire():
    """NDWI outranks NDBI, as the trace has always declared. Reporting open
    water as built-up is a headline failure: "identify built-up and
    water-covered regions" is one of the problem statement's own queries."""
    red, green, blue, nir, swir = (_flat(0.10) for _ in range(5))
    green[4:8, :] = 0.90   # strong NDWI
    nir[4:8, :] = 0.05
    swir[4:8, :] = 0.90    # strong NDBI on the same pixels
    nir[8:12, :] = 0.80    # unambiguous vegetation elsewhere

    image = _optical(red, green, blue, nir, swir)
    result = classify_pair(image, image)

    assert set(np.unique(result.classes_t1[4:8, :]).tolist()) == {5}, "water must win"


def test_declared_precedence_is_derived_from_the_write_order():
    """The original defect was a comment claiming one precedence while the
    code applied another. They cannot drift if there is one source."""
    from tools.change_analysis.index_classifier import PRECEDENCE, WRITE_ORDER

    assert list(PRECEDENCE) == list(reversed(WRITE_ORDER))


# --------------------------------------------------------------------------
# B3 -- no land cover where the physics says there is none
# --------------------------------------------------------------------------


def test_no_vegetation_is_invented_in_a_scene_with_none():
    """Otsu splits whatever distribution it is handed, so on bare ground it
    manufactures a vegetation class from the upper half of a negative NDVI
    range. A normalised difference at or below zero means the numerator band
    does not exceed the denominator band -- NIR not above red is definitionally
    not vegetation, whatever the relative ranking says."""
    rng = np.random.default_rng(0)

    def noise():
        return rng.normal(0, 0.01, (16, 16)).astype(np.float32)

    # NDVI = (nir - red)/(nir + red) = -0.14  -> no vegetation
    # NDWI = (green - nir)/(green + nir) = -0.09 -> no water
    # NDBI = (swir - nir)/(swir + nir) = +0.20 -> built-up genuinely present
    red = _flat(0.40) + noise()
    nir = _flat(0.30) + noise()
    green = _flat(0.25) + noise()
    swir = _flat(0.45) + noise()
    blue = _flat(0.30)

    image = _optical(red, green, blue, nir, swir)
    result = classify_pair(image, image)
    classes = result.classes_t1

    assert int(((classes == 2) | (classes == 3)).sum()) == 0, "no vegetation"
    assert int((classes == 5).sum()) == 0, "no water"


def test_a_scene_with_real_vegetation_still_finds_it():
    """The guard must not be a blanket refusal -- the contrast case."""
    red, green, blue, nir, swir = (_flat(0.10) for _ in range(5))
    nir[:8, :] = 0.80  # NDVI strongly positive on the top half

    image = _optical(red, green, blue, nir, swir)
    result = classify_pair(image, image)

    assert int(((result.classes_t1 == 2) | (result.classes_t1 == 3)).sum()) > 0


# --------------------------------------------------------------------------
# B4/B5 -- contract invariants shared with the rest of the branch
# --------------------------------------------------------------------------


def test_confidence_is_zero_with_an_explicit_basis_never_a_plausible_number():
    """The producer has never been scored, so it has no measured accuracy to
    report. 0.50 was a number chosen to look reasonable -- the same failure as
    the co-registration penalty removed from pipeline.py."""
    image = _optical(*(_flat(0.2) for _ in range(5)))

    result = classify_pair(image, image)

    assert result.confidence == 0.0
    assert "not" in result.confidence_basis.lower()
    assert "0.5" not in result.confidence_basis


def test_result_is_a_structured_object_not_a_bare_tuple():
    image = _optical(*(_flat(0.2) for _ in range(5)))

    result = classify_pair(image, image)

    assert not isinstance(result, tuple)
    for name in ("s_t1", "s_t2", "trace", "confidence", "semantic_available"):
        assert hasattr(result, name)


def test_unchanged_pixels_are_zero_in_both_maps():
    """SECOND semantics: a pixel carries a class only where it changed, and
    the two maps agree exactly on which pixels those are."""
    red, green, blue, nir, swir = (_flat(0.10) for _ in range(5))
    nir[:8, :] = 0.80
    t1 = _optical(red, green, blue, nir, swir)
    swir2 = swir.copy()
    swir2[:8, :] = 0.95   # vegetation becomes built-up
    t2 = _optical(red, green, blue, nir, swir2)

    result = classify_pair(t1, t2)

    assert np.array_equal(result.s_t1 == 0, result.s_t2 == 0)


def test_semantic_maps_stay_inside_the_second_class_range():
    rng = np.random.default_rng(1)
    planes = [rng.uniform(0, 1, (16, 16)).astype(np.float32) for _ in range(5)]
    image = _optical(*planes)

    result = classify_pair(image, image)

    for semantic in (result.s_t1, result.s_t2):
        assert semantic.min() >= 0 and semantic.max() <= 6
        assert int((semantic == 6).sum()) == 0, "playgrounds are never predicted"
