"""Tests for mask cleaning and connected-component labelling (step 4)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.morphology import (
    clean_mask,
    label_regions,
    remove_small_regions,
)


def test_opening_removes_isolated_speckle():
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 8] = True  # single-pixel speckle

    assert clean_mask(mask, opening_radius=1, closing_radius=0).sum() == 0


def test_opening_preserves_a_solid_blob():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True

    cleaned = clean_mask(mask, opening_radius=1, closing_radius=0)

    # A disk element is isotropic, so it rounds the square's four corners:
    # 64 pixels become 60. That is correct morphology, not a bug -- a square
    # is not open with respect to a disk. We use a disk deliberately, since a
    # square element would impose an axis-aligned bias on region shapes.
    # What matters is that the blob survives essentially intact and its
    # interior is untouched.
    assert cleaned.sum() == 60
    assert cleaned[5:11, 5:11].all()
    assert cleaned.sum() / mask.sum() > 0.9


def test_closing_fills_a_pinhole():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    mask[8, 8] = False  # one-pixel hole

    cleaned = clean_mask(mask, opening_radius=0, closing_radius=1)

    assert cleaned[8, 8]


def test_zero_radii_is_a_no_op():
    """An explicit 'do nothing' must not quietly alter the mask."""
    rng = np.random.default_rng(0)
    mask = rng.random((16, 16)) > 0.5

    assert np.array_equal(clean_mask(mask, 0, 0), mask)


def test_negative_radius_is_rejected():
    with pytest.raises(ValueError):
        clean_mask(np.zeros((4, 4), dtype=bool), opening_radius=-1)


def test_output_is_boolean_not_integer():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    assert clean_mask(mask, 1, 1).dtype == np.bool_


# --------------------------------------------------------------------------
# Connected components
# --------------------------------------------------------------------------


def test_two_separated_blobs_get_distinct_labels():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:14, 10:14] = True

    labels, count = label_regions(mask)

    assert count == 2
    assert set(np.unique(labels).tolist()) == {0, 1, 2}


def test_diagonal_touching_blobs_depend_on_connectivity():
    """4- vs 8-connectivity changes the region count, so it is explicit."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 2] = True
    mask[3, 3] = True

    assert label_regions(mask, connectivity=1)[1] == 2
    assert label_regions(mask, connectivity=2)[1] == 1


def test_empty_mask_yields_no_regions():
    labels, count = label_regions(np.zeros((8, 8), dtype=bool))

    assert count == 0
    assert labels.sum() == 0


def test_small_regions_are_removed_by_pixel_count():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:4, 2:4] = True     # 4 pixels
    mask[8:13, 8:13] = True   # 25 pixels

    pruned = remove_small_regions(mask, min_pixels=10)

    assert pruned.sum() == 25
    assert not pruned[2, 2]
