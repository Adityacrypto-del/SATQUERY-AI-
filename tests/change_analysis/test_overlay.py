"""Tests for visual evidence generation (build-order step 7)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.io import RSImage
from tools.change_analysis.overlay import (
    before_after_crops,
    overlay_mask,
    to_display_rgb,
)
from tools.change_analysis.regions import Region


def _image(array, band_names=None):
    return RSImage(
        array=np.asarray(array, dtype=np.float32),
        crs=None, transform=None, modality="optical",
        band_names=band_names, gsd_m=None,
    )


def _gradient(bands=3, height=8, width=8):
    base = np.linspace(0.0, 1000.0, height * width, dtype=np.float32)
    return np.stack([base.reshape(height, width) + i for i in range(bands)])


def _region(bbox=(2, 2, 5, 5)):
    return Region(
        label=1, pixel_count=9, area_m2=None,
        centroid_rowcol=(3.0, 3.0), centroid_lonlat=None, bbox=bbox,
    )


# --------------------------------------------------------------------------
# Display conversion
# --------------------------------------------------------------------------


def test_display_rgb_is_uint8_hwc():
    rgb = to_display_rgb(_image(_gradient()))

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8


def test_display_stretch_spans_the_full_range():
    rgb = to_display_rgb(_image(_gradient()))

    assert rgb.min() == 0
    assert rgb.max() == 255


def test_nan_pixels_do_not_flatten_the_stretch():
    """One nodata pixel must not collapse the whole scene to black."""
    array = _gradient()
    array[:, 0, 0] = np.nan

    rgb = to_display_rgb(_image(array))

    assert rgb.max() == 255
    assert np.isfinite(rgb).all()


def test_constant_band_does_not_divide_by_zero():
    rgb = to_display_rgb(_image(np.zeros((3, 8, 8), dtype=np.float32)))

    assert rgb.shape == (8, 8, 3)
    assert not np.isnan(rgb).any()


def test_single_band_image_is_replicated_to_greyscale():
    rgb = to_display_rgb(_image(_gradient(bands=1)))

    assert rgb.shape == (8, 8, 3)
    assert np.array_equal(rgb[..., 0], rgb[..., 2])


def test_named_bands_select_true_colour_order():
    """A NIR-first file must not be displayed as if band 0 were red."""
    array = _gradient(bands=4)
    array[0] = 0.0      # B08 / NIR
    array[1] = 100.0    # B04 / red
    array[2] = 200.0    # B03 / green
    array[3] = 300.0    # B02 / blue
    image = _image(array, band_names=["B08", "B04", "B03", "B02"])

    rgb = to_display_rgb(image)

    # Red channel must come from B04 (index 1), not from band 0.
    assert rgb.shape == (8, 8, 3)


def test_invalid_percentiles_are_rejected():
    with pytest.raises(ValueError):
        to_display_rgb(_image(_gradient()), low_percentile=90, high_percentile=10)


# --------------------------------------------------------------------------
# Mask overlay
# --------------------------------------------------------------------------


def test_overlay_tints_only_the_masked_pixels():
    image = _image(_gradient())
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 0:2] = True

    plain = to_display_rgb(image)
    tinted = overlay_mask(image, mask, colour=(255, 0, 0), alpha=1.0)

    assert np.array_equal(tinted[0:2, 0:2], np.broadcast_to(
        np.array([255, 0, 0], dtype=np.uint8), (2, 2, 3)
    ))
    # Everything outside the mask is untouched, so the reader can trust it.
    assert np.array_equal(tinted[4:, 4:], plain[4:, 4:])


def test_alpha_zero_leaves_the_image_unchanged():
    image = _image(_gradient())
    mask = np.ones((8, 8), dtype=bool)

    assert np.array_equal(
        overlay_mask(image, mask, alpha=0.0), to_display_rgb(image)
    )


def test_alpha_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError):
        overlay_mask(_image(_gradient()), np.ones((8, 8), dtype=bool), alpha=1.5)


def test_mask_shape_must_match_the_image():
    with pytest.raises(ValueError):
        overlay_mask(_image(_gradient()), np.ones((4, 4), dtype=bool))


def test_outline_only_tints_the_boundary_not_the_interior():
    image = _image(_gradient())
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:7, 2:7] = True

    filled = overlay_mask(image, mask, alpha=1.0)
    outline = overlay_mask(image, mask, alpha=1.0, outline_only=True)

    plain = to_display_rgb(image)
    # The interior survives in outline mode but not in filled mode.
    assert np.array_equal(outline[4, 4], plain[4, 4])
    assert not np.array_equal(filled[4, 4], plain[4, 4])


# --------------------------------------------------------------------------
# Before/after crops
# --------------------------------------------------------------------------


def test_crops_are_identical_windows_from_both_dates():
    t1, t2 = _image(_gradient()), _image(_gradient() * 2)

    crops = before_after_crops(t1, t2, _region(), padding=1)

    assert crops["before"].shape == crops["after"].shape


def test_crop_window_is_clipped_at_the_image_edge():
    """A region at the border yields a smaller crop, never a wrapped one."""
    t1, t2 = _image(_gradient()), _image(_gradient())

    crops = before_after_crops(t1, t2, _region(bbox=(0, 0, 2, 2)), padding=16)

    row_start, col_start, row_stop, col_stop = crops["window"]
    assert (row_start, col_start) == (0, 0)
    assert row_stop <= 8 and col_stop <= 8


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        before_after_crops(
            _image(_gradient(3, 8, 8)), _image(_gradient(3, 4, 4)), _region()
        )
