import numpy as np
import pytest

from tools.change_analysis.image_diff import (
    compute_difference,
    create_change_mask,
)
from tools.change_analysis.area import calculate_changed_area


def test_identical_images_have_zero_difference():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    difference = compute_difference(image, image)

    assert np.all(difference == 0)


def test_mismatched_images_raise_error():
    image_t1 = np.zeros((10, 10, 3), dtype=np.uint8)
    image_t2 = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        compute_difference(image_t1, image_t2)


def test_change_mask():
    image_t1 = np.zeros((10, 10, 3), dtype=np.uint8)
    image_t2 = np.zeros((10, 10, 3), dtype=np.uint8)

    image_t2[:5, :5] = 255

    mask = create_change_mask(
        image_t1,
        image_t2,
        threshold=30,
    )

    assert mask.sum() == 25


def test_changed_area():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :5] = True

    result = calculate_changed_area(
        mask,
        pixel_area_m2=2.0,
    )

    assert result["changed_pixels"] == 25
    assert result["area_m2"] == 50.0


def test_invalid_pixel_area():
    mask = np.ones((5, 5), dtype=bool)

    with pytest.raises(ValueError):
        calculate_changed_area(mask, 0)
