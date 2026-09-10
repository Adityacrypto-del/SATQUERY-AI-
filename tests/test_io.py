import numpy as np
import pytest

from tools.change_analysis.io import RSImage


def test_rsimage_valid():
    array = np.zeros(
        (4, 512, 512),
        dtype=np.float32,
    )

    image = RSImage(
        array=array,
        crs=None,
        transform=None,
        modality="optical",
        band_names=["red", "green", "blue", "nir"],
        gsd_m=2.0,
    )

    assert image.array.shape == (4, 512, 512)
    assert image.array.dtype == np.float32
    assert image.modality == "optical"
    assert image.gsd_m == 2.0


def test_rsimage_requires_three_dimensions():
    array = np.zeros(
        (512, 512),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        RSImage(
            array=array,
            crs=None,
            transform=None,
            modality="optical",
            band_names=None,
            gsd_m=None,
        )


def test_rsimage_requires_float32():
    array = np.zeros(
        (4, 512, 512),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        RSImage(
            array=array,
            crs=None,
            transform=None,
            modality="optical",
            band_names=None,
            gsd_m=None,
        )


def test_rsimage_rejects_invalid_modality():
    array = np.zeros(
        (4, 512, 512),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        RSImage(
            array=array,
            crs=None,
            transform=None,
            modality="thermal",
            band_names=None,
            gsd_m=None,
        )