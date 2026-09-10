import numpy as np
import pytest

from tools.change_analysis.io import RSImage
from tools.change_analysis.pair import validate_pair


def make_image(
    crs="EPSG:32643",
    gsd_m=2.0,
    shape=(4, 512, 512),
    modality="optical",
):
    array = np.zeros(shape, dtype=np.float32)

    return RSImage(
        array=array,
        crs=crs,
        transform=None,
        modality=modality,
        band_names=None,
        gsd_m=gsd_m,
    )


def test_valid_pair():
    t1 = make_image()
    t2 = make_image()

    result = validate_pair(t1, t2)

    assert result.valid is True
    assert result.errors == []


def test_rejects_different_crs():
    t1 = make_image(crs="EPSG:32643")
    t2 = make_image(crs="EPSG:32644")

    result = validate_pair(t1, t2)

    assert result.valid is False
    assert any("CRS" in error for error in result.errors)


def test_rejects_different_dimensions():
    t1 = make_image(shape=(4, 512, 512))
    t2 = make_image(shape=(4, 256, 256))

    result = validate_pair(t1, t2)

    assert result.valid is False
    assert any("dimension" in error.lower() for error in result.errors)


def test_rejects_different_gsd():
    t1 = make_image(gsd_m=2.0)
    t2 = make_image(gsd_m=10.0)

    result = validate_pair(t1, t2)

    assert result.valid is False
    assert any("GSD" in error for error in result.errors)


def test_rejects_different_band_count():
    t1 = make_image(shape=(4, 512, 512))
    t2 = make_image(shape=(3, 512, 512))

    result = validate_pair(t1, t2)

    assert result.valid is False
    assert any("band" in error.lower() for error in result.errors)