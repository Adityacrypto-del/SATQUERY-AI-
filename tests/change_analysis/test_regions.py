"""Tests for per-region attributes (build-order step 4)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from tools.change_analysis.io import RSImage
from tools.change_analysis.regions import extract_regions, summarise_regions


def _image(height=16, width=16, georeferenced=True, pixel=10.0):
    """An RSImage carrying just the georeferencing the region maths needs."""
    if georeferenced:
        transform = Affine(pixel, 0.0, 500000.0, 0.0, -pixel, 3000000.0)
        crs = CRS.from_epsg(32643)
        gsd = pixel
        size = (pixel, pixel)
    else:
        transform, crs, gsd, size = None, None, None, None
    return RSImage(
        array=np.zeros((1, height, width), dtype=np.float32),
        crs=crs,
        transform=transform,
        modality="optical",
        band_names=None,
        gsd_m=gsd,
        _pixel_size_m=size,
        _gsd_source="projected_crs_exact" if georeferenced else "unavailable",
    )


def _two_blobs(height=16, width=16):
    mask = np.zeros((height, width), dtype=bool)
    mask[2:4, 2:4] = True      # 4 pixels
    mask[8:13, 8:13] = True    # 25 pixels
    return mask


def test_regions_are_sorted_by_descending_size():
    regions = extract_regions(_two_blobs(), _image())

    assert [r.pixel_count for r in regions] == [25, 4]


def test_area_m2_comes_from_the_geotransform():
    """20 m pixels means 400 m2 each -- derived, never passed in."""
    regions = extract_regions(_two_blobs(), _image(pixel=20.0))

    assert regions[0].area_m2 == pytest.approx(25 * 400.0)


def test_ungeoreferenced_image_reports_pixels_but_not_metres():
    """CDVQA PNGs have no transform. A pixel count is still a measurement."""
    regions = extract_regions(_two_blobs(), _image(georeferenced=False))

    assert regions[0].pixel_count == 25
    assert regions[0].area_m2 is None
    assert regions[0].centroid_lonlat is None


def test_centroid_converts_to_lonlat_when_georeferenced():
    regions = extract_regions(_two_blobs(), _image())

    lonlat = regions[0].centroid_lonlat
    assert lonlat is not None
    lon, lat = lonlat
    assert 74.0 < lon < 76.0
    assert 26.0 < lat < 28.0


def test_bbox_bounds_the_region():
    regions = extract_regions(_two_blobs(), _image())

    row_min, col_min, row_max, col_max = regions[0].bbox
    assert (row_min, col_min) == (8, 8)
    assert (row_max, col_max) == (13, 13)


def test_min_pixels_drops_small_regions():
    regions = extract_regions(_two_blobs(), _image(), min_pixels=10)

    assert len(regions) == 1
    assert regions[0].pixel_count == 25


def test_empty_mask_yields_no_regions():
    assert extract_regions(np.zeros((16, 16), dtype=bool), _image()) == []


def test_mask_shape_must_match_the_image():
    with pytest.raises(ValueError):
        extract_regions(np.zeros((8, 8), dtype=bool), _image(16, 16))


def test_summary_aggregates_over_regions():
    image = _image()
    regions = extract_regions(_two_blobs(), image)

    summary = summarise_regions(regions, image)

    assert summary["n_regions"] == 2
    assert summary["total_changed_pixels"] == 29
    assert summary["total_area_m2"] == pytest.approx(29 * 100.0)
    assert summary["fraction_of_scene"] == pytest.approx(29 / 256.0)


def test_summary_area_is_none_without_georeferencing():
    """None, not 0.0 -- zero would read as 'no change detected'."""
    image = _image(georeferenced=False)
    regions = extract_regions(_two_blobs(), image)

    summary = summarise_regions(regions, image)

    assert summary["total_changed_pixels"] == 29
    assert summary["total_area_m2"] is None
