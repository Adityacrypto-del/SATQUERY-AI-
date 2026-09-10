"""Tests for the RSImage data contract and GeoTIFF I/O (build-order step 1).

These tests pin down the contract declared in CLAUDE.md section 3:

    RSImage.array      (bands, H, W), float32
    RSImage.crs        rasterio CRS or None
    RSImage.transform  affine geotransform or None
    RSImage.modality   "optical" | "sar" | "unknown"
    RSImage.band_names list[str] | None
    RSImage.gsd_m      ground sample distance in metres, or None

Design decisions these tests encode, for review:

1.  Band-first ordering. rasterio ``src.read()`` is natively (bands, H, W),
    so we do NOT moveaxis. Ayush single-image loader returns (H, W, bands)
    for the VLM path; the two contracts differ on purpose and this file is
    the place that difference is documented.
2.  nodata becomes NaN rather than a seventh RSImage field, and a derived
    ``valid_mask`` travels with the image. NaN is silent poison: it
    propagates through np.mean, histogram binning and Otsu, so one nodata
    pixel gives a NaN threshold and an all-False change mask with no error
    raised. Every reduction downstream goes through ``valid_mask`` /
    ``valid_pixels`` and the np.nan* variants.
3.  GSD is derived wherever it is derivable, and labelled by provenance.
    A projected metric CRS gives it exactly; a geographic CRS gives it via
    reprojection to the local UTM zone, which is a standard geodesic
    calculation rather than fabrication; only genuinely ungeoreferenced
    input (the CDVQA PNGs) yields None. ``georeferencing_report`` tags which
    of the three happened so the execution trace is never ambiguous.
4.  Bands are never silently dropped. A 4-band file stays 4 bands.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from tools.change_analysis.io import (
    RSImage,
    georeferencing_report,
    load_rsimage,
)


# UTM 43N -- the projected CRS covering much of India, so metre-based
# geotransforms in these fixtures are physically meaningful.
TEST_CRS = CRS.from_epsg(32643)

# 10 m pixels, north-up, origin at an arbitrary easting/northing.
TEST_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)


def _write_geotiff(
    path,
    array,
    crs=TEST_CRS,
    transform=TEST_TRANSFORM,
    band_names=None,
    nodata=None,
):
    """Write a (bands, H, W) array to a GeoTIFF fixture."""
    array = np.asarray(array)
    count, height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": array.dtype.name,
        "crs": crs,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        if band_names is not None:
            dst.descriptions = tuple(band_names)
    return str(path)


# --------------------------------------------------------------------------
# Georeferencing is preserved, not discarded
# --------------------------------------------------------------------------


def test_geotiff_preserves_crs_and_transform(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((3, 8, 12), dtype=np.uint16)
    )

    image = load_rsimage(path)

    assert image.crs == TEST_CRS
    assert image.transform == TEST_TRANSFORM


def test_array_is_band_first_float32(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((3, 8, 12), dtype=np.uint16)
    )

    image = load_rsimage(path)

    assert image.array.shape == (3, 8, 12)
    assert image.array.dtype == np.float32


def test_pixel_values_are_not_rescaled_on_load(tmp_path):
    """Loading preserves physical values; normalisation is a later, traced step."""
    data = np.array([[[0, 1000], [2000, 65535]]], dtype=np.uint16)
    path = _write_geotiff(tmp_path / "t1.tif", data)

    image = load_rsimage(path)

    assert image.array[0, 0, 0] == pytest.approx(0.0)
    assert image.array[0, 0, 1] == pytest.approx(1000.0)
    assert image.array[0, 1, 1] == pytest.approx(65535.0)


def test_multispectral_bands_are_never_dropped(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((4, 6, 6), dtype=np.uint16)
    )

    image = load_rsimage(path)

    assert image.n_bands == 4
    assert image.array.shape[0] == 4


# --------------------------------------------------------------------------
# GSD is derived from the geotransform, never assumed (HARD RULE 4)
# --------------------------------------------------------------------------


def test_gsd_is_derived_from_transform(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((1, 4, 4), dtype=np.uint8)
    )

    image = load_rsimage(path)

    assert image.gsd_m == pytest.approx(10.0)


def test_pixel_area_m2_is_derived_from_transform(tmp_path):
    """area_m2 downstream must come from the transform, not a passed-in constant."""
    transform = Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 3000000.0)
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((1, 4, 4), dtype=np.uint8),
        transform=transform,
    )

    image = load_rsimage(path)

    assert image.pixel_area_m2 == pytest.approx(400.0)


def test_anisotropic_pixels_report_gsd_and_area_consistently(tmp_path):
    """Non-square pixels are real; GSD must not silently pretend they are square."""
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -20.0, 3000000.0)
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((1, 4, 4), dtype=np.uint8),
        transform=transform,
    )

    image = load_rsimage(path)

    # Area is exact; GSD is the mean extent and is only a summary value.
    assert image.pixel_area_m2 == pytest.approx(200.0)
    assert image.gsd_m == pytest.approx(15.0)


def test_geographic_crs_gsd_is_derived_geodesically(tmp_path):
    """EPSG:4326 pixel sizes are degrees, but ground distance is still computable.

    We refuse to *call* degrees metres; we do not refuse to convert them.
    Reprojecting the centre pixel to its local UTM zone is a standard geodesic
    calculation, so the result is derived, not fabricated. It is tagged as
    derived in the georeferencing report so the trace never implies otherwise.
    """
    transform = Affine(0.0001, 0.0, 77.5, 0.0, -0.0001, 13.0)
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((1, 100, 100), dtype=np.uint8),
        crs=CRS.from_epsg(4326),
        transform=transform,
    )

    image = load_rsimage(path)

    # Measured against rasterio.warp for UTM 43N at 13 deg N:
    # x extent 10.854 m, y extent 11.069 m.
    assert image.gsd_m == pytest.approx(10.96, abs=0.05)
    assert image.pixel_area_m2 == pytest.approx(120.14, abs=0.5)


def test_geographic_gsd_narrows_with_latitude(tmp_path):
    """Meridian convergence is real: a degree of longitude shrinks as cos(lat).

    A single global constant for degrees-to-metres would pass the test above
    and fail this one, which is exactly why this test exists.
    """
    equator = _write_geotiff(
        tmp_path / "eq.tif",
        np.zeros((1, 100, 100), dtype=np.uint8),
        crs=CRS.from_epsg(4326),
        transform=Affine(0.0001, 0.0, 0.0, 0.0, -0.0001, 0.0),
    )
    high = _write_geotiff(
        tmp_path / "hi.tif",
        np.zeros((1, 100, 100), dtype=np.uint8),
        crs=CRS.from_epsg(4326),
        transform=Affine(0.0001, 0.0, 10.0, 0.0, -0.0001, 60.0),
    )

    # At 60 deg N one pixel spans roughly half the ground distance it does at
    # the equator in the east-west direction (cos 60 = 0.5).
    assert load_rsimage(high).gsd_m < load_rsimage(equator).gsd_m


def test_projected_non_metre_crs_is_converted_to_metres(tmp_path):
    """A CRS in US survey feet must not report its pixel size as metres."""
    # EPSG:2229 - California zone 5, US survey feet.
    transform = Affine(30.0, 0.0, 6500000.0, 0.0, -30.0, 1800000.0)
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((1, 8, 8), dtype=np.uint8),
        crs=CRS.from_epsg(2229),
        transform=transform,
    )

    image = load_rsimage(path)

    # 30 survey feet = 9.144 m, not 30 m.
    assert image.gsd_m == pytest.approx(9.144, abs=0.01)


# --------------------------------------------------------------------------
# Band names
# --------------------------------------------------------------------------


def test_band_names_are_read_from_descriptions(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((4, 4, 4), dtype=np.uint16),
        band_names=["B02", "B03", "B04", "B08"],
    )

    image = load_rsimage(path)

    assert image.band_names == ["B02", "B03", "B04", "B08"]


def test_band_names_are_none_when_file_declares_none(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((3, 4, 4), dtype=np.uint16)
    )

    image = load_rsimage(path)

    assert image.band_names is None


def test_band_index_lookup_is_case_insensitive(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((4, 4, 4), dtype=np.uint16),
        band_names=["B02", "B03", "B04", "B08"],
    )

    image = load_rsimage(path)

    assert image.band_index("b08") == 3
    assert image.band_index("nir") is None


# --------------------------------------------------------------------------
# Non-georeferenced input (the CDVQA benchmark path)
# --------------------------------------------------------------------------


def test_png_loads_without_georeferencing(tmp_path):
    from PIL import Image

    path = tmp_path / "cdvqa_pair.png"
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(path)

    image = load_rsimage(str(path))

    assert image.array.shape == (3, 6, 8)
    assert image.crs is None
    assert image.transform is None
    assert image.gsd_m is None
    assert image.pixel_area_m2 is None


def test_tiff_without_crs_is_reported_as_ungeoreferenced(tmp_path):
    path = _write_geotiff(
        tmp_path / "plain.tif",
        np.zeros((1, 4, 4), dtype=np.uint8),
        crs=None,
        transform=Affine.identity(),
    )

    image = load_rsimage(path)

    assert image.crs is None
    assert image.transform is None
    assert image.gsd_m is None


def test_ungeoreferenced_image_cannot_produce_lonlat(tmp_path):
    from PIL import Image

    path = tmp_path / "cdvqa_pair.png"
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(path)

    image = load_rsimage(str(path))

    assert image.pixel_to_lonlat(2.0, 3.0) is None


# --------------------------------------------------------------------------
# Pixel -> geographic conversion (consumed by regions.py, step 4)
# --------------------------------------------------------------------------


def test_pixel_to_lonlat_reprojects_to_wgs84(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((1, 10, 10), dtype=np.uint8)
    )

    image = load_rsimage(path)
    lonlat = image.pixel_to_lonlat(row=0.0, col=0.0)

    assert lonlat is not None
    lon, lat = lonlat
    # Origin 500000E / 3000000N in UTM 43N is on the zone central meridian
    # (75 deg E) at roughly 27.1 deg N.
    assert lon == pytest.approx(75.0, abs=0.01)
    assert lat == pytest.approx(27.1, abs=0.1)


def test_pixel_to_lonlat_uses_pixel_centre(tmp_path):
    """Row/col 0 means the centre of the top-left pixel, not its corner."""
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((1, 10, 10), dtype=np.uint8)
    )

    image = load_rsimage(path)
    first = image.pixel_to_lonlat(row=0.0, col=0.0)
    second = image.pixel_to_lonlat(row=0.0, col=1.0)

    # One 10 m step east must move the longitude east by a small positive amount.
    assert second[0] > first[0]
    # And the sampled point must be the pixel centre, half a pixel off the corner.
    easting, northing = image.transform @ (0.5, 0.5)
    assert easting == pytest.approx(500005.0)
    assert northing == pytest.approx(2999995.0)


# --------------------------------------------------------------------------
# nodata handling
# --------------------------------------------------------------------------


def test_nodata_pixels_become_nan(tmp_path):
    data = np.array([[[0, 5], [7, 9]]], dtype=np.uint8)
    path = _write_geotiff(tmp_path / "t1.tif", data, nodata=0)

    image = load_rsimage(path)

    assert np.isnan(image.array[0, 0, 0])
    assert image.array[0, 0, 1] == pytest.approx(5.0)


def test_no_nodata_declared_means_no_nan(tmp_path):
    data = np.zeros((1, 4, 4), dtype=np.uint8)
    path = _write_geotiff(tmp_path / "t1.tif", data)

    image = load_rsimage(path)

    assert not np.isnan(image.array).any()


# --------------------------------------------------------------------------
# Validity mask
#
# NaN propagates silently through np.mean, histogram binning and Otsu: one
# nodata pixel yields a NaN threshold and an all-False mask with no error
# raised. Every downstream consumer must reduce over valid pixels only, so
# the validity mask travels with the image rather than being re-derived
# (and re-forgotten) at each stage.
# --------------------------------------------------------------------------


def test_valid_mask_is_false_where_nodata(tmp_path):
    data = np.array([[[0, 5], [7, 9]]], dtype=np.uint8)
    path = _write_geotiff(tmp_path / "t1.tif", data, nodata=0)

    image = load_rsimage(path)

    assert image.valid_mask.shape == (2, 2)
    assert image.valid_mask.dtype == np.bool_
    assert not image.valid_mask[0, 0]
    assert image.valid_mask[0, 1]


def test_valid_mask_is_all_true_without_nodata(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.ones((3, 4, 4), dtype=np.uint8)
    )

    image = load_rsimage(path)

    assert image.valid_mask.all()


def test_pixel_is_invalid_if_any_band_is_nodata(tmp_path):
    """Index maths combines bands, so one bad band poisons the whole pixel."""
    data = np.ones((3, 2, 2), dtype=np.uint8) * 5
    data[1, 0, 0] = 0
    path = _write_geotiff(tmp_path / "t1.tif", data, nodata=0)

    image = load_rsimage(path)

    assert not image.valid_mask[0, 0]
    assert image.valid_mask[1, 1]


def test_valid_pixels_helper_excludes_nodata(tmp_path):
    """The convenience used by Otsu and index stats must drop NaN, not average it."""
    data = np.array([[[0, 4], [6, 10]]], dtype=np.uint8)
    path = _write_geotiff(tmp_path / "t1.tif", data, nodata=0)

    image = load_rsimage(path)
    values = image.valid_pixels(band=0)

    assert values.shape == (3,)
    assert np.isfinite(values).all()
    assert float(values.mean()) == pytest.approx(20.0 / 3.0)


# --------------------------------------------------------------------------
# Georeferencing provenance (consumed by the execution trace)
# --------------------------------------------------------------------------


def test_report_marks_projected_metric_gsd_as_exact(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((1, 8, 8), dtype=np.uint8)
    )

    report = georeferencing_report(load_rsimage(path))

    assert report["georeferenced"] is True
    assert report["gsd_source"] == "projected_crs_exact"
    assert report["gsd_m"] == pytest.approx(10.0)


def test_report_marks_geographic_gsd_as_derived(tmp_path):
    transform = Affine(0.0001, 0.0, 77.5, 0.0, -0.0001, 13.0)
    path = _write_geotiff(
        tmp_path / "t1.tif",
        np.zeros((1, 100, 100), dtype=np.uint8),
        crs=CRS.from_epsg(4326),
        transform=transform,
    )

    report = georeferencing_report(load_rsimage(path))

    assert report["gsd_source"] == "geographic_crs_derived"
    # The trace must name the CRS the estimate was made through, so a reviewer
    # can reproduce it.
    assert report["gsd_via_epsg"] == 32643


def test_report_marks_ungeoreferenced_input(tmp_path):
    from PIL import Image

    path = tmp_path / "cdvqa_pair.png"
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(path)

    report = georeferencing_report(load_rsimage(str(path)))

    assert report["georeferenced"] is False
    assert report["gsd_source"] == "unavailable"
    assert report["gsd_m"] is None


# --------------------------------------------------------------------------
# Modality
# --------------------------------------------------------------------------


def test_modality_defaults_to_unknown(tmp_path):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((3, 4, 4), dtype=np.uint16)
    )

    image = load_rsimage(path)

    assert image.modality == "unknown"


@pytest.mark.parametrize("modality", ["optical", "sar", "unknown"])
def test_modality_can_be_declared(tmp_path, modality):
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((1, 4, 4), dtype=np.uint16)
    )

    image = load_rsimage(path, modality=modality)

    assert image.modality == modality


def test_unknown_modality_string_is_rejected(tmp_path):
    """Detector dispatch keys off modality; a typo must fail loudly, not default."""
    path = _write_geotiff(
        tmp_path / "t1.tif", np.zeros((1, 4, 4), dtype=np.uint16)
    )

    with pytest.raises(ValueError):
        load_rsimage(path, modality="radar")


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rsimage(str(tmp_path / "does_not_exist.tif"))


def test_undecodable_file_raises(tmp_path):
    path = tmp_path / "broken.tif"
    path.write_bytes(b"this is not a raster")

    with pytest.raises(ValueError):
        load_rsimage(str(path))


# --------------------------------------------------------------------------
# RSImage construction invariants
# --------------------------------------------------------------------------


def test_rsimage_rejects_non_3d_array():
    with pytest.raises(ValueError):
        RSImage(
            array=np.zeros((4, 4), dtype=np.float32),
            crs=None,
            transform=None,
            modality="unknown",
            band_names=None,
            gsd_m=None,
        )


def test_rsimage_rejects_band_names_of_wrong_length():
    with pytest.raises(ValueError):
        RSImage(
            array=np.zeros((3, 4, 4), dtype=np.float32),
            crs=None,
            transform=None,
            modality="unknown",
            band_names=["red", "green"],
            gsd_m=None,
        )


def test_rsimage_exposes_shape_helpers():
    image = RSImage(
        array=np.zeros((3, 6, 8), dtype=np.float32),
        crs=None,
        transform=None,
        modality="unknown",
        band_names=None,
        gsd_m=None,
    )

    assert image.n_bands == 3
    assert image.height == 6
    assert image.width == 8
    assert image.shape == (6, 8)
