"""GeoTIFF I/O and the RSImage data contract (build-order step 1).

One object flows through every stage of the bi-temporal pipeline. Bare numpy
arrays are never passed between modules, and PIL is never used to open a
GeoTIFF -- it drops the CRS and geotransform, which are the only reason
area-in-metres and centroid-in-lat/lon are computable at all.

This module owns the geospatial data path. Ayush's single-image loader in
``satquery.preprocessing.image_loader`` prepares an (H, W, bands) RGB
representation for the VLM, which is a different job; where it is importable
its per-image checks are layered on top as an *extra* check by
``tools.validation.pair``, never as a substitute for these.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.warp import transform as warp_transform

__all__ = [
    "MODALITIES",
    "RSImage",
    "georeferencing_report",
    "load_rsimage",
]

MODALITIES = ("optical", "sar", "unknown")

# Latitude beyond which UTM is not defined; polar input falls back to a
# spherical estimate rather than silently reprojecting into a bad zone.
_UTM_LATITUDE_LIMIT = 84.0

# Mean Earth radius (metres), used only for the polar fallback.
_EARTH_RADIUS_M = 6371008.8


@dataclass
class RSImage:
    """A single remote-sensing observation with its georeferencing intact.

    Attributes
    ----------
    array:
        ``(bands, H, W)`` float32. Band-first, matching ``rasterio.read()``.
        nodata pixels are NaN, which is why the dtype is float.
    crs:
        rasterio CRS, or None when the source carries no georeferencing.
    transform:
        affine geotransform, or None when the source carries none.
    modality:
        one of :data:`MODALITIES`. Detector dispatch keys off this.
    band_names:
        per-band names when the file declares them, else None.
    gsd_m:
        ground sample distance in metres, or None when not derivable.
        See :func:`georeferencing_report` for how it was obtained.
    """

    array: np.ndarray
    crs: Any
    transform: Any
    modality: str
    band_names: Optional[List[str]]
    gsd_m: Optional[float]

    # Exact per-axis ground extents in metres. Kept private so the public
    # contract stays at the six fields specified in CLAUDE.md section 3.
    _pixel_size_m: Optional[Tuple[float, float]] = None
    _gsd_source: str = "unavailable"
    _gsd_via_epsg: Optional[int] = None

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(
                "RSImage.array must be (bands, H, W); "
                f"got shape {self.array.shape!r}"
            )
        if self.modality not in MODALITIES:
            raise ValueError(
                f"modality must be one of {MODALITIES}; got {self.modality!r}"
            )
        if self.band_names is not None and len(self.band_names) != self.n_bands:
            raise ValueError(
                f"band_names has {len(self.band_names)} entries for "
                f"{self.n_bands} bands"
            )

    # -- shape ------------------------------------------------------------

    @property
    def n_bands(self) -> int:
        return int(self.array.shape[0])

    @property
    def height(self) -> int:
        return int(self.array.shape[1])

    @property
    def width(self) -> int:
        return int(self.array.shape[2])

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial shape ``(H, W)``, excluding the band axis."""
        return (self.height, self.width)

    # -- validity ---------------------------------------------------------

    @property
    def valid_mask(self) -> np.ndarray:
        """``(H, W)`` bool, True where *every* band is finite.

        Conservative on purpose: index arithmetic combines bands, so a pixel
        with one nodata band cannot produce a trustworthy NDVI.
        """
        return np.isfinite(self.array).all(axis=0)

    def valid_pixels(self, band: int = 0) -> np.ndarray:
        """Finite values of ``band`` as a flat 1-D array.

        Use this for any reduction -- means, percentiles, histograms, Otsu.
        Reducing over ``array`` directly propagates NaN and produces a NaN
        threshold with no error raised.
        """
        values = self.array[band]
        return values[np.isfinite(values)]

    # -- bands ------------------------------------------------------------

    def band_index(self, name: str) -> Optional[int]:
        """Index of the band called ``name``, case-insensitively, else None."""
        if not self.band_names:
            return None
        target = name.strip().lower()
        for index, band_name in enumerate(self.band_names):
            if str(band_name).strip().lower() == target:
                return index
        return None

    # -- geometry ---------------------------------------------------------

    @property
    def pixel_area_m2(self) -> Optional[float]:
        """Ground area of one pixel, or None when not derivable.

        Always computed from the geotransform, never from a caller-supplied
        constant, and from both axes so anisotropic pixels stay exact.
        """
        if self._pixel_size_m is None:
            return None
        x_m, y_m = self._pixel_size_m
        return float(x_m * y_m)

    def pixel_to_lonlat(
        self, row: float, col: float
    ) -> Optional[Tuple[float, float]]:
        """Convert pixel ``(row, col)`` to WGS84 ``(lon, lat)``.

        ``row``/``col`` address pixel *centres*: row 0 is the middle of the
        top row, not its upper edge. Returns None for ungeoreferenced input
        rather than inventing a coordinate.
        """
        if self.transform is None or self.crs is None:
            return None
        x, y = self.transform @ (col + 0.5, row + 0.5)
        lons, lats = warp_transform(self.crs, "EPSG:4326", [x], [y])
        return (float(lons[0]), float(lats[0]))


# --------------------------------------------------------------------------
# Ground sample distance
# --------------------------------------------------------------------------


def _utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing ``(lon, lat)``."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _projected_pixel_size_m(
    crs: Any, transform: Any
) -> Optional[Tuple[float, float]]:
    """Per-axis pixel size in metres for a projected CRS.

    Uses the full affine rather than ``a``/``e`` alone so rotated (non
    north-up) transforms stay correct, and applies the CRS linear-unit
    factor so a grid in feet is not reported as metres.
    """
    factor = 1.0
    units = getattr(crs, "linear_units_factor", None)
    if units is not None:
        try:
            factor = float(units[1])
        except (TypeError, ValueError, IndexError):
            factor = 1.0
    x_m = math.hypot(transform.a, transform.d) * factor
    y_m = math.hypot(transform.b, transform.e) * factor
    if not (math.isfinite(x_m) and math.isfinite(y_m)) or x_m <= 0 or y_m <= 0:
        return None
    return (x_m, y_m)


def _geographic_pixel_size_m(
    crs: Any, transform: Any, height: int, width: int
) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
    """Per-axis ground extent in metres for a geographic (degree) CRS.

    Reprojects the image-centre pixel and its two immediate neighbours into
    the local UTM zone and measures the resulting distances. This is a
    standard geodesic conversion, not an assumed constant: it accounts for
    meridian convergence, so an east-west pixel correctly shrinks as
    cos(latitude).
    """
    centre_col = width / 2.0
    centre_row = height / 2.0
    centre_lon, centre_lat = transform @ (centre_col + 0.5, centre_row + 0.5)

    if abs(centre_lat) > _UTM_LATITUDE_LIMIT:
        # UTM is undefined near the poles; fall back to a spherical estimate.
        deg_y = math.hypot(transform.b, transform.e)
        deg_x = math.hypot(transform.a, transform.d)
        metres_per_degree = math.pi * _EARTH_RADIUS_M / 180.0
        y_m = deg_y * metres_per_degree
        x_m = deg_x * metres_per_degree * math.cos(math.radians(centre_lat))
        if x_m <= 0 or y_m <= 0:
            return (None, None)
        return ((x_m, y_m), None)

    epsg = _utm_epsg(centre_lon, centre_lat)
    corners = [
        (centre_col + 0.5, centre_row + 0.5),
        (centre_col + 1.5, centre_row + 0.5),
        (centre_col + 0.5, centre_row + 1.5),
    ]
    lons: List[float] = []
    lats: List[float] = []
    for col, row in corners:
        lon, lat = transform @ (col, row)
        lons.append(lon)
        lats.append(lat)

    try:
        xs, ys = warp_transform(crs, f"EPSG:{epsg}", lons, lats)
    except Exception:
        return (None, None)

    x_m = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
    y_m = math.hypot(xs[2] - xs[0], ys[2] - ys[0])
    if not (math.isfinite(x_m) and math.isfinite(y_m)) or x_m <= 0 or y_m <= 0:
        return (None, None)
    return ((x_m, y_m), epsg)


def _resolve_gsd(
    crs: Any, transform: Any, height: int, width: int
) -> Tuple[Optional[Tuple[float, float]], Optional[float], str, Optional[int]]:
    """Return ``(pixel_size_m, gsd_m, source, via_epsg)``.

    ``gsd_m`` is the mean of the two axis extents and is a summary value
    only; anisotropic pixels keep their exact area via ``pixel_size_m``.
    """
    if crs is None or transform is None:
        return (None, None, "unavailable", None)

    if getattr(crs, "is_geographic", False):
        size, epsg = _geographic_pixel_size_m(crs, transform, height, width)
        if size is None:
            return (None, None, "unavailable", None)
        return (size, (size[0] + size[1]) / 2.0, "geographic_crs_derived", epsg)

    size = _projected_pixel_size_m(crs, transform)
    if size is None:
        return (None, None, "unavailable", None)
    return (size, (size[0] + size[1]) / 2.0, "projected_crs_exact", None)


def georeferencing_report(image: RSImage) -> Dict[str, Any]:
    """Provenance of an image's georeferencing, for the execution trace.

    The trace must never leave it ambiguous whether a GSD was read from a
    projected grid, derived geodesically from degrees, or unavailable.
    """
    return {
        "georeferenced": image.crs is not None and image.transform is not None,
        "crs": str(image.crs) if image.crs is not None else None,
        "gsd_m": image.gsd_m,
        "gsd_source": image._gsd_source,
        "gsd_via_epsg": image._gsd_via_epsg,
        "pixel_size_m": (
            list(image._pixel_size_m)
            if image._pixel_size_m is not None
            else None
        ),
        "pixel_area_m2": image.pixel_area_m2,
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _clean_band_names(descriptions: Any, count: int) -> Optional[List[str]]:
    """Band names, or None when the file declares none for any band."""
    if not descriptions:
        return None
    names = [d for d in descriptions]
    if len(names) != count or any(d in (None, "") for d in names):
        return None
    return [str(d) for d in names]


def _is_georeferenced(crs: Any, transform: Any) -> bool:
    """True when the file carries usable georeferencing.

    GDAL hands back an identity transform for a plain image, which is a
    placeholder rather than a real geotransform; treating it as real would
    put every CDVQA PNG at longitude 0.
    """
    if crs is None or transform is None:
        return False
    return not transform.is_identity


def load_rsimage(path: str, modality: str = "unknown") -> RSImage:
    """Load a raster into an :class:`RSImage`, preserving its georeferencing.

    Parameters
    ----------
    path:
        GeoTIFF, TIFF, PNG or JPEG. Non-georeferenced formats load with
        ``crs``, ``transform`` and ``gsd_m`` set to None.
    modality:
        one of :data:`MODALITIES`. Rejected if unrecognised -- the detector
        picks difference vs ratio from this, so a typo must fail loudly
        rather than fall through to a default.

    Raises
    ------
    FileNotFoundError:
        the path does not exist.
    ValueError:
        the modality is unrecognised, or the file cannot be decoded.
    """
    if modality not in MODALITIES:
        raise ValueError(
            f"modality must be one of {MODALITIES}; got {modality!r}"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        with rasterio.open(path) as src:
            # Read every band. Selecting bands at load time makes the choice
            # unauditable and is destructive for multispectral input.
            array = src.read().astype(np.float32)
            crs = src.crs
            transform = src.transform
            band_names = _clean_band_names(src.descriptions, src.count)
            nodata = src.nodata
    except RasterioIOError as exc:
        raise ValueError(f"could not decode raster {path!r}: {exc}") from exc

    if nodata is not None and np.isfinite(nodata):
        array[array == np.float32(nodata)] = np.nan

    if not _is_georeferenced(crs, transform):
        crs = None
        transform = None

    height, width = array.shape[1], array.shape[2]
    pixel_size_m, gsd_m, gsd_source, via_epsg = _resolve_gsd(
        crs, transform, height, width
    )

    return RSImage(
        array=array,
        crs=crs,
        transform=transform,
        modality=modality,
        band_names=band_names,
        gsd_m=gsd_m,
        _pixel_size_m=pixel_size_m,
        _gsd_source=gsd_source,
        _gsd_via_epsg=via_epsg,
    )
