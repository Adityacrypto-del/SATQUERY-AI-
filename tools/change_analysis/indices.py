"""Spectral indices for optical change detection (build-order step 4).

Change detection differences the *indices*, not the raw DNs: raw brightness
moves with sun angle, atmosphere and sensor gain between two acquisition
dates, so differencing it measures the acquisition as much as the ground.
A normalised index is a ratio of bands from the same acquisition, which
cancels most of that.

    NDVI = (NIR  - RED ) / (NIR  + RED )   vegetation
    NDWI = (GREEN - NIR ) / (GREEN + NIR )   open water
    NDBI = (SWIR - NIR ) / (SWIR + NIR )   built-up

All three are undefined where the denominator is zero. That is returned as
NaN, never inf and never a substituted zero: inf survives into Otsu looking
like a finite extreme and drags the threshold with it, and zero is a real
index value that means "no contrast", which is a different claim.

Indices are optical-only. SAR uses the ratio operator in ``detector`` --
speckle is multiplicative, so differencing it gives a false alarm rate that
climbs with backscatter magnitude.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .io import RSImage

__all__ = [
    "INDEX_REQUIREMENTS",
    "available_indices",
    "index_by_name",
    "ndbi",
    "ndvi",
    "ndwi",
    "normalized_difference",
]

# Accepted spellings per logical band. Sentinel-2 codes and plain names both
# appear in the wild, and a GeoTIFF written by a different tool may use
# either, so resolution is by alias set rather than by position.
_BAND_ALIASES: Dict[str, frozenset] = {
    "red": frozenset({"red", "r", "b04", "b4"}),
    "green": frozenset({"green", "g", "b03", "b3"}),
    "blue": frozenset({"blue", "b", "b02", "b2"}),
    "nir": frozenset({"nir", "n", "b08", "b8", "b8a"}),
    "swir": frozenset({"swir", "swir1", "b11"}),
}

# Which logical bands each index needs, in (numerator, denominator) order.
INDEX_REQUIREMENTS: Dict[str, Tuple[str, str]] = {
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),
    "ndbi": ("swir", "nir"),
}


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)``, with an undefined denominator returned as NaN.

    NaN in either input propagates, which is intended: the validity mask
    carried by :class:`~tools.change_analysis.io.RSImage` is what downstream
    reductions use to exclude those pixels.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / denominator
    out = np.asarray(out, dtype=np.float32)
    out[denominator == 0] = np.nan
    return out


def _resolve(image: RSImage, logical: str, index_name: str) -> np.ndarray:
    """Return the band plane for a logical band name, or raise."""
    aliases = _BAND_ALIASES[logical]
    if image.band_names:
        for position, name in enumerate(image.band_names):
            if str(name).strip().lower() in aliases:
                return image.array[position]
    raise ValueError(
        f"cannot compute {index_name.upper()}: no {logical.upper()} band found. "
        f"Declared band names: {image.band_names!r}. "
        f"Accepted spellings for {logical.upper()}: {sorted(aliases)!r}."
    )


def index_by_name(image: RSImage, name: str) -> np.ndarray:
    """Compute a named index over ``image``."""
    key = name.strip().lower()
    if key not in INDEX_REQUIREMENTS:
        raise ValueError(
            f"unknown index {name!r}; expected one of {sorted(INDEX_REQUIREMENTS)!r}"
        )
    numerator, denominator = INDEX_REQUIREMENTS[key]
    return normalized_difference(
        _resolve(image, numerator, key), _resolve(image, denominator, key)
    )


def ndvi(image: RSImage) -> np.ndarray:
    """Normalised Difference Vegetation Index. Positive over vegetation."""
    return index_by_name(image, "ndvi")


def ndwi(image: RSImage) -> np.ndarray:
    """Normalised Difference Water Index. Positive over open water."""
    return index_by_name(image, "ndwi")


def ndbi(image: RSImage) -> np.ndarray:
    """Normalised Difference Built-up Index. Positive over built-up surfaces."""
    return index_by_name(image, "ndbi")


def _has_band(image: RSImage, logical: str) -> bool:
    if not image.band_names:
        return False
    aliases = _BAND_ALIASES[logical]
    return any(
        str(name).strip().lower() in aliases for name in image.band_names
    )


def available_indices(image: RSImage) -> List[str]:
    """Indices computable from this image's declared bands.

    Empty for SAR: the ratio operator in ``detector`` is the SAR path, and
    offering an optical index for radar backscatter would be meaningless.
    """
    if image.modality == "sar":
        return []
    return [
        name
        for name, needed in INDEX_REQUIREMENTS.items()
        if all(_has_band(image, logical) for logical in needed)
    ]
