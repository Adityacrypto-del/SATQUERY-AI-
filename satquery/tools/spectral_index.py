"""Spectral-index fallback for questions an RGB render cannot answer.

NDVI / NDWI / NDBI need NIR or SWIR reflectance. The VLM only ever sees an
RGB representation (see ``satquery.preprocessing.multispectral``), so asking
it for a vegetation index would get a plausible-sounding guess. This tool
computes the index from the real bands instead, and says so in its trace.

It refuses rather than fabricates:
- no NIR/SWIR band declared (plain RGB, or a multiband file without band
  names -- band positions are never guessed);
- SAR input (optical indices are meaningless on backscatter).

The index maths is NOT implemented here. It is imported from the bi-temporal
branch's ``tools.change_analysis.indices`` so there is one definition of each
index in the project. That checkout is found on ``sys.path`` or through the
``SATQUERY_BITEMPORAL_ROOT`` environment variable; if neither works the tool
reports ``status="unavailable"`` instead of crashing the single-image API.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

BITEMPORAL_ROOT_ENV = "SATQUERY_BITEMPORAL_ROOT"

FORMULAS = {
    "ndvi": "(NIR - RED) / (NIR + RED)",
    "ndwi": "(GREEN - NIR) / (GREEN + NIR)",
    "ndbi": "(SWIR - NIR) / (SWIR + NIR)",
}

METHOD = "physics-based spectral index computation (normalised band difference), not a VLM answer"


def _import_bitemporal():
    names = ("tools.change_analysis.indices", "tools.change_analysis.io")
    try:
        return tuple(importlib.import_module(n) for n in names), None
    except ImportError:
        pass
    root = os.environ.get(BITEMPORAL_ROOT_ENV)
    if not root:
        return (None, None), (
            f"tools.change_analysis is not importable and {BITEMPORAL_ROOT_ENV} is not set"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return tuple(importlib.import_module(n) for n in names), None
    except ImportError as exc:
        return (None, None), f"could not import tools.change_analysis from {root!r}: {exc}"


(indices, rs_io), _IMPORT_ERROR = _import_bitemporal()


@dataclass
class SpectralIndexResult:
    status: str                      # "ok" | "refused" | "unavailable"
    index: str
    answer: Optional[str]
    statistics: Optional[Dict[str, float]]
    reason: Optional[str]
    trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _trace(index: str, modality: str, bands_used: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    return {
        "tool": "satquery.tools.spectral_index",
        "method": METHOD,
        "is_vlm_answer": False,
        "index": index,
        "formula": FORMULAS[index],
        "bands_used": bands_used,
        "index_source": "tools.change_analysis.indices (bi-temporal branch)",
        "modality": modality,
    }


def _refuse(index: str, modality: str, reason: str, status: str = "refused") -> SpectralIndexResult:
    return SpectralIndexResult(status, index, None, None, reason, _trace(index, modality))


def _bands_used(image, index: str) -> Dict[str, str]:
    used = {}
    for logical in indices.INDEX_REQUIREMENTS[index]:
        aliases = indices._BAND_ALIASES[logical]
        used[logical] = next(n for n in image.band_names if str(n).strip().lower() in aliases)
    return used


def compute_spectral_index(image_path: str, index: str, modality: str = "unknown") -> SpectralIndexResult:
    key = index.strip().lower()
    if key not in FORMULAS:
        raise ValueError(f"unknown index {index!r}; expected one of {sorted(FORMULAS)}")
    if indices is None:
        return _refuse(key, modality, _IMPORT_ERROR, status="unavailable")
    if modality == "sar":
        return _refuse(
            key, modality,
            f"{key.upper()} is an optical index; SAR backscatter has no optical bands. "
            "Use the ratio operator in the bi-temporal detector for SAR change.",
        )

    image = rs_io.load_rsimage(image_path, modality=modality)
    try:
        values = indices.index_by_name(image, key)
    except ValueError as exc:
        why = (
            "This input declares no band names, so band positions are not guessed."
            if not image.band_names else
            "The required band is not among the declared bands."
        )
        if image.n_bands <= 3:
            why += " An RGB image carries no NIR/SWIR reflectance, so the index cannot be derived from it."
        return _refuse(key, modality, f"{exc} {why}")

    valid = np.isfinite(values) & image.valid_mask
    n_valid = int(valid.sum())
    if n_valid == 0:
        return _refuse(key, modality, f"{key.upper()} is undefined at every pixel (zero denominators or nodata).")

    v = values[valid].astype(np.float64)
    stats = {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
        # The sign of a normalised difference is definitional (numerator band
        # brighter than denominator band), not a tuned cutoff.
        "fraction_positive": float((v > 0).mean()),
        "valid_fraction": n_valid / valid.size,
        "valid_pixels": n_valid,
    }
    bands = _bands_used(image, key)
    answer = (
        f"{key.upper()} = {FORMULAS[key]} computed from bands "
        f"{', '.join(f'{k.upper()}={b}' for k, b in bands.items())}: "
        f"mean {stats['mean']:.3f}, median {stats['median']:.3f}, "
        f"5-95th percentile [{stats['p05']:.3f}, {stats['p95']:.3f}] over "
        f"{stats['valid_fraction']:.1%} valid pixels; positive on {stats['fraction_positive']:.1%} of them. "
        "Physics-based index computation, not a VLM answer."
    )
    return SpectralIndexResult("ok", key, answer, stats, None, _trace(key, modality, bands))
