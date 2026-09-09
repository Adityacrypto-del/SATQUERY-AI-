from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np


@dataclass
class PreprocessTrace:
    bands_used: List[str]
    normalization: str
    resize: str


def _find_band(descriptions: list[str], candidates: set[str]) -> int | None:
    for index, description in enumerate(descriptions):
        normalized = str(description or "").strip().upper().replace("_", "")
        if normalized in candidates:
            return index
    return None


def to_model_rgb(
    array: np.ndarray, metadata: Mapping[str, Any] | None = None
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Create an RGB representation without claiming the VLM uses all bands.

    RGB GeoTIFFs use their declared colour interpretation.  Sentinel-style
    named bands use B04/B03/B02.  An unnamed multi-band file falls back to its
    first three bands and records that conservative assumption in the trace.
    """
    if array.ndim != 3:
        raise ValueError("Expected HxWxC array")

    channels = array.shape[-1]
    metadata = metadata or {}
    descriptions = list(metadata.get("band_descriptions") or [])
    color_interpretations = [str(v).lower() for v in metadata.get("color_interpretations") or []]
    if channels >= 3:
        rgb_indices = [
            next((i for i, value in enumerate(color_interpretations) if value == name), None)
            for name in ("red", "green", "blue")
        ]
        if all(index is not None for index in rgb_indices):
            indices = [int(index) for index in rgb_indices]
            bands = ["red", "green", "blue"]
            selection = "declared_rgb_channels"
        else:
            sentinel_indices = [
                _find_band(descriptions, {"B04", "B4"}),
                _find_band(descriptions, {"B03", "B3"}),
                _find_band(descriptions, {"B02", "B2"}),
            ]
            if all(index is not None for index in sentinel_indices):
                indices = [int(index) for index in sentinel_indices]
                bands = ["B04", "B03", "B02"]
                selection = "named_sentinel_bands"
            else:
                indices = [0, 1, 2]
                bands = ["band_1", "band_2", "band_3"]
                selection = "first_three_bands_fallback"
        rgb = array[..., indices]
    else:
        rgb = np.repeat(array[..., :1], 3, axis=-1)
        bands = ["single_band_repeated"]
        selection = "single_band_repeated"

    rgb = rgb.astype("float32")
    # Robust percentile scaling prevents one bright outlier from making the
    # whole satellite tile nearly black.  It also produces a valid PIL image
    # for regular PNG/JPEG and multispectral GeoTIFF inputs.
    low, high = np.nanpercentile(rgb, (2, 98))
    if np.isfinite(low) and np.isfinite(high) and high > low:
        rgb = np.clip((rgb - low) / (high - low), 0.0, 1.0)
    else:
        rgb = np.zeros_like(rgb, dtype="float32")

    return rgb, {
        "bands_used": bands,
        "source_band_count": channels,
        "band_selection": selection,
        "normalization": "per_image_p02_p98_clip",
        "resize": "model_default",
        "model_input": "RGB representation; non-RGB source bands are not passed to the VLM",
    }
