from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PreprocessTrace:
    bands_used: List[str]
    normalization: str
    resize: str


def to_model_rgb(array: np.ndarray) -> tuple[np.ndarray, Dict[str, str]]:
    if array.ndim != 3:
        raise ValueError("Expected HxWxC array")

    channels = array.shape[-1]
    if channels >= 3:
        rgb = array[..., :3]
        bands = ["B04", "B03", "B02"] if channels > 3 else ["R", "G", "B"]
    else:
        rgb = np.repeat(array[..., :1], 3, axis=-1)
        bands = ["single_band_repeated"]

    rgb = rgb.astype("float32")
    if rgb.max() > 0:
        rgb = rgb / rgb.max()

    return rgb, {
        "bands_used": ",".join(bands),
        "normalization": "minmax_per_image",
        "resize": "model_default",
    }
