"""Visual evidence: mask overlays and before/after crops (build-order step 7).

The problem statement grades observable evidence, so a change claim should
come with a picture a human can check. Two products:

*   a mask overlay on the later image, showing where the change was found,
*   paired before/after crops around each region, showing what changed.

Display conversion is deliberately separate from analysis. Percentile
stretching makes a satellite tile legible on a screen but destroys physical
values, so it happens here and never in ``io`` or ``detector`` -- the
analysis path always sees the real numbers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .io import RSImage
from .regions import Region

__all__ = [
    "before_after_crops",
    "overlay_mask",
    "to_display_rgb",
]

_RGB_ALIASES = (
    ("red", {"red", "r", "b04", "b4"}),
    ("green", {"green", "g", "b03", "b3"}),
    ("blue", {"blue", "b", "b02", "b2"}),
)


def _select_rgb_bands(image: RSImage) -> Tuple[List[int], str]:
    """Choose three bands to display, and say how they were chosen."""
    if image.band_names:
        indices = []
        for _, aliases in _RGB_ALIASES:
            found = next(
                (i for i, name in enumerate(image.band_names)
                 if str(name).strip().lower() in aliases),
                None,
            )
            indices.append(found)
        if all(i is not None for i in indices):
            return [int(i) for i in indices], "declared_rgb_bands"
    if image.n_bands >= 3:
        return [0, 1, 2], "first_three_bands_fallback"
    return [0, 0, 0], "single_band_replicated"


def to_display_rgb(
    image: RSImage, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> np.ndarray:
    """Convert an RSImage to an ``(H, W, 3)`` uint8 image for display.

    Percentile stretch, computed over valid pixels only so one nodata pixel
    cannot flatten the whole scene. This is a display transform: the values
    it produces are not physical and must never be fed back into analysis.
    """
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError(
            f"percentiles must satisfy 0 <= low < high <= 100; "
            f"got {low_percentile} and {high_percentile}"
        )

    indices, _ = _select_rgb_bands(image)
    planes = []
    for index in indices:
        band = image.array[index].astype(np.float32)
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            planes.append(np.zeros(band.shape, dtype=np.float32))
            continue
        low, high = np.percentile(finite, (low_percentile, high_percentile))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            planes.append(np.zeros(band.shape, dtype=np.float32))
            continue
        scaled = (np.nan_to_num(band, nan=low) - low) / (high - low)
        planes.append(np.clip(scaled, 0.0, 1.0))
    return (np.stack(planes, axis=-1) * 255.0).astype(np.uint8)


def overlay_mask(
    image: RSImage,
    mask: np.ndarray,
    colour: Sequence[int] = (255, 0, 0),
    alpha: float = 0.4,
    outline_only: bool = False,
) -> np.ndarray:
    """Tint the masked pixels of ``image``, returning ``(H, W, 3)`` uint8.

    ``alpha`` is the tint strength: 0 leaves the image untouched, 1 replaces
    the masked pixels outright. Unmasked pixels are never modified, so a
    reader can trust that the untinted area is the original scene.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {image.shape}"
        )

    rgb = to_display_rgb(image).astype(np.float32)
    paint = mask
    if outline_only:
        from scipy import ndimage

        eroded = ndimage.binary_erosion(mask, iterations=1)
        paint = mask & ~eroded

    tint = np.asarray(colour, dtype=np.float32)
    rgb[paint] = (1.0 - alpha) * rgb[paint] + alpha * tint
    return np.clip(rgb, 0, 255).astype(np.uint8)


def before_after_crops(
    t1: RSImage,
    t2: RSImage,
    region: Region,
    padding: int = 16,
) -> Dict[str, np.ndarray]:
    """Matching crops around one region, from both dates.

    Both crops use identical bounds, so the pair is directly comparable; the
    window is clipped to the image, which means a region at the edge yields a
    smaller crop rather than a padded or wrapped one.
    """
    if padding < 0:
        raise ValueError(f"padding must be >= 0; got {padding}")
    if t1.shape != t2.shape:
        raise ValueError(
            f"t1 and t2 must share a shape; got {t1.shape} and {t2.shape}"
        )

    row_min, col_min, row_max, col_max = region.bbox
    row_start = max(row_min - padding, 0)
    col_start = max(col_min - padding, 0)
    row_stop = min(row_max + padding, t1.height)
    col_stop = min(col_max + padding, t1.width)

    before = to_display_rgb(t1)[row_start:row_stop, col_start:col_stop]
    after = to_display_rgb(t2)[row_start:row_stop, col_start:col_stop]
    return {
        "before": before,
        "after": after,
        "window": (row_start, col_start, row_stop, col_stop),
    }


def save_png(path: str, rgb: np.ndarray) -> str:
    """Write an ``(H, W, 3)`` uint8 array to a PNG."""
    from PIL import Image

    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
    return path
