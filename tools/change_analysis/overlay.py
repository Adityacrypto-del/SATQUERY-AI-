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
    "DISPLAY_PALETTE",
    "before_after_crops",
    "focus_comparison",
    "focus_legend",
    "overlay_mask",
    "save_png",
    "to_display_rgb",
]

# Overlay colours, chosen for contrast against aerial imagery rather than
# taken from SECOND's analysis palette. This module already separates display
# from analysis -- the percentile stretch destroys physical values and lives
# here for exactly that reason -- and the same argument applies to colour.
#
# SECOND paints NVG_surface mid-grey (128,128,128). Tinted at 55% over grey
# asphalt it is invisible, which is not a cosmetic problem: on one validation
# scene 65.4% of the queried change went to NVG_surface and 24.2% to
# buildings, and the rendered figure showed only the buildings. The picture
# was correct and read as the opposite of the answer beside it.
#
# So NVG_surface becomes orange, and the rest keep hues close to their
# analysis colours where those already had contrast. The mapping is returned
# by `focus_legend` so a caption can state it rather than leaving a viewer to
# infer it.
DISPLAY_PALETTE = {
    1: ((255, 140, 0), "NVG_surface"),      # orange, not grey-on-grey
    2: ((0, 200, 0), "low_vegetation"),     # green
    3: ((0, 100, 0), "trees"),              # dark green
    4: ((220, 30, 30), "buildings"),        # red
    5: ((0, 90, 255), "water"),             # blue
    6: ((230, 0, 230), "playgrounds"),      # magenta
}


def focus_legend(*class_maps) -> Dict[str, str]:
    """Class name to hex colour, for the classes actually present."""
    present = set()
    for semantic in class_maps:
        if semantic is not None:
            present.update(int(v) for v in np.unique(semantic))
    return {
        name: "#%02x%02x%02x" % rgb
        for index, (rgb, name) in DISPLAY_PALETTE.items()
        if index in present
    }


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


# --------------------------------------------------------------------------
# Query-scoped side-by-side evidence
# --------------------------------------------------------------------------


def focus_comparison(
    t1: RSImage,
    t2: RSImage,
    s_t1: np.ndarray,
    s_t2: np.ndarray,
    target_class: Optional[str] = None,
    alpha: float = 0.55,
    dim: float = 0.45,
    gap: int = 8,
) -> np.ndarray:
    """Both dates side by side, with only the queried change highlighted.

    The picture answers the same question the text does, from the same
    evidence. That is the point: a highlight computed independently of the
    answer could disagree with it, and a viewer would have no way to tell
    which was wrong. Here the mask *is* the pixels the rule counted.

    Scoping:

    *   with a target class, the focus is every pixel where that class was
        involved -- it was the class at t1, or became it at t2. Asking what
        water changed into highlights the former lake in both frames.
    *   without one, the focus is every changed pixel.

    Rendering. Inside the focus, each date is tinted by *its own* class, so a
    lake becoming a building reads as blue on the left and red on the right
    and the transition is legible at a glance. Colours come from
    :data:`DISPLAY_PALETTE`, chosen for contrast against aerial imagery
    rather than from SECOND's analysis palette -- see the note there for the
    case that forced it. ``alpha`` is partial so the underlying imagery stays visible
    through the tint; the evidence is the photograph, not the paint.

    Outside the focus, context is dimmed rather than erased. Blanking it
    would remove the surroundings a human needs to judge whether the
    highlighted region is plausible -- a lake is verified by its shoreline.

    Returns one ``(H, 2W + gap, 3)`` uint8 array, t1 left, t2 right, ready
    for :func:`save_png`.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if not 0.0 <= dim <= 1.0:
        raise ValueError(f"dim must be in [0, 1]; got {dim}")
    if s_t1.shape != s_t2.shape:
        raise ValueError(
            f"class maps must share a shape; got {s_t1.shape} and {s_t2.shape}"
        )

    from .cdvqa import NAME_TO_CLASS

    colours = {index: np.array(rgb, dtype=np.float32) / 255.0
               for index, (rgb, _) in DISPLAY_PALETTE.items()}

    if target_class is not None:
        wanted = NAME_TO_CLASS.get(target_class)
        if wanted is None:
            raise ValueError(f"unknown land-cover class {target_class!r}")
        focus = (s_t1 == wanted) | (s_t2 == wanted)
    else:
        focus = s_t1 != 0

    panels = []
    for image, semantic in ((t1, s_t1), (t2, s_t2)):
        # Blend in 0-1 for sane arithmetic, convert back at the end:
        # to_display_rgb emits uint8 and save_png expects uint8.
        rgb = to_display_rgb(image).astype(np.float32) / 255.0
        # Dim the context: toward grey, not toward black, so dark land cover
        # does not disappear into the background.
        outside = ~focus
        rgb[outside] = rgb[outside] * dim + (1.0 - dim) * 0.5
        for index, colour in colours.items():
            if index == 0:
                continue
            paint = focus & (semantic == index)
            if paint.any():
                rgb[paint] = (1.0 - alpha) * rgb[paint] + alpha * colour
        panels.append(np.clip(rgb, 0.0, 1.0))

    height = panels[0].shape[0]
    separator = np.ones((height, gap, 3), dtype=np.float32)
    joined = np.concatenate([panels[0], separator, panels[1]], axis=1)
    return (np.clip(joined, 0.0, 1.0) * 255.0).astype(np.uint8)
