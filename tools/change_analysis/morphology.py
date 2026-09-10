"""Mask cleaning and connected-component labelling (build-order step 4).

Opening then closing, in that order: opening removes isolated speckle that
would otherwise become thousands of one-pixel "regions", and closing then
fills pinholes inside the surviving blobs. Doing it the other way round
would first grow the speckle, then fail to remove it.

No radius is hardcoded as a magic number. The defaults here are structural
(one pixel = the smallest element that can remove a single-pixel speckle),
and any larger value belongs in the config file with the ground sample
distance that motivates it -- a 3 m kernel means something different at
0.5 m/px than at 10 m/px.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

__all__ = ["clean_mask", "label_regions", "remove_small_regions"]


def _disk(radius: int) -> np.ndarray:
    """A disk-shaped structuring element of the given radius in pixels."""
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= radius * radius + 1e-9


def clean_mask(
    mask: np.ndarray, opening_radius: int = 1, closing_radius: int = 1
) -> np.ndarray:
    """Open then close a binary mask.

    Parameters
    ----------
    mask:
        boolean change mask.
    opening_radius:
        removes speckle smaller than this radius. 0 skips the step.
    closing_radius:
        fills holes smaller than this radius. 0 skips the step.

    Raises
    ------
    ValueError:
        a negative radius, which is silently meaningless rather than
        obviously wrong.
    """
    if opening_radius < 0 or closing_radius < 0:
        raise ValueError(
            f"radii must be >= 0; got opening={opening_radius}, "
            f"closing={closing_radius}"
        )
    out = np.asarray(mask, dtype=bool)
    if opening_radius > 0:
        out = ndimage.binary_opening(out, structure=_disk(opening_radius))
    if closing_radius > 0:
        out = ndimage.binary_closing(out, structure=_disk(closing_radius))
    return np.asarray(out, dtype=bool)


def label_regions(
    mask: np.ndarray, connectivity: int = 1
) -> Tuple[np.ndarray, int]:
    """Label connected components.

    ``connectivity=1`` is 4-connectivity, ``2`` is 8-connectivity. It is an
    explicit argument because the choice changes the region count for
    diagonally touching blobs, and region counts are reported to the user.

    Returns ``(labels, count)`` where ``labels`` is 0 for background.
    """
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2; got {connectivity}")
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labels, int(count)


def remove_small_regions(
    mask: np.ndarray, min_pixels: int, connectivity: int = 1
) -> np.ndarray:
    """Drop connected components smaller than ``min_pixels``.

    ``min_pixels`` is a caller decision, not a constant: the meaningful
    minimum depends on the ground sample distance and on what the user asked
    about. Convert an area in m2 to pixels via ``RSImage.pixel_area_m2``
    rather than guessing a pixel count.
    """
    if min_pixels <= 1:
        return np.asarray(mask, dtype=bool)
    labels, count = label_regions(mask, connectivity)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(sizes.shape[0], dtype=bool)
    keep[1:] = sizes[1:] >= min_pixels
    return keep[labels]
