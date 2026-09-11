"""Per-region attributes from a change mask (build-order step 4).

Region-level (object-based) reasoning is measurably more robust to
misregistration than pixel-level -- 40-133% better depending on landscape
type -- so every statistic reported to the user is a region aggregate rather
than a pixel tally.

Area always comes from the image's geotransform, never from a caller-supplied
constant. When the source carries no georeferencing -- every CDVQA benchmark
PNG -- ``area_m2`` and ``centroid_lonlat`` are None. A pixel count is still a
true measurement; a fabricated square-metre figure is not (HARD RULE 3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .io import RSImage
from .morphology import label_regions

__all__ = ["Region", "extract_regions", "summarise_regions"]


@dataclass
class Region:
    """One connected changed area."""

    label: int
    pixel_count: int
    area_m2: Optional[float]
    centroid_rowcol: Tuple[float, float]
    centroid_lonlat: Optional[Tuple[float, float]]
    bbox: Tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max)
    # Mean spectral index values inside the region, before and after. This is
    # the region's spectral signature: it lets a reader see *what kind* of
    # change happened (NDVI collapsing and NDBI rising reads as vegetation
    # lost to construction) rather than only that something changed. None
    # when no index stack was supplied.
    signature_t1: Optional[Dict[str, float]] = None
    signature_t2: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_regions(
    mask: np.ndarray,
    image: RSImage,
    min_pixels: int = 1,
    connectivity: int = 1,
    index_stacks: Optional[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = None,
) -> List[Region]:
    """Extract per-region attributes from a binary change mask.

    Parameters
    ----------
    mask:
        boolean change mask, same spatial shape as ``image``.
    image:
        the RSImage the mask was computed over. Supplies the geotransform,
        which is the only legitimate source of area and coordinates.
    min_pixels:
        regions smaller than this are dropped. Convert a real-world minimum
        area via ``image.pixel_area_m2`` rather than guessing pixels.

    Regions come back sorted by descending pixel count, so the first entry is
    the dominant change.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {image.shape}"
        )
    if min_pixels < 1:
        raise ValueError(f"min_pixels must be >= 1; got {min_pixels}")

    labels, count = label_regions(mask, connectivity=connectivity)
    if count == 0:
        return []

    indices = list(range(1, count + 1))
    sizes = ndimage.sum_labels(mask, labels, index=indices)
    centroids = ndimage.center_of_mass(mask, labels, indices)
    boxes = ndimage.find_objects(labels)

    pixel_area = image.pixel_area_m2
    stack_t1, stack_t2 = index_stacks if index_stacks else (None, None)

    def _signature(stack, region_mask):
        """Mean index values over the region, ignoring NaN."""
        if not stack:
            return None
        out: Dict[str, float] = {}
        for name, plane in stack.items():
            values = plane[region_mask]
            values = values[np.isfinite(values)]
            out[name] = float(values.mean()) if values.size else float("nan")
        return out

    regions: List[Region] = []
    for position, label in enumerate(indices):
        pixel_count = int(sizes[position])
        if pixel_count < min_pixels:
            continue
        row, col = centroids[position]
        row_slice, col_slice = boxes[position]
        region_mask = labels == label
        regions.append(
            Region(
                label=label,
                pixel_count=pixel_count,
                area_m2=(pixel_count * pixel_area) if pixel_area is not None else None,
                centroid_rowcol=(float(row), float(col)),
                centroid_lonlat=image.pixel_to_lonlat(float(row), float(col)),
                bbox=(
                    int(row_slice.start), int(col_slice.start),
                    int(row_slice.stop), int(col_slice.stop),
                ),
                signature_t1=_signature(stack_t1, region_mask),
                signature_t2=_signature(stack_t2, region_mask),
            )
        )

    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return regions


def summarise_regions(
    regions: List[Region], image: RSImage
) -> Dict[str, Any]:
    """Aggregate statistics over regions, for the trace and the summary text.

    ``total_area_m2`` is None when the image is ungeoreferenced, rather than
    a zero that would read as "no change detected".
    """
    pixel_total = sum(r.pixel_count for r in regions)
    areas = [r.area_m2 for r in regions if r.area_m2 is not None]
    return {
        "n_regions": len(regions),
        "total_changed_pixels": pixel_total,
        "total_area_m2": sum(areas) if areas else None,
        "largest_region_pixels": regions[0].pixel_count if regions else 0,
        "largest_region_area_m2": regions[0].area_m2 if regions else None,
        "fraction_of_scene": (
            pixel_total / float(image.height * image.width)
            if image.height and image.width else None
        ),
    }
