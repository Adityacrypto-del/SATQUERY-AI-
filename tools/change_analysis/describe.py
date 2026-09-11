"""Template description generator: region attributes into a sentence.

The unconditional fallback. No model inference and no query parsing -- it
always works, so the pipeline can always say something truthful even when
no rule applies and no checkpoint is loaded.

Every number in the output is one that was actually measured. Where the
input is ungeoreferenced the description says "pixels" and never converts to
an area it cannot derive, because a fabricated square-metre figure is the
exact failure HARD RULE 3 exists to prevent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .cdvqa import CLASS_NAMES
from .io import RSImage
from .regions import Region

__all__ = ["describe_change", "describe_transitions"]

_READABLE = {
    "NVG_surface": "non-vegetated ground surface",
    "low_vegetation": "low vegetation",
    "trees": "trees",
    "buildings": "buildings",
    "water": "water",
    "playgrounds": "playgrounds",
}


def _readable(name: Optional[str]) -> str:
    return _READABLE.get(name or "", name or "an unidentified class")


def _area_phrase(region: Region) -> str:
    """Area in m2 where derivable, otherwise an honest pixel count."""
    if region.area_m2 is not None:
        if region.area_m2 >= 1e6:
            return f"{region.area_m2 / 1e6:.2f} km2"
        return f"{region.area_m2:,.0f} m2"
    return f"{region.pixel_count:,} pixels"


def _location_phrase(region: Region) -> str:
    if region.centroid_lonlat is not None:
        lon, lat = region.centroid_lonlat
        hemisphere = "N" if lat >= 0 else "S"
        meridian = "E" if lon >= 0 else "W"
        return f"centred near {abs(lat):.4f}{hemisphere}, {abs(lon):.4f}{meridian}"
    row, col = region.centroid_rowcol
    return f"centred at pixel (row {row:.0f}, col {col:.0f})"


def describe_transitions(
    s_t1: Optional[np.ndarray], s_t2: Optional[np.ndarray], top: int = 3
) -> List[str]:
    """Most common ``X became Y`` transitions, largest first."""
    if s_t1 is None or s_t2 is None:
        return []
    changed = s_t1 != 0
    if not changed.any():
        return []
    pairs, counts = np.unique(
        np.stack([s_t1[changed], s_t2[changed]]), axis=1, return_counts=True
    )
    order = np.argsort(-counts)
    out: List[str] = []
    for index in order[:top]:
        before = int(pairs[0, index])
        after = int(pairs[1, index])
        if before == 0 or after == 0 or before == after:
            continue
        out.append(
            f"{_readable(CLASS_NAMES.get(before))} became "
            f"{_readable(CLASS_NAMES.get(after))}"
        )
    return out


def describe_change(
    regions: Sequence[Region],
    summary: Dict[str, Any],
    image: RSImage,
    s_t1: Optional[np.ndarray] = None,
    s_t2: Optional[np.ndarray] = None,
) -> str:
    """One or two sentences describing what changed, where, and how much."""
    if not regions or summary.get("total_changed_pixels", 0) == 0:
        return "No change was detected between the two observations."

    count = len(regions)
    fraction = summary.get("fraction_of_scene")
    total_area = summary.get("total_area_m2")

    if total_area is not None:
        extent = (
            f"{total_area / 1e6:.2f} km2" if total_area >= 1e6
            else f"{total_area:,.0f} m2"
        )
    else:
        extent = f"{summary['total_changed_pixels']:,} pixels"

    plural = "region" if count == 1 else "regions"
    opening = f"Detected change across {count} {plural}, covering {extent}"
    if fraction is not None:
        opening += f" ({fraction * 100:.1f}% of the scene)"
    opening += "."

    largest = regions[0]
    detail = (
        f" The largest single area spans {_area_phrase(largest)}, "
        f"{_location_phrase(largest)}."
    )

    transitions = describe_transitions(s_t1, s_t2)
    if transitions:
        if len(transitions) == 1:
            detail += f" The dominant transition is {transitions[0]}."
        else:
            listed = "; ".join(transitions[:-1]) + f"; and {transitions[-1]}"
            detail += f" The main transitions are {listed}."

    return opening + detail
