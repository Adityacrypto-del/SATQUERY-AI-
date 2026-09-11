"""Producer B: training-free semantic change maps from spectral indices.

The trained SECOND model is a 3-band RGB segmenter. It has no coverage for
multispectral stacks with arbitrary band counts, and none at all for SAR --
which is exactly the shape of the ISRO/SAC evaluation data (Cartosat-2S
optical and RISAT SAR). This producer fills that gap using physics rather
than training: NDVI, NDWI and NDBI from ``indices.py``, thresholded by Otsu
from ``detector.py``, mapped onto SECOND's class IDs.

It is an approximation and says so. Three limitations are structural, not
bugs, and each is reported rather than hidden:

*   ``playgrounds`` has no spectral signature in these indices, so class 6 is
    never predicted.
*   trees versus low vegetation is a *relative* ranking within the scene's own
    vegetation, not an absolute determination. There is no sensor-independent
    NDVI value that separates them.
*   SAR supports binary change detection through the log-ratio operator but
    carries no mapping to land-cover classes, so it yields **no semantic
    maps at all** rather than empty ones.

Two thresholds are in play and they do different jobs. Otsu supplies the
data-adaptive cut. A sign gate supplies the physical floor: a normalised
difference above zero means the numerator band genuinely exceeds the
denominator band, so NDVI at or below zero is definitionally not vegetation
whatever the scene's relative ranking says. Without that floor Otsu
manufactures every class in every scene, because it splits whatever
distribution it is handed -- measured on a bare-ground scene with NDVI
between -0.231 and -0.084, the unguarded version labelled 100 of 256 pixels
as vegetation. Zero is not a tuned value; it is where the inequality flips.

A bimodality gate was considered for the trees/low-vegetation and
buildings/ground sub-splits and rejected on measurement: Otsu's separability
(between-class over total variance) does not discriminate. Across 900
unimodal and 300 genuinely bimodal synthetic distributions the populations
overlap -- unimodal reaches 0.7753 while bimodal starts at 0.6955 -- so no
cut exists. The separability is recorded in the trace as an observation, and
must not be promoted into a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .detector import detect_change, otsu_threshold
from .indices import available_indices, index_by_name
from .io import RSImage

__all__ = [
    "IndexClassification",
    "PRECEDENCE",
    "WRITE_ORDER",
    "classify_pair",
]

# SECOND-compatible class IDs.
UNCHANGED = 0
NVG_SURFACE = 1
LOW_VEGETATION = 2
TREES = 3
BUILDINGS = 4
WATER = 5
PLAYGROUNDS = 6

# Assignment order. Later writes overwrite earlier ones, so the effective
# precedence is this reversed. Both the behaviour and the reported precedence
# derive from this one tuple: the defect being fixed here was a comment
# declaring NDWI > NDBI while the code applied NDBI > NDWI, which put open
# water into the buildings class. Two sources of truth drifted; now there is
# one.
WRITE_ORDER = ("ndvi", "ndbi", "ndwi")
PRECEDENCE = tuple(reversed(WRITE_ORDER))

# Physical floor per index. A normalised difference is positive only where
# the numerator band exceeds the denominator band, which is the definitional
# requirement for the feature: NIR above red for vegetation, green above NIR
# for water, SWIR above NIR for built-up. Not a tuned threshold.
SIGN_FLOOR = 0.0


@dataclass
class IndexClassification:
    """What Producer B can say about a pair, and what it cannot.

    ``s_t1``/``s_t2`` are SECOND-style semantic *change* maps: a pixel carries
    a class only where the class differs between dates, and 0 means unchanged.
    They are ``None`` when no semantic claim is possible -- the SAR path --
    because an all-zero map is a positive assertion that nothing changed, and
    the CDVQA rules cannot distinguish that from "unknown". They would answer
    "no" and "0" with full confidence while this producer's own trace recorded
    detected change.
    """

    s_t1: Optional[np.ndarray]
    s_t2: Optional[np.ndarray]
    classes_t1: Optional[np.ndarray]
    classes_t2: Optional[np.ndarray]
    change_mask: Optional[np.ndarray]
    semantic_available: bool
    trace: Dict[str, Any] = field(default_factory=dict)
    # Measured-or-zero, as everywhere else in this branch. This producer has
    # never been scored against reference labels, so it has no accuracy to
    # report and says so rather than offering a plausible-looking number.
    confidence: float = 0.0
    confidence_basis: str = (
        "none: physics-based index heuristic, not calibrated against "
        "reference labels"
    )

    def require_semantic(self):
        """Return ``(s_t1, s_t2)`` or raise if no semantic claim is possible.

        Callers that intend to apply the CDVQA rules must go through this, so
        a SAR result cannot reach them silently.
        """
        if not self.semantic_available or self.s_t1 is None:
            raise ValueError(
                "no semantic maps available for this input: "
                f"{self.trace.get('limitation', 'semantic classification not supported')}"
            )
        return self.s_t1, self.s_t2


def _separability(values: np.ndarray, threshold: float) -> Optional[float]:
    """Otsu separability: between-class over total variance.

    Recorded as an observation only. Measured across 900 unimodal and 300
    bimodal distributions it does not separate the two populations, so it is
    not usable as a gate and must not become one.
    """
    low, high = values[values <= threshold], values[values > threshold]
    if low.size == 0 or high.size == 0:
        return None
    total = float(values.var())
    if total <= 0:
        return None
    w0, w1 = low.size / values.size, high.size / values.size
    return float(w0 * w1 * (low.mean() - high.mean()) ** 2 / total)


def _elevated(values: np.ndarray, trace: Dict[str, Any], name: str) -> np.ndarray:
    """Pixels above both the Otsu cut and the physical sign floor."""
    finite = np.isfinite(values)
    threshold = otsu_threshold(values) if finite.any() else None
    trace["thresholds"][name] = threshold
    if threshold is None:
        return np.zeros(values.shape, dtype=bool)
    trace["separability"][name] = _separability(values[finite], threshold)
    # Otsu gives the data-adaptive cut; the sign floor keeps a scene with none
    # of this feature from producing some anyway.
    effective = max(float(threshold), SIGN_FLOOR)
    trace["effective_threshold"][name] = effective
    if effective > float(threshold):
        trace["ambiguity"].append(
            f"{name.upper()} Otsu cut {threshold:.4f} is at or below the sign "
            f"floor; raised to {effective:.4f} so no {name.upper()} feature is "
            f"claimed where the index is not positive"
        )
    return finite & (values > effective)


def _sub_split(
    values: np.ndarray,
    mask: np.ndarray,
    trace: Dict[str, Any],
    name: str,
    upper_class: int,
    lower_class: int,
    semantic: np.ndarray,
) -> None:
    """Split an elevated mask into a stronger and a weaker class.

    This is a relative ranking within the scene, not an absolute call: no
    sensor-independent NDVI value separates trees from low vegetation. When
    the distribution cannot be split the conservative (lower) class is used
    for all of it, and the trace says so.
    """
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    threshold = otsu_threshold(selected) if selected.size >= 2 else None
    trace["class_splits"][name] = threshold
    if threshold is None:
        semantic[mask] = lower_class
        if mask.any():
            trace["ambiguity"].append(
                f"{name}: distribution could not be split; assigned the "
                f"conservative class for all of it"
            )
        return
    trace["class_splits"][f"{name}_separability"] = _separability(selected, threshold)
    upper = mask & np.isfinite(values) & (values > threshold)
    semantic[mask & ~upper] = lower_class
    semantic[upper] = upper_class


def _classify_optical(image: RSImage) -> tuple:
    """Assign a pseudo land-cover class per pixel for one date."""
    height, width = image.array.shape[1:]
    semantic = np.zeros((height, width), dtype=np.uint8)

    available = available_indices(image)
    trace: Dict[str, Any] = {
        "indices": list(available),
        "thresholds": {},
        "effective_threshold": {},
        "separability": {},
        "class_splits": {},
        "ambiguity": [],
        "precedence": list(PRECEDENCE),
        "sign_floor": SIGN_FLOOR,
    }
    if not available:
        trace["ambiguity"].append(
            "no spectral index computable from the declared bands; no "
            "land-cover class assigned"
        )
        return semantic, trace

    index_maps = {name: index_by_name(image, name) for name in available}
    elevated = {
        name: _elevated(values, trace, name) for name, values in index_maps.items()
    }

    # Written in WRITE_ORDER so the effective precedence is PRECEDENCE.
    for name in WRITE_ORDER:
        if name not in index_maps:
            continue
        values, mask = index_maps[name], elevated[name]
        if name == "ndvi":
            _sub_split(values, mask, trace, "ndvi_trees_vs_low_vegetation",
                       TREES, LOW_VEGETATION, semantic)
        elif name == "ndbi":
            _sub_split(values, mask, trace, "ndbi_buildings_vs_ground",
                       BUILDINGS, NVG_SURFACE, semantic)
        elif name == "ndwi":
            semantic[mask] = WATER

    overlaps = sum(m.astype(np.uint8) for m in elevated.values())
    contested = int((overlaps > 1).sum()) if len(elevated) > 1 else 0
    trace["contested_pixels"] = contested
    if contested:
        trace["ambiguity"].append(
            f"{contested} pixels exceeded multiple index thresholds; "
            f"resolved by precedence {' > '.join(PRECEDENCE)}"
        )
    return semantic, trace


def classify_pair(t1: RSImage, t2: RSImage) -> IndexClassification:
    """Produce SECOND-compatible semantic change maps, or say why not."""
    if t1.array.shape[1:] != t2.array.shape[1:]:
        raise ValueError("T1 and T2 must have the same spatial shape")
    if t1.modality != t2.modality:
        raise ValueError("T1 and T2 must have the same modality")

    if t1.modality == "sar":
        detection = detect_change(t1, t2)
        return IndexClassification(
            s_t1=None, s_t2=None, classes_t1=None, classes_t2=None,
            change_mask=detection.mask, semantic_available=False,
            trace={
                "producer": "index_classifier",
                "indices": [],
                "sar_operator": detection.operator,
                "sar_threshold": detection.threshold,
                "sar_change_pixels": int(detection.mask.sum()),
                "note": "physics-based approximation, not a trained classifier",
                "limitation": (
                    "SAR backscatter supports binary change detection through "
                    "the log-ratio operator but carries no mapping to the six "
                    "SECOND land-cover classes, so no semantic map is produced"
                ),
            },
        )

    classes_t1, trace_t1 = _classify_optical(t1)
    classes_t2, trace_t2 = _classify_optical(t2)

    # SECOND semantics: a class is carried only where it differs between
    # dates; identical class at both dates is "unchanged".
    same = classes_t1 == classes_t2
    s_t1 = np.where(same, UNCHANGED, classes_t1).astype(np.uint8)
    s_t2 = np.where(same, UNCHANGED, classes_t2).astype(np.uint8)

    indices: List[str] = sorted(set(trace_t1["indices"]) | set(trace_t2["indices"]))
    return IndexClassification(
        s_t1=s_t1, s_t2=s_t2, classes_t1=classes_t1, classes_t2=classes_t2,
        change_mask=(s_t1 != UNCHANGED), semantic_available=bool(indices),
        trace={
            "producer": "index_classifier",
            "note": "physics-based approximation, not a trained classifier",
            "precedence": list(PRECEDENCE),
            "sign_floor": SIGN_FLOOR,
            "indices": indices,
            "never_predicted": ["playgrounds"],
            "approximation": (
                "trees versus low vegetation, and buildings versus ground, are "
                "relative rankings within this scene, not absolute determinations"
            ),
            "t1": trace_t1,
            "t2": trace_t2,
            "changed_pixels": int((s_t1 != UNCHANGED).sum()),
        },
    )
