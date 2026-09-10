"""
Training-free semantic change-map producer based on spectral indices.

Producer B is a fallback for remote-sensing inputs that the trained
SECOND segmentation model cannot reliably interpret, such as
multispectral images with arbitrary band counts or SAR imagery.

Optical classification uses the existing NDVI, NDWI and NDBI
implementations from ``indices.py`` and the existing Otsu threshold
implementation from ``detector.py``.

Important limitation:
``playgrounds`` have no reliable spectral signature in these indices,
so this producer never predicts class 6 (playgrounds).

This is a physics-based approximation, not a trained classifier.
"""

from typing import Any, Dict, Tuple

import numpy as np

from .detector import detect_change, otsu_threshold
from .indices import available_indices, index_by_name
from .io import RSImage


# SECOND-compatible class IDs.
UNCHANGED = 0
NVG_SURFACE = 1
LOW_VEGETATION = 2
TREES = 3
BUILDINGS = 4
WATER = 5
PLAYGROUNDS = 6


def _second_stage_threshold(values: np.ndarray) -> float:
    """
    Derive a second data-dependent threshold for splitting a class.

    Returns NaN when there are not enough valid values or when Otsu
    cannot produce a meaningful split.
    """
    finite = values[np.isfinite(values)]

    if finite.size < 2:
        return np.nan

    threshold = otsu_threshold(finite)

    if threshold is None:
        return np.nan

    return float(threshold)


def _classify_optical(
    image: RSImage,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Independently assign a pseudo land-cover class to each pixel.

    The returned map contains temporary land-cover classes. The
    T1/T2 comparison is performed separately by ``classify_pair``.
    """

    height, width = image.array.shape[1:]
    semantic = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    available = available_indices(image)

    trace: Dict[str, Any] = {
        "indices": list(available),
        "thresholds": {},
        "class_splits": {},
        "ambiguity": [],
        "precedence": ["ndwi", "ndbi", "ndvi"],
    }

    if not available:
        return semantic, trace

    index_maps: Dict[str, np.ndarray] = {}
    elevated_masks: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------
    # 1. Compute every index that is actually available.
    # ------------------------------------------------------------
    for name in available:
        values = index_by_name(image, name)

        index_maps[name] = values

        threshold = otsu_threshold(values)

        trace["thresholds"][name] = threshold

        if threshold is None:
            elevated_masks[name] = np.zeros(
                values.shape,
                dtype=bool,
            )
        else:
            elevated_masks[name] = (
                np.isfinite(values)
                & (values > threshold)
            )

    # ------------------------------------------------------------
    # 2. Build vegetation classification (lowest precedence).
    #
    # NDVI elevated pixels are vegetation.
    # A second Otsu split separates stronger vegetation (trees)
    # from moderate vegetation (low vegetation).
    #
    # Precedence: NDVI assignments may be overwritten by NDWI
    # (water) or NDBI (built-up) in later steps.
    # ------------------------------------------------------------
    if "ndvi" in index_maps:
        ndvi_values = index_maps["ndvi"]
        vegetation_mask = elevated_masks["ndvi"]

        vegetation_values = ndvi_values[vegetation_mask]
        tree_threshold = _second_stage_threshold(
            vegetation_values
        )

        trace["class_splits"]["ndvi_tree_threshold"] = (
            None
            if np.isnan(tree_threshold)
            else tree_threshold
        )

        if not np.isnan(tree_threshold):
            trees = (
                vegetation_mask
                & np.isfinite(ndvi_values)
                & (ndvi_values > tree_threshold)
            )

            low_vegetation = (
                vegetation_mask
                & ~trees
            )

            semantic[low_vegetation] = LOW_VEGETATION
            semantic[trees] = TREES
        else:
            # If the vegetation distribution cannot be split,
            # conservatively keep it as low vegetation.
            semantic[vegetation_mask] = LOW_VEGETATION

            if vegetation_mask.any():
                trace["ambiguity"].append(
                    "NDVI vegetation could not be split into "
                    "trees and low vegetation"
                )

    # ------------------------------------------------------------
    # 3. Water (highest precedence).
    #
    # Water is assigned where NDWI fires. Water takes precedence
    # over vegetation because open water has a distinct spectral
    # signature and should not be overwritten by another index.
    # ------------------------------------------------------------
    if "ndwi" in elevated_masks:
        water_mask = elevated_masks["ndwi"]

        semantic[water_mask] = WATER

    # ------------------------------------------------------------
    # 4. Built-up / non-vegetated ground (middle precedence).
    #
    # NDBI elevated pixels are split using a second Otsu threshold.
    # Stronger NDBI → buildings.
    # Lower elevated NDBI → non-vegetated ground surface.
    #
    # NDBI overwrites earlier NDVI assignments but is itself
    # overwritten by NDWI water in the step above (water was
    # assigned first, but NDBI writes after NDVI, so the
    # effective precedence is: NDWI > NDBI > NDVI).
    # ------------------------------------------------------------
    if "ndbi" in index_maps:
        ndbi_values = index_maps["ndbi"]
        built_mask = elevated_masks["ndbi"]

        built_values = ndbi_values[built_mask]
        building_threshold = _second_stage_threshold(
            built_values
        )

        trace["class_splits"]["ndbi_building_threshold"] = (
            None
            if np.isnan(building_threshold)
            else building_threshold
        )

        if not np.isnan(building_threshold):
            buildings = (
                built_mask
                & np.isfinite(ndbi_values)
                & (ndbi_values > building_threshold)
            )

            nvg_surface = (
                built_mask
                & ~buildings
            )

            semantic[nvg_surface] = NVG_SURFACE
            semantic[buildings] = BUILDINGS
        else:
            # If no meaningful split exists, use the conservative
            # non-vegetated-ground interpretation.
            semantic[built_mask] = NVG_SURFACE

            if built_mask.any():
                trace["ambiguity"].append(
                    "NDBI built-up pixels could not be split into "
                    "buildings and non-vegetated ground"
                )

    # ------------------------------------------------------------
    # 5. Resolve pixels where multiple indices fired.
    #
    # The effective precedence is:
    #     NDWI (water) > NDBI (built-up) > NDVI (vegetation)
    # This is enforced by assignment order: NDVI is written
    # first, then NDWI overwrites, then NDBI overwrites.
    # Record the existence of such conflicts for the trace.
    # ------------------------------------------------------------
    masks = [
        elevated_masks[name]
        for name in available
    ]

    if masks:
        fire_count = np.zeros(
            semantic.shape,
            dtype=np.uint8,
        )

        for mask in masks:
            fire_count += mask.astype(np.uint8)

        overlap = fire_count > 1

        if overlap.any():
            trace["ambiguity"].append(
                f"{int(overlap.sum())} pixels had multiple "
                "spectral indices above their thresholds"
            )

    # Class 6 is intentionally impossible here.
    semantic[semantic == PLAYGROUNDS] = UNCHANGED

    return semantic, trace


def _classify_sar(
    t1: RSImage,
    t2: RSImage,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run the existing SAR log-ratio detector.

    SAR has no reliable mapping from log-ratio magnitude to the
    six SECOND semantic land-cover classes, so the semantic maps
    remain zero. The binary change mask is preserved in the trace.
    """

    result = detect_change(t1, t2)

    height, width = result.mask.shape

    s_t1 = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    s_t2 = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    trace = {
        "indices": [],
        "sar_change_pixels": int(result.mask.sum()),
        "sar_threshold": result.threshold,
        "sar_operator": result.operator,
        "producer": "index_classifier",
        "note": "physics-based approximation, not a trained classifier",
        "confidence_type": "conservative_heuristic",
        "limitation": (
            "SAR-only input supports binary change detection through "
            "log-ratio, but does not provide reliable semantic class "
            "assignment."
        ),
    }

    return s_t1, s_t2, trace


def classify_pair(
    t1: RSImage,
    t2: RSImage,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Produce SECOND-compatible semantic change maps for T1 and T2.

    Each date is classified independently first. Pixels whose
    predicted class is identical in both dates are encoded as
    class 0 (unchanged). Pixels whose classes differ retain their
    respective T1/T2 classes.

    Returns
    -------
    s_t1:
        ``(H, W)`` uint8 semantic change map for T1.

    s_t2:
        ``(H, W)`` uint8 semantic change map for T2.

    trace:
        Metadata describing the producer, indices used, thresholds,
        ambiguity and confidence.
    """

    if t1.array.shape[1:] != t2.array.shape[1:]:
        raise ValueError(
            "T1 and T2 must have the same spatial shape"
        )

    if t1.modality != t2.modality:
        raise ValueError(
            "T1 and T2 must have the same modality"
        )

    # ------------------------------------------------------------
    # SAR path.
    # ------------------------------------------------------------
    if t1.modality == "sar":
        s_t1, s_t2, trace = _classify_sar(
            t1,
            t2,
        )

        # Conservative, non-calibrated confidence: this producer
        # is a physics-based heuristic, not a trained classifier.
        trace["confidence"] = 0.50

        return s_t1, s_t2, trace

    # ------------------------------------------------------------
    # Optical path.
    #
    # Classify each date independently.
    # ------------------------------------------------------------
    classes_t1, trace_t1 = _classify_optical(t1)
    classes_t2, trace_t2 = _classify_optical(t2)

    # Same semantic class at both dates means unchanged.
    same_class = classes_t1 == classes_t2

    s_t1 = classes_t1.copy()
    s_t2 = classes_t2.copy()

    s_t1[same_class] = UNCHANGED
    s_t2[same_class] = UNCHANGED

    # Explicitly guarantee SECOND's valid class range.
    s_t1 = np.clip(
        s_t1,
        0,
        6,
    ).astype(np.uint8)

    s_t2 = np.clip(
        s_t2,
        0,
        6,
    ).astype(np.uint8)

    trace = {
        "producer": "index_classifier",
        "note": "physics-based approximation, not a trained classifier",
        "confidence": 0.50,
        "confidence_type": "conservative_heuristic",
        "precedence": ["ndwi", "ndbi", "ndvi"],
        "indices": sorted(
            set(trace_t1["indices"])
            | set(trace_t2["indices"])
        ),
        "t1": trace_t1,
        "t2": trace_t2,
        "changed_pixels": int((s_t1 != 0).sum()),
    }

    return s_t1, s_t2, trace