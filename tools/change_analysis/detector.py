"""Change detection and data-derived thresholding (build-order step 4).

Two operators, chosen by modality:

*   **Optical** -- difference the spectral index, not the raw DNs. Raw
    brightness moves with sun angle, atmosphere and sensor gain between
    acquisitions, so differencing it measures the acquisition as much as the
    ground.
*   **SAR** -- the ratio operator, ``I2 / (I1 + eps)``. Speckle is
    multiplicative, so the difference operator's false alarm rate rises with
    backscatter magnitude, whereas the ratio gives a constant false alarm
    rate. (Log-ratio vs plain ratio is contested in the literature -- same
    ROC. Ratio vs difference is not contested.) The log is taken so the
    magnitude is symmetric about no-change: a halving and a doubling then
    score equally.

The threshold is always Otsu, computed over valid pixels only. There is no
constant anywhere in this module, and when the histogram cannot be split the
threshold is None rather than a number nobody measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.filters import threshold_otsu

from .indices import available_indices, index_by_name
from .io import RSImage

__all__ = ["DetectionResult", "detect_change", "otsu_threshold"]

# Guards 0/0 in the SAR ratio. Small relative to any physical backscatter
# value, and its only role is to keep the operator finite.
_RATIO_EPS = 1e-6


@dataclass
class DetectionResult:
    """A change magnitude map, its mask, and how both were obtained."""

    change_map: np.ndarray
    mask: np.ndarray
    valid_mask: np.ndarray
    threshold: Optional[float]
    operator: str
    index_name: Optional[str]
    n_valid: int
    note: Optional[str] = None

    def as_trace_params(self) -> dict:
        """The subset a trace entry should record."""
        return {
            "operator": self.operator,
            "index": self.index_name,
            "threshold": self.threshold,
            "threshold_method": "otsu",
            "n_valid_pixels": self.n_valid,
            "n_changed_pixels": int(self.mask.sum()),
            "note": self.note,
        }


def otsu_threshold(values: np.ndarray) -> Optional[float]:
    """Otsu's threshold over the finite entries of ``values``.

    Returns None when the histogram cannot be split -- no valid pixels, or
    every valid pixel identical. scikit-image raises on all-NaN input rather
    than returning NaN, but ``np.mean`` and friends do propagate silently,
    so filtering here (not merely relying on the raise) is what keeps the
    whole chain honest.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if finite.min() == finite.max():
        return None
    return float(threshold_otsu(finite))


def _change_magnitude(
    t1: RSImage, t2: RSImage, index: Optional[str]
) -> tuple[np.ndarray, str, Optional[str]]:
    """Return ``(magnitude, operator, index_name)`` for the pair's modality."""
    if t1.modality == "sar":
        # Ratio, then log, so that x2 and x0.5 have equal magnitude.
        a = t1.array[0].astype(np.float32)
        b = t2.array[0].astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (b + _RATIO_EPS) / (a + _RATIO_EPS)
            magnitude = np.abs(np.log(ratio))
        magnitude = np.asarray(magnitude, dtype=np.float32)
        magnitude[~np.isfinite(magnitude)] = np.nan
        return magnitude, "log_ratio", None

    chosen = index
    if chosen is None:
        options = available_indices(t1)
        if not options:
            raise ValueError(
                "no optical index is computable from the declared bands "
                f"{t1.band_names!r}; pass index= explicitly or name the bands"
            )
        # NDVI first when available: it is the most broadly informative of
        # the three for land-cover change.
        chosen = "ndvi" if "ndvi" in options else options[0]

    magnitude = np.abs(index_by_name(t2, chosen) - index_by_name(t1, chosen))
    return magnitude.astype(np.float32), "index_difference", chosen


def detect_change(
    t1: RSImage, t2: RSImage, index: Optional[str] = None
) -> DetectionResult:
    """Detect change between two co-registered observations.

    Parameters
    ----------
    t1, t2:
        the earlier and later observations, same shape and same modality.
    index:
        optical index to difference. Defaults to the best available.

    Raises
    ------
    ValueError:
        shapes differ, modalities differ, or no index is computable.
    """
    if t1.shape != t2.shape:
        raise ValueError(
            f"t1 and t2 must have the same spatial shape; "
            f"got {t1.shape} and {t2.shape}"
        )
    if t1.modality != t2.modality:
        raise ValueError(
            f"t1 and t2 must share a modality; got {t1.modality!r} and "
            f"{t2.modality!r}. Comparing across modalities is not a change signal."
        )

    magnitude, operator, index_name = _change_magnitude(t1, t2, index)

    valid = t1.valid_mask & t2.valid_mask & np.isfinite(magnitude)
    n_valid = int(valid.sum())

    threshold = otsu_threshold(np.where(valid, magnitude, np.nan))

    note = None
    if threshold is None:
        # No split exists: either nothing is valid, or the magnitude is
        # uniform (which includes the identical-images case).
        mask = np.zeros(magnitude.shape, dtype=bool)
        note = (
            "no_valid_pixels" if n_valid == 0 else "degenerate_histogram_no_change"
        )
    else:
        mask = (magnitude > threshold) & valid

    return DetectionResult(
        change_map=magnitude,
        mask=mask,
        valid_mask=valid,
        threshold=threshold,
        operator=operator,
        index_name=index_name,
        n_valid=n_valid,
        note=note,
    )
