"""Bi-temporal pair compatibility validation (build-order step 2).

Returns a structured report, never a bare bool: "invalid" is not actionable,
whereas "CRS differs: EPSG:32643 vs EPSG:32644" tells the user what to fix.

This is primarily the GeoTIFF path shown to evaluators. CDVQA benchmark
images are ungeoreferenced PNGs, so every geospatial check degrades to
NOT_APPLICABLE for them rather than failing a pair that is perfectly usable.
Co-registration is the exception: it needs pixels, not coordinates, so it
runs on both paths.

Ayush's per-image validation in ``satquery.preprocessing.image_loader`` is
layered on as an *extra* check when importable. It is never a substitute --
this module owns the geospatial data path, and which checks actually ran is
recorded in the report so the trace is never ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from tools.change_analysis.io import RSImage

__all__ = [
    "CO_REGISTRATION_MIN_PSR",
    "CO_REGISTRATION_WARN_PX",
    "Check",
    "PairReport",
    "Severity",
    "estimate_shift_px",
    "validate_pair",
]

# Townshend (1992) and Dai & Khorram (1998): registration accuracy better
# than 0.2 pixels is required to keep change-detection error under 10%.
# Literature-derived, not tuned.
#
# IMPORTANT -- what this estimator can and cannot support. Phase correlation
# on a bi-temporal pair measures *apparent* displacement, which mixes true
# misregistration with genuine land-cover change. Measured on 40 real,
# genuinely co-registered SECOND pairs: median apparent shift 6.20 px, with
# 90% above 1 px. So a hard failure at 1 px would reject roughly nine in ten
# valid benchmark pairs. The literature figures describe true registration
# error; this estimator does not isolate it. Co-registration is therefore
# reported as ADVISORY -- it can warn, never fail.
CO_REGISTRATION_WARN_PX = 0.2

# Peak-to-sidelobe ratio floor below which the correlation peak is not
# distinguishable from the surface and the shift estimate means nothing.
# Derived, not chosen: synthetic pairs with no shared structure scored
# 4.77-6.55, while 40 real SECOND pairs scored 6.84 at minimum (p5 = 7.01).
# 6.7 sits in the gap between those two measured populations.
CO_REGISTRATION_MIN_PSR = 6.7

# Ground sample distances rarely match to the digit across sensors; beyond
# this ratio the pair is not comparable without resampling.
GSD_RATIO_TOLERANCE = 1.05


class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class Check:
    """One named check, its verdict, and the number behind it."""

    name: str
    severity: Severity
    detail: str
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "detail": self.detail,
            "value": self.value,
        }


@dataclass
class PairReport:
    """The full verdict on a bi-temporal pair."""

    checks: List[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no check failed. Warnings do not block analysis."""
        return not any(c.severity is Severity.FAIL for c in self.checks)

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.severity is Severity.WARN]

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.severity is Severity.FAIL]

    def check(self, name: str) -> Check:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(f"no check named {name!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "n_failures": len(self.failures),
            "n_warnings": len(self.warnings),
        }


def estimate_shift_px(
    a: np.ndarray, b: np.ndarray, return_confidence: bool = False
):
    """Residual translation between two planes, in pixels, via phase correlation.

    Phase correlation works in the Fourier domain, so it is insensitive to
    the overall brightness difference between two acquisition dates -- which
    matters here, since that difference is exactly what we are not trying to
    measure.
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    b = np.nan_to_num(np.asarray(b, dtype=np.float64))
    a = a - a.mean()
    b = b - b.mean()
    if not np.any(a) or not np.any(b):
        return (0.0, 0.0) if return_confidence else 0.0

    # Hann window suppresses the edge discontinuity that would otherwise
    # dominate the spectrum of a non-periodic image.
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    magnitude = np.abs(cross)
    magnitude[magnitude == 0] = 1e-12
    correlation = np.fft.ifft2(cross / magnitude).real

    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    shifts = []
    for axis, index in enumerate(peak):
        size = correlation.shape[axis]
        # Wrap: a peak near the end of the axis is a negative shift.
        shifts.append(index - size if index > size // 2 else index)
    distance = float(np.hypot(*shifts))
    if not return_confidence:
        return distance
    spread = float(correlation.std())
    psr = (
        (float(correlation.max()) - float(correlation.mean())) / spread
        if spread > 0 else 0.0
    )
    return distance, psr


def _co_registration_check(t1: RSImage, t2: RSImage) -> Check:
    """Advisory co-registration estimate. Warns, never fails.

    See CO_REGISTRATION_WARN_PX for why: the estimate conflates true
    misregistration with real land-cover change, so a large value is a
    reason to prefer region-level statistics, not grounds to reject a pair.
    """
    # Average across bands rather than taking band 0. Blue is often the
    # weakest, noisiest band, and averaging raises the shared-structure
    # signal the correlation depends on.
    a = np.nanmean(t1.array, axis=0)
    b = np.nanmean(t2.array, axis=0)
    shift, psr = estimate_shift_px(a, b, return_confidence=True)
    if psr < CO_REGISTRATION_MIN_PSR:
        return Check(
            "co_registration", Severity.NOT_APPLICABLE,
            f"correlation peak is indistinct (PSR {psr:.2f} < "
            f"{CO_REGISTRATION_MIN_PSR}); the images share too little structure "
            "for a shift estimate to mean anything, so none is reported",
            value=None,
        )
    if shift > CO_REGISTRATION_WARN_PX:
        return Check(
            "co_registration", Severity.WARN,
            f"apparent shift {shift:.2f} px (PSR {psr:.2f}) exceeds "
            f"{CO_REGISTRATION_WARN_PX} px. Literature (Townshend 1992; Dai & "
            "Khorram 1998) puts change-detection error above 10% beyond this, "
            "but this estimate also absorbs genuine land-cover change and "
            "cannot separate the two -- prefer region-level statistics",
            value=shift,
        )
    return Check(
        "co_registration", Severity.PASS,
        f"apparent shift {shift:.2f} px (PSR {psr:.2f}) is within "
        f"{CO_REGISTRATION_WARN_PX} px",
        value=shift,
    )


def _extents_overlap(t1: RSImage, t2: RSImage) -> bool:
    def bounds(image: RSImage):
        left, top = image.transform @ (0, 0)
        right, bottom = image.transform @ (image.width, image.height)
        return (min(left, right), min(top, bottom),
                max(left, right), max(top, bottom))

    a, b = bounds(t1), bounds(t2)
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _external_per_image_checks(paths: Optional[List[str]]) -> Check:
    """Layer Ayush's per-image validation on top, when it is importable."""
    if not paths:
        return Check(
            "per_image_external", Severity.NOT_APPLICABLE,
            "no source paths supplied, so external per-image checks were not run",
        )
    try:
        from satquery.preprocessing.image_loader import validate_single_image
    except Exception:
        return Check(
            "per_image_external", Severity.NOT_APPLICABLE,
            "satquery.preprocessing.image_loader is not importable on this "
            "branch; this module's own rasterio checks were used instead",
        )
    problems = []
    for path in paths:
        result = validate_single_image(path)
        if not result.get("ok"):
            problems.append(f"{path}: {result.get('reason')}")
    if problems:
        return Check(
            "per_image_external", Severity.FAIL, "; ".join(problems)
        )
    return Check(
        "per_image_external", Severity.PASS,
        f"external per-image validation passed for {len(paths)} image(s)",
    )


def validate_pair(
    t1: RSImage,
    t2: RSImage,
    t1_datetime: Optional[str] = None,
    t2_datetime: Optional[str] = None,
    source_paths: Optional[List[str]] = None,
) -> PairReport:
    """Validate that two observations can be compared.

    ``t1_datetime`` / ``t2_datetime`` are ISO-8601 strings where the metadata
    exists; without them the temporal-order check is NOT_APPLICABLE rather
    than silently assumed correct.
    """
    checks: List[Check] = []

    # -- shape ------------------------------------------------------------
    if t1.shape == t2.shape:
        checks.append(Check("shape", Severity.PASS, f"both {t1.shape}"))
    else:
        checks.append(Check(
            "shape", Severity.FAIL,
            f"spatial shapes differ: {t1.shape} vs {t2.shape}",
        ))

    # -- modality ---------------------------------------------------------
    if t1.modality == t2.modality:
        checks.append(Check("modality", Severity.PASS, f"both {t1.modality!r}"))
    else:
        checks.append(Check(
            "modality", Severity.FAIL,
            f"modalities differ: {t1.modality!r} vs {t2.modality!r}; "
            "an optical-versus-SAR difference is not a change signal",
        ))

    # -- band count -------------------------------------------------------
    if t1.n_bands == t2.n_bands:
        checks.append(Check("band_count", Severity.PASS, f"both {t1.n_bands}"))
    else:
        checks.append(Check(
            "band_count", Severity.FAIL,
            f"band counts differ: {t1.n_bands} vs {t2.n_bands}",
        ))

    # -- band semantics ---------------------------------------------------
    if t1.band_names and t2.band_names:
        if [n.lower() for n in t1.band_names] == [n.lower() for n in t2.band_names]:
            checks.append(Check("band_names", Severity.PASS, "band names match"))
        else:
            checks.append(Check(
                "band_names", Severity.WARN,
                f"band names differ: {t1.band_names} vs {t2.band_names}; "
                "index maths may compare different wavelengths",
            ))
    else:
        checks.append(Check(
            "band_names", Severity.NOT_APPLICABLE,
            "at least one image declares no band names",
        ))

    # -- georeferencing ---------------------------------------------------
    georeferenced = t1.crs is not None and t2.crs is not None
    if not georeferenced:
        reason = (
            "at least one image is ungeoreferenced (expected for CDVQA "
            "benchmark PNGs); geospatial checks do not apply"
        )
        checks.append(Check("crs", Severity.NOT_APPLICABLE, reason))
        checks.append(Check("geotransform", Severity.NOT_APPLICABLE, reason))
        checks.append(Check("gsd_ratio", Severity.NOT_APPLICABLE, reason))
    else:
        if t1.crs == t2.crs:
            checks.append(Check("crs", Severity.PASS, f"both {t1.crs}"))
        else:
            checks.append(Check(
                "crs", Severity.FAIL,
                f"CRS differs: {t1.crs} vs {t2.crs}; reproject before analysis",
            ))

        if t1.transform == t2.transform:
            checks.append(Check(
                "geotransform", Severity.PASS, "geotransforms are identical"
            ))
        elif _extents_overlap(t1, t2):
            checks.append(Check(
                "geotransform", Severity.WARN,
                "geotransforms differ but extents overlap; pixels are not "
                "co-located and the pair needs resampling onto a common grid",
            ))
        else:
            checks.append(Check(
                "geotransform", Severity.FAIL,
                "extents do not overlap; these images do not cover the same ground",
            ))

        if t1.gsd_m and t2.gsd_m:
            ratio = max(t1.gsd_m, t2.gsd_m) / min(t1.gsd_m, t2.gsd_m)
            if ratio <= GSD_RATIO_TOLERANCE:
                checks.append(Check(
                    "gsd_ratio", Severity.PASS,
                    f"GSD {t1.gsd_m:.3f} m vs {t2.gsd_m:.3f} m (ratio {ratio:.3f})",
                    value=ratio,
                ))
            else:
                checks.append(Check(
                    "gsd_ratio", Severity.FAIL,
                    f"GSD ratio {ratio:.3f} exceeds {GSD_RATIO_TOLERANCE}: "
                    f"{t1.gsd_m:.3f} m vs {t2.gsd_m:.3f} m; resample first",
                    value=ratio,
                ))
        else:
            checks.append(Check(
                "gsd_ratio", Severity.NOT_APPLICABLE,
                "ground sample distance is not derivable for at least one image",
            ))

    # -- temporal order ---------------------------------------------------
    if t1_datetime and t2_datetime:
        if str(t1_datetime) < str(t2_datetime):
            checks.append(Check(
                "temporal_order", Severity.PASS,
                f"t1 {t1_datetime} precedes t2 {t2_datetime}",
            ))
        else:
            checks.append(Check(
                "temporal_order", Severity.FAIL,
                f"t1 {t1_datetime} does not precede t2 {t2_datetime}; "
                "every increase/decrease answer would be inverted",
            ))
    else:
        checks.append(Check(
            "temporal_order", Severity.NOT_APPLICABLE,
            "acquisition datetimes were not supplied",
        ))

    # -- co-registration --------------------------------------------------
    if t1.shape == t2.shape:
        checks.append(_co_registration_check(t1, t2))
    else:
        checks.append(Check(
            "co_registration", Severity.NOT_APPLICABLE,
            "shapes differ, so a residual shift is not meaningful",
        ))

    checks.append(_external_per_image_checks(source_paths))
    return PairReport(checks=checks)
