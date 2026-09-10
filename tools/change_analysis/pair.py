from dataclasses import dataclass
from typing import List

from tools.change_analysis.io import RSImage


@dataclass
class PairValidationResult:
    """Result of validating two remote-sensing images."""

    valid: bool
    errors: List[str]
    warnings: List[str]


def validate_pair(
    t1: RSImage,
    t2: RSImage,
) -> PairValidationResult:
    """
    Validate whether two remote-sensing images can be
    compared for bi-temporal analysis.
    """

    errors = []
    warnings = []

    # 1. CRS compatibility
    if t1.crs != t2.crs:
        errors.append(
            "CRS mismatch between T1 and T2"
        )

    # 2. Image dimensions
    if t1.array.shape[1:] != t2.array.shape[1:]:
        errors.append(
            "Image dimensions do not match between T1 and T2"
        )

    # 3. Band count
    if t1.array.shape[0] != t2.array.shape[0]:
        errors.append(
            "Band count does not match between T1 and T2"
        )

    # 4. GSD compatibility
    if t1.gsd_m is not None and t2.gsd_m is not None:
        if t1.gsd_m != t2.gsd_m:
            errors.append(
                "GSD mismatch between T1 and T2"
            )

    # 5. Modality compatibility
    if (
        t1.modality != "unknown"
        and t2.modality != "unknown"
        and t1.modality != t2.modality
    ):
        errors.append(
            "Modality mismatch between T1 and T2"
        )

    return PairValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )