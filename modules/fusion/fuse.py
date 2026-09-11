"""Evidence Fusion: merge Qwen VLM analysis with specialist evidence.

The fusion layer is intentionally simple. It preserves all
specialist quantitative measurements as authoritative and
records where the two sources agree or disagree.

The final Qwen reasoner receives ``FusedEvidence`` and
synthesises a grounded answer.
"""

from typing import Optional

from modules.bi_temporal.schemas import ChangeEvidence
from modules.qwen.schemas import QwenAnalysis

from .schemas import FusedEvidence


def fuse_evidence(
    qwen: QwenAnalysis,
    specialist: Optional[ChangeEvidence] = None,
) -> FusedEvidence:
    """Merge independent Qwen analysis with specialist evidence.

    Parameters
    ----------
    qwen:
        Structured analysis from the independent Qwen VLM pass.

    specialist:
        Quantitative evidence from the specialist pipeline.
        ``None`` if the specialist was unavailable.

    Returns
    -------
    FusedEvidence
        Merged evidence with agreements, disagreements, and
        authoritative measurements extracted.
    """
    if specialist is None:
        return FusedEvidence(
            qwen_analysis=qwen,
            specialist_evidence=None,
            specialist_available=False,
            agreements=[],
            disagreements=[],
            authoritative_measurements={},
        )

    # ----------------------------------------------------------
    # Extract authoritative measurements from specialist.
    # ----------------------------------------------------------
    measurements = {}

    if specialist.changed_area_pixels is not None:
        measurements["changed_pixels"] = specialist.changed_area_pixels

    if specialist.changed_area_m2 is not None:
        measurements["changed_area_m2"] = specialist.changed_area_m2

    if specialist.confidence > 0:
        measurements["specialist_confidence"] = specialist.confidence

    if specialist.change_type is not None:
        measurements["change_type"] = specialist.change_type

    if specialist.regions:
        measurements["region_count"] = len(specialist.regions)

    # ----------------------------------------------------------
    # Detect agreements and disagreements.
    # ----------------------------------------------------------
    agreements = []
    disagreements = []

    # Change / no-change agreement.
    qwen_sees_change = bool(qwen.possible_changes)

    if qwen_sees_change and specialist.changed:
        agreements.append(
            "Both Qwen and specialist detect change."
        )
    elif not qwen_sees_change and not specialist.changed:
        agreements.append(
            "Both Qwen and specialist agree: no change detected."
        )
    elif qwen_sees_change and not specialist.changed:
        disagreements.append(
            "Qwen reports possible changes but specialist "
            "found no change."
        )
    elif not qwen_sees_change and specialist.changed:
        disagreements.append(
            "Specialist detected change but Qwen did not "
            "report any."
        )

    # Confidence divergence.
    if specialist.confidence > 0 and qwen.confidence > 0:
        gap = abs(specialist.confidence - qwen.confidence)

        if gap > 0.3:
            disagreements.append(
                f"Confidence divergence: Qwen={qwen.confidence:.2f}, "
                f"specialist={specialist.confidence:.2f}."
            )
        else:
            agreements.append(
                "Confidence levels are broadly consistent."
            )

    return FusedEvidence(
        qwen_analysis=qwen,
        specialist_evidence=specialist,
        specialist_available=True,
        agreements=agreements,
        disagreements=disagreements,
        authoritative_measurements=measurements,
    )


def format_measurements(fused: FusedEvidence) -> str:
    """Format authoritative measurements for the final prompt.

    Returns a human-readable string suitable for inclusion
    in the final reasoning prompt.
    """
    if not fused.authoritative_measurements:
        return "No specialist measurements available."

    lines = []

    for key, value in fused.authoritative_measurements.items():
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def format_specialist_summary(fused: FusedEvidence) -> str:
    """Format specialist evidence summary for the final prompt."""
    if not fused.specialist_available or fused.specialist_evidence is None:
        return "Specialist pipeline was not available."

    parts = [f"Summary: {fused.specialist_evidence.summary}"]

    if fused.agreements:
        parts.append("Agreements:")
        for a in fused.agreements:
            parts.append(f"  - {a}")

    if fused.disagreements:
        parts.append("Disagreements:")
        for d in fused.disagreements:
            parts.append(f"  - {d}")

    return "\n".join(parts)
