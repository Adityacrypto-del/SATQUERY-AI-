"""Final Qwen reasoning pass.

Receives the fused evidence (Qwen + specialist) and produces
a grounded ``FinalResponse``. The prompt explicitly instructs
the model to use specialist measurements as authoritative
and to not invent quantities.
"""

import logging
import time
from typing import Optional

from modules.fusion.fuse import format_measurements, format_specialist_summary
from modules.fusion.schemas import FusedEvidence

from .model import QwenVL
from .prompts import final_reasoning_prompt
from .schemas import FinalResponse

logger = logging.getLogger(__name__)


def final_reasoning(
    model: QwenVL,
    query: str,
    fused: FusedEvidence,
    images: Optional[list] = None,
) -> FinalResponse:
    """Run the final Qwen reasoning pass over fused evidence.

    Parameters
    ----------
    model:
        A loaded ``QwenVL`` instance.

    query:
        The user's original question.

    fused:
        Merged evidence from the fusion layer.

    images:
        Optional list of image paths to include as visual
        context for the final reasoning.

    Returns
    -------
    FinalResponse
        Structured final answer with evidence and limitations.
    """
    prompt = final_reasoning_prompt(
        query=query,
        qwen_observation=fused.qwen_analysis.observation,
        specialist_summary=format_specialist_summary(fused),
        measurements=format_measurements(fused),
    )

    # Build message content.
    content = []

    if images:
        for img_path in images:
            content.append({
                "type": "image",
                "image": f"file://{img_path}",
            })

    content.append({
        "type": "text",
        "text": prompt,
    })

    messages = [{"role": "user", "content": content}]

    start = time.monotonic()
    raw = model.generate(messages)
    elapsed = time.monotonic() - start

    logger.info(
        "Final reasoning completed in %.2f s",
        elapsed,
    )

    return _parse_final_response(
        raw=raw,
        fused=fused,
    )


def _parse_final_response(
    raw: str,
    fused: FusedEvidence,
) -> FinalResponse:
    """Parse the final reasoning output into ``FinalResponse``.

    Combines the raw model output with metadata from the
    fused evidence.
    """
    # Carry forward disagreements from fusion.
    disagreements = list(fused.disagreements)

    # Build limitations.
    limitations = []

    if not fused.specialist_available:
        limitations.append(
            "Specialist pipeline was not available; "
            "answer relies on Qwen visual analysis only."
        )

    # Derive confidence: average of Qwen and specialist if both
    # are available; otherwise use Qwen's confidence alone.
    if (
        fused.specialist_available
        and fused.specialist_evidence is not None
        and fused.specialist_evidence.confidence > 0
    ):
        confidence = (
            fused.qwen_analysis.confidence
            + fused.specialist_evidence.confidence
        ) / 2.0
    else:
        confidence = fused.qwen_analysis.confidence

    # Supporting evidence from both sources.
    supporting = []

    if fused.qwen_analysis.observation:
        supporting.append(
            f"Qwen observation: {fused.qwen_analysis.observation[:200]}"
        )

    if fused.specialist_evidence is not None:
        supporting.append(
            f"Specialist: {fused.specialist_evidence.summary}"
        )

    for key, val in fused.authoritative_measurements.items():
        supporting.append(f"{key}: {val}")

    return FinalResponse(
        answer=raw.strip(),
        confidence=round(confidence, 3),
        supporting_evidence=supporting,
        disagreements=disagreements,
        limitations=limitations,
        raw_response=raw,
    )
