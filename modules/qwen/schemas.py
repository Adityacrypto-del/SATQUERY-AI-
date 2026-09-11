"""Structured output schemas for Qwen VLM analysis.

These dataclasses follow the same convention as
``modules.bi_temporal.schemas`` and are designed to integrate
cleanly with the Evidence Fusion layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QwenAnalysis:
    """Structured analysis produced by Qwen2.5-VL.

    Attributes
    ----------
    observation:
        Free-text description of what the model sees.

    possible_changes:
        List of changes detected or suspected.

    objects:
        Identified objects or land-cover features.

    confidence:
        Self-assessed confidence in [0, 1].

    raw_response:
        Unprocessed model output for debugging.

    model_name:
        Which model produced this analysis.

    latency_s:
        Wall-clock inference time in seconds.
    """

    observation: str
    possible_changes: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    latency_s: Optional[float] = None


@dataclass
class FinalResponse:
    """Structured response from the final Qwen reasoning pass.

    Attributes
    ----------
    answer:
        Direct answer to the user's question.

    confidence:
        Overall confidence in [0, 1].

    supporting_evidence:
        Evidence items that support the answer.

    disagreements:
        Cases where Qwen and specialist evidence conflict.

    limitations:
        Known limitations of this response.

    raw_response:
        Unprocessed model output for debugging.
    """

    answer: str
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    raw_response: str = ""
