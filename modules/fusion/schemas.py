"""Schemas for the Evidence Fusion layer.

``FusedEvidence`` combines the independent Qwen VLM analysis
with specialist pipeline evidence into a single structure
that the final reasoner can consume.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modules.bi_temporal.schemas import ChangeEvidence
from modules.qwen.schemas import QwenAnalysis


@dataclass
class FusedEvidence:
    """Merged evidence from Qwen and the specialist pipeline.

    Attributes
    ----------
    qwen_analysis:
        Independent visual analysis from Qwen2.5-VL.

    specialist_evidence:
        Quantitative evidence from the specialist pipeline
        (DeltaVLM or deterministic fallback). ``None`` when
        the specialist is unavailable.

    agreements:
        Points where Qwen and specialist agree.

    disagreements:
        Points where Qwen and specialist conflict.

    authoritative_measurements:
        Specialist-sourced quantitative measurements that
        must not be overridden by Qwen estimates.

    specialist_available:
        Whether the specialist pipeline actually ran.
    """

    qwen_analysis: QwenAnalysis
    specialist_evidence: Optional[ChangeEvidence] = None
    agreements: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)
    authoritative_measurements: Dict[str, Any] = field(
        default_factory=dict,
    )
    specialist_available: bool = False
