"""
satquery/agent/query_router.py — Multi-modal Query Router for SatQuery AI.

Implements the central Query Router from the project architecture:
                     USER
                       │
                 Natural Language
                       │
                       ▼
                 QUERY ROUTER
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   SINGLE IMAGE    BI-TEMPORAL    OPTICAL + SAR
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class QueryRoutingDecision:
    mode: str  # 'single_image' | 'bitemporal' | 'optical_sar'
    confidence: float
    reasoning: str
    rule_matched: str
    suggested_task: str  # 'vqa' | 'captioning' | 'change_detection' | 'cross_modal_fusion'
    tokens_expected: List[str] = field(default_factory=list)


# Keywords mapped to modes
BITEMPORAL_KEYWORDS = [
    "before", "after", "change", "changed", "difference", "delta",
    "timeline", "temporal", "evolution", "growth", "inundation",
    "deforestation", "between", "pre", "post", "progression",
    "compare", "expansion", "year over year", "historical", "damage assessment"
]

OPTICAL_SAR_KEYWORDS = [
    "sar", "radar", "optical + sar", "optical and sar", "sentinel-1",
    "sentinel-2", "backscatter", "penetrate", "penetration", "cloud cover",
    "all-weather", "roughness", "dielectric", "polarization", "vv", "vh",
    "cross-modal", "fusion", "synthetic aperture"
]

CAPTIONING_KEYWORDS = [
    "describe", "caption", "overview", "summary", "summarize",
    "what is this", "scene description", "tell me about"
]


def route_query_and_inputs(
    query: str,
    explicit_mode: Optional[str] = None,
    has_t1_t2: bool = False,
    has_optical_sar: bool = False,
    image_count: int = 1,
) -> QueryRoutingDecision:
    """
    Intelligently routes the user query and uploaded assets to the appropriate specialist branch.
    
    If explicit_mode is specified and not 'auto', it respects the user's manual selection.
    Otherwise, it analyzes modal inputs (filenames/counts) and semantic NLP triggers in the query.
    """
    q = (query or "").strip().lower()
    
    # 1. Manual User Override
    if explicit_mode and explicit_mode != "auto":
        mode_clean = explicit_mode.lower().replace("-", "_")
        if mode_clean in ["single", "single_image"]:
            return QueryRoutingDecision(
                mode="single_image",
                confidence=1.0,
                reasoning="Manual override selected: Single Image Specialist",
                rule_matched="manual_override:single_image",
                suggested_task="captioning" if any(k in q for k in CAPTIONING_KEYWORDS) else "vqa",
                tokens_expected=["Features", "Objects", "Regions", "Land Cover Distribution"]
            )
        elif mode_clean in ["bitemporal", "bi_temporal", "temporal", "change"]:
            return QueryRoutingDecision(
                mode="bitemporal",
                confidence=1.0,
                reasoning="Manual override selected: Bi-Temporal Specialist",
                rule_matched="manual_override:bitemporal",
                suggested_task="change_detection",
                tokens_expected=["Change tokens", "T1/T2 features", "Change map", "Area Delta"]
            )
        elif mode_clean in ["optical_sar", "opticalsar", "fusion"]:
            return QueryRoutingDecision(
                mode="optical_sar",
                confidence=1.0,
                reasoning="Manual override selected: Optical + SAR Specialist",
                rule_matched="manual_override:optical_sar",
                suggested_task="cross_modal_fusion",
                tokens_expected=["Fused tokens", "Optical tokens", "SAR tokens", "Radar Backscatter"]
            )

    # 2. Check Explicit Image Modal Inputs
    if has_optical_sar:
        return QueryRoutingDecision(
            mode="optical_sar",
            confidence=0.99,
            reasoning="Optical (Sentinel-2) + SAR (Sentinel-1) image pair provided.",
            rule_matched="inputs:optical_sar_pair",
            suggested_task="cross_modal_fusion",
            tokens_expected=["Fused tokens (64)", "Optical tokens (64)", "SAR tokens (64)"]
        )
    
    if has_t1_t2 or image_count >= 2:
        return QueryRoutingDecision(
            mode="bitemporal",
            confidence=0.98,
            reasoning="Pre-event (T1) and Post-event (T2) temporal image pair provided.",
            rule_matched="inputs:bitemporal_pair",
            suggested_task="change_detection",
            tokens_expected=["Change tokens", "T1/T2 features", "Difference heatmap"]
        )

    # 3. NLP Analysis of Query Text
    # Check for Optical+SAR / Radar signals
    for kw in OPTICAL_SAR_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', q):
            return QueryRoutingDecision(
                mode="optical_sar",
                confidence=0.95,
                reasoning=f"Query contains radar/optical fusion keyword '{kw}'.",
                rule_matched=f"nlp_keyword:{kw}",
                suggested_task="cross_modal_fusion",
                tokens_expected=["Fused tokens", "Optical tokens", "SAR tokens"]
            )

    # Check for Temporal / Change signals
    for kw in BITEMPORAL_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', q):
            return QueryRoutingDecision(
                mode="bitemporal",
                confidence=0.93,
                reasoning=f"Query contains temporal change keyword '{kw}'.",
                rule_matched=f"nlp_keyword:{kw}",
                suggested_task="change_detection",
                tokens_expected=["Change tokens", "T1/T2 features", "Change map"]
            )

    # 4. Default to Single Image Specialist
    is_cap = any(k in q for k in CAPTIONING_KEYWORDS)
    return QueryRoutingDecision(
        mode="single_image",
        confidence=0.91,
        reasoning="Single scene inquiry detected. Routing to Single Image Specialist.",
        rule_matched="default_single_image",
        suggested_task="captioning" if is_cap else "vqa",
        tokens_expected=["Features", "Objects", "Regions", "Land Cover Distribution"]
    )
