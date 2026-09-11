from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RoutingResult:
    task: str
    rule: str
    index: str | None = None


# Queries that need reflectance an RGB render does not carry. These go to the
# physics-based spectral tool, never to the VLM. Matched on word boundaries,
# longest pattern first so the rule names the most specific match.
SPECTRAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "ndvi": (
        "ndvi",
        "normalized difference vegetation index",
        "normalised difference vegetation index",
        "vegetation index",
    ),
    "ndwi": (
        "ndwi",
        "normalized difference water index",
        "normalised difference water index",
        "water index",
    ),
    "ndbi": (
        "ndbi",
        "normalized difference built-up index",
        "normalised difference built-up index",
        "built-up index",
        "built up index",
    ),
}

_SPECTRAL_ORDER = sorted(
    ((p, idx) for idx, ps in SPECTRAL_PATTERNS.items() for p in ps),
    key=lambda item: -len(item[0]),
)


def classify_query(query: str) -> RoutingResult:
    q = (query or "").strip().lower()

    for pattern, index in _SPECTRAL_ORDER:
        if re.search(rf"\b{re.escape(pattern)}\b", q):
            return RoutingResult(task="spectral_index", rule=f"spectral:{pattern}", index=index)

    caption_keywords = [
        "describe",
        "scene description",
        "caption",
        "summarize the image",
        "what do you see",
    ]
    for key in caption_keywords:
        if key in q:
            return RoutingResult(task="captioning", rule=f"keyword:{key}")

    # Default to VQA for explicit question answering prompts.
    return RoutingResult(task="vqa", rule="default_vqa")
