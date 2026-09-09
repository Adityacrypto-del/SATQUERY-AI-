from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingResult:
    task: str
    rule: str


def classify_query(query: str) -> RoutingResult:
    q = (query or "").strip().lower()
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
