"""High-level Qwen2.5-VL inference for remote-sensing images.

Provides ``analyze_single`` and ``analyze_bitemporal`` which
build the proper message format, call the model, and return
a structured ``QwenAnalysis``.
"""

import json
import logging
import re
import time
from typing import Optional

from .model import QwenVL
from .prompts import bitemporal_prompt, single_image_prompt
from .schemas import QwenAnalysis

logger = logging.getLogger(__name__)


def analyze_single(
    model: QwenVL,
    image_path: str,
    query: str,
) -> QwenAnalysis:
    """Analyze a single remote-sensing image.

    Parameters
    ----------
    model:
        A loaded ``QwenVL`` instance.

    image_path:
        Path to the image file (PNG, JPEG, TIFF, etc.).

    query:
        Natural-language question about the image.

    Returns
    -------
    QwenAnalysis
        Structured analysis result.
    """
    prompt = single_image_prompt(query)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    start = time.monotonic()
    raw = model.generate(messages)
    elapsed = time.monotonic() - start

    logger.info(
        "Single-image inference completed in %.2f s",
        elapsed,
    )

    return _parse_response(
        raw,
        model_name=model.model_name,
        latency_s=elapsed,
    )


def analyze_bitemporal(
    model: QwenVL,
    t1_path: str,
    t2_path: str,
    query: str,
) -> QwenAnalysis:
    """Analyze a bi-temporal image pair.

    Parameters
    ----------
    model:
        A loaded ``QwenVL`` instance.

    t1_path:
        Path to the **earlier** observation (T1).

    t2_path:
        Path to the **later** observation (T2).

    query:
        Natural-language question about temporal change.

    Returns
    -------
    QwenAnalysis
        Structured analysis result.
    """
    prompt = bitemporal_prompt(query)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{t1_path}",
                },
                {
                    "type": "image",
                    "image": f"file://{t2_path}",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    start = time.monotonic()
    raw = model.generate(messages)
    elapsed = time.monotonic() - start

    logger.info(
        "Bi-temporal inference completed in %.2f s",
        elapsed,
    )

    return _parse_response(
        raw,
        model_name=model.model_name,
        latency_s=elapsed,
    )


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

_CONFIDENCE_MAP = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.8,
    "very high": 0.9,
}


def _parse_response(
    raw: str,
    model_name: str,
    latency_s: Optional[float] = None,
) -> QwenAnalysis:
    """Parse a free-text Qwen response into ``QwenAnalysis``.

    The parser is intentionally lenient — Qwen may produce
    varied output formats. We extract what we can and preserve
    the full raw response for downstream inspection.
    """
    confidence = _extract_confidence(raw)

    return QwenAnalysis(
        observation=raw.strip(),
        possible_changes=_extract_list_section(raw, "changes"),
        objects=_extract_list_section(raw, "objects"),
        confidence=confidence,
        raw_response=raw,
        model_name=model_name,
        latency_s=latency_s,
    )


def _extract_confidence(text: str) -> float:
    """Try to extract a confidence level from model output."""
    lower = text.lower()

    for label, value in sorted(
        _CONFIDENCE_MAP.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    ):
        if label in lower:
            return value

    return 0.5


def _extract_list_section(text: str, keyword: str) -> list:
    """Extract bullet-point items near a keyword.

    This is a best-effort heuristic for parsing free-text
    model responses into structured lists.
    """
    items = []

    pattern = re.compile(
        rf"(?:^|\n)\s*[-•*\d.]+\s*(.+)",
        re.MULTILINE,
    )

    # Find the keyword location and extract nearby bullets.
    lower = text.lower()
    idx = lower.find(keyword.lower())

    if idx == -1:
        return items

    # Search within 500 chars after the keyword.
    region = text[idx: idx + 500]

    for match in pattern.finditer(region):
        item = match.group(1).strip()
        if item and len(item) > 2:
            items.append(item)

    return items
