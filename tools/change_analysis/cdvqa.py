"""CDVQA question parsing and the answer rules (build-order step 6).

This is the single implementation of the rules. ``eval.oracle_cdvqa`` imports
from here rather than carrying its own copy, so the oracle always measures
what the pipeline actually does -- two copies would drift and the evaluation
would quietly stop describing the tool.

**SECOND's label1/label2 are semantic CHANGE maps, not land-cover maps.**
Class 0 is "unchanged" and is identical in both (verified at exactly 100%
agreement across all 4,662 SECOND pairs). A land-cover class exists only on
pixels that changed: label1 says what a pixel was, label2 what it became.
Everywhere below, "area of class l" means area of l *among changed pixels*.

Two rule details were established empirically against gold answers rather
than assumed, and both are load-bearing:

1.  A question's image reference scopes which map answers it. "in the
    pre-change image" -> s_t1, "in the second image" -> s_t2, no mention ->
    both. About a third of change_or_not / largest_change / smallest_change
    questions name no image; defaulting those to s_t1 costs roughly 40
    points on smallest_change and 33 on largest_change.
2.  ``change_ratio`` asks about unchanged area as often as changed area
    ("What is the percentage of unchanged areas?"), which inverts the answer.

Every answer carries the intermediate numbers it was derived from, so the
execution trace can show the arithmetic rather than assert a conclusion.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CDVQAAnswer",
    "CLASS_NAMES",
    "NAME_TO_CLASS",
    "PALETTE",
    "answer_question",
    "class_areas",
    "decode_label",
    "parse_question",
    "question_side",
    "ratio_bin",
]

# SECOND's 7-colour palette. Confirmed by the oracle test: change_to_what,
# increase_or_not, decrease_or_not and change_ratio_types all reach exactly
# 100% agreement with gold, which a permuted binding cannot do.
PALETTE: Dict[Tuple[int, int, int], int] = {
    (255, 255, 255): 0,  # unchanged
    (128, 128, 128): 1,  # non-vegetated ground surface
    (0, 128, 0): 2,      # low vegetation
    (0, 255, 0): 3,      # trees
    (128, 0, 0): 4,      # buildings
    (0, 0, 255): 5,      # water
    (255, 0, 0): 6,      # playgrounds
}

CLASS_NAMES: Dict[int, str] = {
    1: "NVG_surface",
    2: "low_vegetation",
    3: "trees",
    4: "buildings",
    5: "water",
    6: "playgrounds",
}

NAME_TO_CLASS: Dict[str, int] = {name: index for index, name in CLASS_NAMES.items()}

N_CLASSES = 7  # 0 = unchanged, plus the six land-cover classes

# Phrases used in question text for each class. Longest first so
# "low vegetation" is not shadowed by a shorter alternative.
_CLASS_PHRASES: List[Tuple[str, str]] = [
    ("non-vegetated ground surface", "NVG_surface"),
    ("non vegetated ground surface", "NVG_surface"),
    ("low vegetation", "low_vegetation"),
    ("playgrounds", "playgrounds"),
    ("playground", "playgrounds"),
    ("buildings", "buildings"),
    ("building", "buildings"),
    ("trees", "trees"),
    ("tree", "trees"),
    ("water", "water"),
]

_NON_CHANGE = re.compile(
    r"non-change|non change|nonchange|unchanged|not changed|has not changed"
)
_POST = re.compile(r"post-event|post event|post-change|post change|second image")
_PRE = re.compile(r"pre-event|pre event|pre-change|pre change|first image")


@dataclass
class CDVQAAnswer:
    """One answer plus the intermediate numbers it was derived from."""

    answer: Optional[str]
    question_type: str
    target_class: Optional[str] = None
    side: str = "none"
    evidence: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


def decode_label(path: str) -> np.ndarray:
    """Decode a SECOND RGB label PNG into an ``(H, W)`` class-index map.

    Raises on any colour outside the palette. An unmapped colour would
    silently become a wrong class and corrupt every area count downstream,
    so this fails loudly rather than guessing a nearest neighbour.
    """
    from PIL import Image  # local import: the rules do not need PIL

    rgb = np.array(Image.open(path).convert("RGB"))
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for colour, index in PALETTE.items():
        out[np.all(rgb == np.array(colour, dtype=np.uint8), axis=-1)] = index
    if (out == 255).any():
        bad = np.unique(rgb[out == 255].reshape(-1, 3), axis=0)[:5]
        raise ValueError(
            f"{path!r} contains colours outside the SECOND palette: "
            f"{[tuple(int(v) for v in c) for c in bad]}"
        )
    return out


def ratio_bin(percent: float) -> str:
    """Quantise a percentage into CDVQA's 11 bins.

    Exactly zero is its own answer ("0"); everything else falls in a
    half-open decade: (0, 10] -> "0_to_10", (90, 100] -> "90_to_100".
    """
    if percent <= 0:
        return "0"
    for low in range(0, 100, 10):
        if percent <= low + 10:
            return f"{low}_to_{low + 10}"
    return "90_to_100"


def question_side(text: str) -> str:
    """Which temporal map a question refers to: "pre", "post" or "none"."""
    lowered = text.lower()
    if _POST.search(lowered):
        return "post"
    if _PRE.search(lowered):
        return "pre"
    return "none"


def parse_question(text: str, question_type: str = "") -> Dict[str, Any]:
    """Extract the target land-cover class, temporal side, and polarity."""
    lowered = text.lower()
    target: Optional[str] = None
    for phrase, name in _CLASS_PHRASES:
        if phrase in lowered:
            target = name
            break
    return {
        "target": target,
        "side": question_side(text),
        "non_change": bool(_NON_CHANGE.search(lowered)),
        "question_type": question_type,
    }


def class_areas(semantic: np.ndarray) -> Dict[int, int]:
    """Changed-pixel count per land-cover class."""
    return {c: int((semantic == c).sum()) for c in CLASS_NAMES}


def answer_question(
    question: str,
    question_type: str,
    s_t1: np.ndarray,
    s_t2: np.ndarray,
) -> CDVQAAnswer:
    """Apply the CDVQA rule for ``question_type`` to two semantic change maps.

    Returns a :class:`CDVQAAnswer` whose ``answer`` is one of the 19 CDVQA
    strings, or None when the rule cannot be applied -- an unrecognised
    question type, no class named, or a scene with no changed pixels. None is
    deliberate: fabricating a class label to avoid a blank is exactly the
    failure HARD RULE 3 forbids.
    """
    parsed = parse_question(question, question_type)
    target = parsed["target"]
    side = parsed["side"]
    cls = NAME_TO_CLASS.get(target) if target else None

    a1 = class_areas(s_t1)
    a2 = class_areas(s_t2)
    total = int(s_t1.size)
    changed = int((s_t1 != 0).sum())

    def _result(answer: Optional[str], reason: Optional[str] = None, **evidence):
        return CDVQAAnswer(
            answer=answer,
            question_type=question_type,
            target_class=target,
            side=side,
            evidence={
                "total_pixels": total,
                "changed_pixels": changed,
                **evidence,
            },
            reason=reason,
        )

    if question_type in ("change_or_not", "increase_or_not", "decrease_or_not",
                         "change_to_what", "change_ratio_types") and cls is None:
        return _result(None, reason="no_class_named_in_question")

    if question_type == "change_or_not":
        if side == "post":
            present = a2[cls] > 0
        elif side == "pre":
            present = a1[cls] > 0
        else:
            present = a1[cls] > 0 or a2[cls] > 0
        return _result(
            "yes" if present else "no",
            area_t1=a1[cls], area_t2=a2[cls],
        )

    if question_type == "increase_or_not":
        return _result(
            "yes" if a2[cls] > a1[cls] else "no",
            area_t1=a1[cls], area_t2=a2[cls], delta=a2[cls] - a1[cls],
        )

    if question_type == "decrease_or_not":
        return _result(
            "yes" if a2[cls] < a1[cls] else "no",
            area_t1=a1[cls], area_t2=a2[cls], delta=a2[cls] - a1[cls],
        )

    if question_type in ("largest_change", "smallest_change"):
        if side == "post":
            totals = dict(a2)
        elif side == "pre":
            totals = dict(a1)
        else:
            totals = {c: a1[c] + a2[c] for c in CLASS_NAMES}
        # A class with no changed pixels did not undergo the smallest change,
        # it underwent none -- so absent classes are not candidates.
        present = {c: v for c, v in totals.items() if v > 0}
        if not present:
            return _result(
                None, reason="no_changed_pixels_in_scene", areas=totals
            )
        pick = max if question_type == "largest_change" else min
        best = pick(present, key=lambda c: present[c])
        return _result(
            CLASS_NAMES[best],
            areas={CLASS_NAMES[c]: v for c, v in present.items()},
        )

    if question_type == "change_to_what":
        mask = s_t1 == cls
        if not mask.any():
            return _result(None, reason="class_absent_at_t1", area_t1=0)
        destinations = s_t2[mask]
        destinations = destinations[destinations != 0]
        if destinations.size == 0:
            return _result(None, reason="no_destination_class", area_t1=int(mask.sum()))
        counts = Counter(destinations.tolist())
        winner = int(counts.most_common(1)[0][0])
        return _result(
            CLASS_NAMES[winner],
            area_t1=int(mask.sum()),
            destinations={CLASS_NAMES[int(k)]: int(v) for k, v in counts.items()},
        )

    if question_type == "change_ratio":
        percent = 100.0 * changed / total if total else 0.0
        if parsed["non_change"]:
            percent = 100.0 - percent
        return _result(
            ratio_bin(percent), percent=percent, polarity=(
                "non_change" if parsed["non_change"] else "change"
            ),
        )

    if question_type == "change_ratio_types":
        area = a2[cls] if side == "post" else a1[cls]
        percent = 100.0 * area / total if total else 0.0
        return _result(ratio_bin(percent), area=area, percent=percent)

    return _result(None, reason=f"unsupported_question_type:{question_type}")
