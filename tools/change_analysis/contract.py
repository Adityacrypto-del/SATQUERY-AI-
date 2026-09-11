"""The branch-2 handoff contract.

``ChangeEvidence`` in ``modules/bi_temporal/schemas.py`` is a shared file this
branch must not edit. It declares seven fields: changed, summary, change_type,
confidence, changed_area_pixels, changed_area_m2, regions.

That is not enough to carry what the problem statement actually grades. It
asks for "an auditable execution summary containing the selected task,
model/tool names, and key parameters", and for "visual evidence, confidence
information, execution summaries, and downloadable reports". The eight-stage
trace, the answer token, the overlay paths and the basis behind the confidence
number are all produced internally and would be dropped at the boundary.

So rather than edit a shared contract -- which would need Aditya's agreement
and would conflict on merge -- this extends it. ``BiTemporalEvidence``
subclasses ``ChangeEvidence`` and adds the missing fields, all defaulted.
Every consumer that accepts a ``ChangeEvidence`` accepts this too, by
substitution; consumers that know about the extra fields get them. Nothing
downstream has to change, and nothing upstream breaks.

The clean end state is still those fields living in the shared schema. This
makes that a tidy-up rather than a blocker: the diff is additive and every
field is defaulted, so it can land whenever Aditya has time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from modules.bi_temporal.schemas import ChangeEvidence

__all__ = ["BiTemporalEvidence"]


@dataclass
class BiTemporalEvidence(ChangeEvidence):
    """Shared evidence plus what the problem statement grades.

    Substitutable for :class:`ChangeEvidence` anywhere: every added field has
    a default, so ``BiTemporalEvidence(changed=..., summary=...)`` constructs
    exactly as the parent does.
    """

    # The CDVQA answer token, when a rule applied. None where none did --
    # including compound questions, where no single token is defined.
    answer: Optional[str] = None
    # Which of the eight CDVQA types was recognised, or None for descriptive
    # output. The controller needs this to know what kind of answer it holds.
    question_type: Optional[str] = None
    # Why the confidence number is what it is, including when it is 0.0.
    # A bare zero is indistinguishable from "very unsure"; the basis says
    # which, and that distinction is the whole point of measured-or-zero.
    confidence_basis: str = ""
    # The graded artefact: one entry per stage, each with tool, params,
    # observation, duration_ms and why.
    trace: List[Dict[str, Any]] = field(default_factory=list)
    # Paths to rendered visual evidence.
    overlays: Dict[str, str] = field(default_factory=dict)
    # "semantic" or "index" -- which producer answered, so the controller can
    # report which specialist ran.
    route: Optional[str] = None
    # Per-class answers when the question named more than one land-cover
    # class. Empty for single-class questions.
    compound_answers: Dict[str, Optional[str]] = field(default_factory=dict)
