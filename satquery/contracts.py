"""Branch 1 handover contract: the structured evidence a single-image query produces.

A controller consumes this, so the shape is the promise. Two rules it encodes:

1. Nothing is invented. A field that cannot be computed is None or empty -- never a
   plausible-looking placeholder. ``measurements`` holds only deterministic, physically derived
   numbers (spectral indices), so a fusion layer can treat them as authoritative over a
   generalist VLM's estimates. A VLM's free text never lands there.
2. Provenance is explicit. ``model`` names the checkpoint, ``is_vlm_answer`` says whether a
   language model or a deterministic tool produced the answer, and ``confidence_method`` says how
   the confidence was derived (the current one is a text heuristic, not a calibrated probability).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# status values
OK = "ok"                   # an answer was produced
REFUSED = "refused"         # the question cannot be answered from this input (stated reason)
UNAVAILABLE = "unavailable"  # the backend could not run (stated reason)
ERROR = "error"             # invalid input or an unexpected failure (stated reason)

TASKS = ("vqa", "captioning", "spectral_index")


@dataclass
class SingleImageEvidence:
    task: str
    status: str
    answer: Optional[str] = None
    model: str = ""
    is_vlm_answer: bool = False
    confidence: Optional[float] = None
    confidence_method: Optional[str] = None
    measurements: Dict[str, float] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}; got {self.task!r}")
        if self.status not in (OK, REFUSED, UNAVAILABLE, ERROR):
            raise ValueError(f"unknown status {self.status!r}")
        if self.status != OK and self.answer is not None:
            raise ValueError(f"status {self.status!r} must not carry an answer")
        if self.status != OK and not self.reason:
            raise ValueError(f"status {self.status!r} requires a reason")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
