"""ChangeResult -- the complete result object produced by this branch.

This is our contract, owned end to end. ``modules.bi_temporal.schemas``
.ChangeEvidence is the shared, cross-team contract and is never edited from
here; integration is exactly one call, :meth:`ChangeResult.to_change_evidence`.

Two properties matter more than the field list:

*   **Import independence.** ChangeEvidence is imported lazily, inside the
    method that needs it. If schemas.py moves, is renamed, or changes shape on
    Aditya's side, this module still imports and its tests still run. A
    top-level import would couple our whole pipeline to a file we do not own.
*   **Field tolerance in both directions.** The downgrade copies whichever
    fields the shared dataclass actually declares. Today it declares no
    ``answer`` and no ``trace``, so those are dropped. If they are added later,
    they are carried through with no edit here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["ChangeResult"]


# ``eq=False``: two of these fields are numpy arrays, and the dataclass-generated
# __eq__ would raise "truth value of an array is ambiguous" on comparison.
# Identity comparison is the honest default for a result object this large.
@dataclass(eq=False)
class ChangeResult:
    """Everything one bi-temporal analysis produced, including its trace.

    Attributes
    ----------
    answer:
        For a CDVQA question, one of the 19 answer strings. For an
        out-of-scope query, the region-attribute summary. None when no
        question was asked.
    trace:
        Ordered stage records. The PS grades the observable execution trace,
        so this is an output in its own right, not plumbing.
    semantic_t1, semantic_t2:
        ``(H, W)`` int class maps, or None when segmentation did not run.
        Deliberately excluded from the shared contract -- they are large and
        Aditya's schema has no place for them.
    georeferencing:
        Provenance from ``io.georeferencing_report``, so a reader can tell
        whether an area in m2 came from a projected grid or a geodesic
        estimate.
    """

    answer: Optional[str]
    trace: List[Dict[str, Any]]
    changed: bool
    summary: str
    change_type: Optional[str]
    confidence: float
    changed_area_pixels: Optional[int]
    changed_area_m2: Optional[float]
    regions: List[Dict[str, Any]]
    semantic_t1: Optional[np.ndarray]
    semantic_t2: Optional[np.ndarray]
    georeferencing: Dict[str, Any]

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1]; got {self.confidence!r}"
            )

    @classmethod
    def empty(cls, summary: str) -> "ChangeResult":
        """A result with nothing measured behind it.

        Every optional field is None rather than zero: HARD RULE 3 forbids
        placeholder numbers standing in for uncomputed values.
        """
        return cls(
            answer=None,
            trace=[],
            changed=False,
            summary=summary,
            change_type=None,
            confidence=0.0,
            changed_area_pixels=None,
            changed_area_m2=None,
            regions=[],
            semantic_t1=None,
            semantic_t2=None,
            georeferencing={},
        )

    def to_change_evidence(self):
        """Downgrade into the shared contract. Integration is this one call.

        The import is lazy on purpose -- see the module docstring. Only the
        fields the shared dataclass actually declares are copied, so this
        adapter needs no edit whether or not ``answer`` and ``trace`` are
        added to it later.

        Raises
        ------
        ImportError:
            the shared schema is unavailable. Callers that must keep working
            without it should use the ChangeResult directly.
        """
        from modules.bi_temporal.schemas import ChangeEvidence

        accepted = {
            f.name
            for f in dataclasses.fields(ChangeEvidence)
            if f.init
        }
        payload = {
            name: getattr(self, name)
            for name in accepted
            if hasattr(self, name)
        }
        return ChangeEvidence(**payload)
