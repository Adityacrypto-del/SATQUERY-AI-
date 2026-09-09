from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChangeEvidence:
    """Structured evidence produced by bi-temporal analysis."""

    changed: bool
    summary: str
    change_type: Optional[str] = None
    confidence: float = 0.0
    changed_area_pixels: Optional[int] = None
    changed_area_m2: Optional[float] = None
    regions: List[Dict[str, Any]] = field(default_factory=list)
