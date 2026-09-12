"""HTTP-facing service: one analysis request -> a plain dict.

``app/backend_server.py`` calls this. It is a thin shell over ``SingleImageAdapter`` so the HTTP
layer holds no analysis logic and has nothing to embellish.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from satquery.integration.adapter import SingleImageAdapter

_adapter: Optional[SingleImageAdapter] = None


def get_adapter() -> SingleImageAdapter:
    """Process-wide adapter, so a backend loads at most once."""
    global _adapter
    if _adapter is None:
        _adapter = SingleImageAdapter()
    return _adapter


def analyze(image_path: str, query: str) -> Dict[str, Any]:
    """Analyse one image. Raises nothing: failures come back as status + reason."""
    return get_adapter().analyze(image_path, query).to_dict()
