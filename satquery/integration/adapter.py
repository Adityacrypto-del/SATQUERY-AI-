"""SingleImageAdapter: Branch 1's evidence provider for a controller or an HTTP layer.

Deliberately the same shape as a specialist adapter (``is_available`` / ``load_model`` /
``analyze``) so a controller can call it the same way it calls the bi-temporal one.

Routing, in order:
  1. spectral queries (NDVI/NDWI/NDBI) -> the deterministic tool. The VLM is never consulted, and
     its numbers are the only ones that reach ``measurements`` (authoritative for a fusion layer).
     It refuses on RGB-only input rather than guessing.
  2. everything else -> the VLM backend, as captioning or VQA.

Every failure path returns evidence carrying a status and a reason. Nothing is ever invented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from satquery.agent.single_image_router import classify_query
from satquery.contracts import ERROR, OK, REFUSED, UNAVAILABLE, SingleImageEvidence

CAPTION_PROMPT = "[caption] Please provide a detailed description of the image"
VQA_PROMPT = "[vqa] {query}"

RGB_LIMITATION = ("The VLM sees an RGB rendering only; NIR/SWIR/SAR content is not passed to it. "
                  "Spectral questions go to the deterministic tool instead.")
HEURISTIC_LIMITATION = ("Confidence is a text heuristic, not a calibrated probability "
                        "(see rs_vlm._heuristic_confidence).")


class GenerationBackend(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str: ...


class SingleImageAdapter:
    """Single-image specialist. Construct with any backend implementing GenerationBackend."""

    def __init__(self, backend: Optional[GenerationBackend] = None):
        if backend is None:
            from satquery.backends.terraq import TerraQVLBackend

            backend = TerraQVLBackend()
        self.backend = backend

    # -- availability -----------------------------------------------------

    @property
    def is_available(self) -> bool:
        return bool(self.backend.is_available())

    @property
    def unavailable_reason(self) -> str:
        return getattr(self.backend, "unavailable_reason", None) or (
            f"backend {self.backend.name!r} reported itself unavailable"
        )

    def load_model(self) -> None:
        load = getattr(self.backend, "load", None)
        if load is not None:
            load()

    def describe(self) -> dict:
        return {
            "specialist": "single_image",
            "backend": self.backend.name,
            "available": self.is_available,
            "tasks": ["vqa", "captioning", "spectral_index"],
            "measured_results": "outputs/reference_eval/eval_log.md",
        }

    # -- analysis ---------------------------------------------------------

    def analyze(self, image_path: str, query: str) -> SingleImageEvidence:
        route = classify_query(query)
        task = route.task if route.task in ("vqa", "captioning", "spectral_index") else "vqa"
        trace = [f"query_classified:{route.task}:{route.rule}"]
        meta = {"filename": Path(image_path).name, "path": str(image_path)}

        if not Path(image_path).is_file():
            return SingleImageEvidence(
                task=task, status=ERROR, reason=f"image not found: {image_path}",
                source_metadata=meta, execution_trace=trace + ["input_validation_failed"],
            )

        if task == "spectral_index":
            return self._spectral(image_path, route.index, meta, trace)
        return self._vlm(image_path, query, task, meta, trace)

    # -- paths ------------------------------------------------------------

    def _spectral(self, image_path: str, index: str, meta: dict, trace: list) -> SingleImageEvidence:
        from satquery.tools.spectral_index import compute_spectral_index

        result = compute_spectral_index(image_path, index=index)
        trace = trace + [f"spectral_index_tool:{result.index}:{result.status}"]
        if result.status != OK:
            return SingleImageEvidence(
                task="spectral_index", status=result.status, reason=result.reason,
                model=result.trace["index_source"], is_vlm_answer=False,
                source_metadata=meta, execution_trace=trace, limitations=[RGB_LIMITATION],
            )
        return SingleImageEvidence(
            task="spectral_index", status=OK, answer=result.answer,
            model=result.trace["index_source"], is_vlm_answer=False,
            measurements=dict(result.statistics),
            source_metadata={**meta, "bands_used": result.trace["bands_used"]},
            execution_trace=trace,
            limitations=["Index statistics describe the scene; they are not a land-cover classification."],
        )

    def _vlm(self, image_path: str, query: str, task: str, meta: dict, trace: list) -> SingleImageEvidence:
        if not self.is_available:
            return SingleImageEvidence(
                task=task, status=UNAVAILABLE, reason=self.unavailable_reason,
                model=self.backend.name, source_metadata=meta,
                execution_trace=trace + ["backend_unavailable"],
            )
        prompt = CAPTION_PROMPT if task == "captioning" else VQA_PROMPT.format(query=query)
        try:
            text = self.backend.generate(image_path, prompt)
        except Exception as exc:  # report, never fabricate
            return SingleImageEvidence(
                task=task, status=ERROR, reason=f"{type(exc).__name__}: {exc}",
                model=self.backend.name, source_metadata=meta,
                execution_trace=trace + ["generation_failed"],
            )
        if not text or not text.strip():
            return SingleImageEvidence(
                task=task, status=REFUSED, reason="the model returned an empty response",
                model=self.backend.name, is_vlm_answer=True, source_metadata=meta,
                execution_trace=trace + ["empty_response"],
            )

        from satquery.models.rs_vlm import _heuristic_confidence

        confidence, method = _heuristic_confidence(text)
        return SingleImageEvidence(
            task=task, status=OK, answer=text.strip(), model=self.backend.name, is_vlm_answer=True,
            confidence=confidence, confidence_method=method, source_metadata=meta,
            execution_trace=trace + [f"model:{self.backend.name}", f"{task}_inference_complete"],
            limitations=[RGB_LIMITATION, HEURISTIC_LIMITATION],
        )
