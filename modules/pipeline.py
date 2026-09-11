"""SatQuery pipeline orchestrator.

Ties together the independent Qwen VLM analysis, the specialist
pipeline (DeltaVLM or deterministic fallback), evidence fusion,
and the final Qwen reasoning pass into a single entry point.

Architecture::

                  T1 + T2 + QUERY
                        │
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
      Qwen2.5-VL-3B            Specialist Pipeline
      Independent VLM          DeltaVLM / deterministic
           │                         │
           ▼                         ▼
     QwenAnalysis              ChangeEvidence
           │                         │
           └───────────┬─────────────┘
                       ▼
                 Evidence Fusion
                       │
                       ▼
               Final Qwen Reasoner
                       │
                       ▼
                 Final Response
"""

import logging
from typing import Optional

from modules.bi_temporal.adapter import DeltaVLMAdapter
from modules.bi_temporal.schemas import ChangeEvidence
from modules.fusion.fuse import fuse_evidence
from modules.qwen.inference import analyze_bitemporal, analyze_single
from modules.qwen.model import QwenVL
from modules.qwen.reasoner import final_reasoning
from modules.qwen.schemas import FinalResponse

logger = logging.getLogger(__name__)


def run_satquery(
    query: str,
    t1_path: str,
    t2_path: Optional[str] = None,
    qwen_model: Optional[QwenVL] = None,
    deltavlm_adapter: Optional[DeltaVLMAdapter] = None,
) -> FinalResponse:
    """Run the full SatQuery pipeline.

    Parameters
    ----------
    query:
        The user's natural-language question.

    t1_path:
        Path to the first (or only) image.

    t2_path:
        Path to the second image for bi-temporal analysis.
        ``None`` for single-image queries.

    qwen_model:
        Pre-loaded ``QwenVL`` instance. If ``None``, a new
        instance is created and loaded.

    deltavlm_adapter:
        Pre-configured ``DeltaVLMAdapter``. If ``None``, a new
        instance is created (which will report unavailability
        on non-CUDA environments).

    Returns
    -------
    FinalResponse
        The final grounded answer with evidence.
    """
    # ----------------------------------------------------------
    # 1. Ensure Qwen is loaded.
    # ----------------------------------------------------------
    if qwen_model is None:
        qwen_model = QwenVL()

    if not qwen_model.is_loaded:
        logger.info("Loading Qwen model…")
        qwen_model.load()

    # ----------------------------------------------------------
    # 2. Independent Qwen analysis.
    # ----------------------------------------------------------
    if t2_path is not None:
        logger.info("Running bi-temporal Qwen analysis…")
        qwen_result = analyze_bitemporal(
            model=qwen_model,
            t1_path=t1_path,
            t2_path=t2_path,
            query=query,
        )
    else:
        logger.info("Running single-image Qwen analysis…")
        qwen_result = analyze_single(
            model=qwen_model,
            image_path=t1_path,
            query=query,
        )

    # ----------------------------------------------------------
    # 3. Specialist pipeline.
    # ----------------------------------------------------------
    specialist_evidence: Optional[ChangeEvidence] = None

    if t2_path is not None:
        if deltavlm_adapter is None:
            deltavlm_adapter = DeltaVLMAdapter()

        if deltavlm_adapter.is_available:
            try:
                if deltavlm_adapter.model is None:
                    deltavlm_adapter.load_model()

                specialist_evidence = deltavlm_adapter.analyze(
                    image_t1=t1_path,
                    image_t2=t2_path,
                    query=query,
                )
            except RuntimeError as exc:
                logger.warning(
                    "Specialist pipeline failed: %s", exc,
                )
        else:
            logger.info(
                "DeltaVLM not available on this environment. "
                "Proceeding with Qwen analysis only."
            )

    # ----------------------------------------------------------
    # 4. Evidence fusion.
    # ----------------------------------------------------------
    fused = fuse_evidence(
        qwen=qwen_result,
        specialist=specialist_evidence,
    )

    logger.info(
        "Evidence fused: %d agreements, %d disagreements, "
        "specialist_available=%s",
        len(fused.agreements),
        len(fused.disagreements),
        fused.specialist_available,
    )

    # ----------------------------------------------------------
    # 5. Final Qwen reasoning.
    # ----------------------------------------------------------
    images = [t1_path]
    if t2_path is not None:
        images.append(t2_path)

    logger.info("Running final Qwen reasoning…")

    response = final_reasoning(
        model=qwen_model,
        query=query,
        fused=fused,
        images=images,
    )

    logger.info(
        "Pipeline complete. Confidence=%.3f, "
        "limitations=%d",
        response.confidence,
        len(response.limitations),
    )

    return response
