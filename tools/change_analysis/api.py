"""The branch-2 public interface. Everything a controller needs is here.

One call in, one evidence object out::

    from tools.change_analysis.api import BiTemporalSpecialist

    specialist = BiTemporalSpecialist(checkpoint="...")
    evidence = specialist.analyze("t1.tif", "t2.tif", "Did buildings increase?")

The controller should not have to construct an ``RSImage``, choose a route,
know that a checkpoint exists, or understand which producer answered. Before
this module it had to do all four, which meant the "black box" leaked its
internals into every integration.

:func:`describe_tool` returns the registry descriptor: what this specialist
is called, which input configurations it accepts, what it returns, and --
just as importantly -- what it will refuse. A controller that reads the
limitations can route around them instead of discovering them at runtime.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .io import RSImage, load_rsimage
from .pipeline import BiTemporalPipeline, PipelineConfig

__all__ = ["BiTemporalSpecialist", "describe_tool", "analyze"]

TOOL_NAME = "bitemporal_change_analysis"

ImageLike = Union[str, RSImage, np.ndarray]


def describe_tool(checkpoint: Optional[str] = None) -> Dict[str, Any]:
    """Registry descriptor for the agentic controller.

    Limitations are declared, not discovered. Every entry below is something
    measured on this branch rather than assumed, so a controller can decide
    in advance whether to route a query here.
    """
    return {
        "name": TOOL_NAME,
        "task": "bi-temporal change analysis",
        "summary": (
            "Answers change questions about two spatially corresponding "
            "images of the same area at different dates, and returns a change "
            "map, per-region attributes, visual evidence and an execution "
            "trace."
        ),
        "accepts": {
            "images": "exactly two, spatially corresponding, same modality",
            "formats": ["GeoTIFF", "TIFF", "PNG", "JPEG"],
            "modalities": ["optical", "multispectral", "SAR"],
            "query": "natural language; optional (descriptive output without one)",
        },
        "returns": {
            "answer": "one of CDVQA's 19 answer tokens, or null where no rule applies",
            "question_type": "one of the eight CDVQA types, or null",
            "summary": "natural-language change description, always present",
            "changed": "boolean",
            "changed_area_pixels": "integer",
            "changed_area_m2": "float, or null when the input is ungeoreferenced",
            "regions": "per-region pixel count, area, centroid, bbox, spectral signature",
            "confidence": "measured accuracy for this question type and this "
                          "image's decision margin; 0.0 where no measured basis exists",
            "confidence_basis": "why the confidence is what it is, always populated",
            "overlays": "paths to rendered visual evidence",
            "trace": "one entry per stage: tool, params, observation, duration_ms, why",
            "route": "which internal producer answered",
        },
        "limitations": [
            "SAR input yields binary change detection only: backscatter carries "
            "no mapping to land-cover classes, so class questions are refused "
            "rather than answered.",
            "The trained segmenter is a 3-band RGB model trained on SECOND, "
            "which is sub-metre aerial imagery. It does NOT transfer to "
            "medium-resolution satellite imagery: on a real Sentinel-2 pair "
            "at 10 m it detected no change at all while an independent "
            "index-based producer detected change across 34% of the scene. "
            "For such input the deterministic index path is the usable one, "
            "and the reported confidence drops to 0.0 with that stated.",
            "Input outside the training distribution is detected, not "
            "guessed: every optical pair with computable spectral indices is "
            "cross-checked against a physics-based producer, and a flat "
            "contradiction between them zeroes the confidence rather than "
            "letting a calibrated-looking number stand.",
            "Rare land-cover classes are under-predicted: on validation, water "
            "is predicted at 0.22x its true frequency and playgrounds are never "
            "predicted. Questions about those classes are answered with "
            "correspondingly low, measured confidence.",
            "Co-registration is reported but advisory: phase correlation on a "
            "bi-temporal pair cannot separate misregistration from real change.",
            "Compound questions naming several classes are answered per class; "
            "no single combined answer token exists in the benchmark.",
        ],
        "measured_accuracy": {
            "benchmark": "CDVQA Test (968 scenes, 39,686 questions)",
            "average_accuracy": 0.6814,
            "overall_accuracy": 0.7489,
            "caveat": (
                "1.3% of Test scenes are near-duplicates of training scenes; "
                "bounding their effect puts the true figure at or above 0.677."
            ),
        },
        "checkpoint": checkpoint,
    }


class BiTemporalSpecialist:
    """Branch-2 specialist, ready for a tool registry.

    Construct once and reuse: loading the checkpoint costs about four seconds
    and each subsequent query runs in a few hundred milliseconds, so a
    per-call constructor would dominate the latency.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        overlay_dir: Optional[str] = "outputs/evidence",
        device: Optional[str] = None,
        **config_overrides: Any,
    ) -> None:
        # Overlays default ON. Visual evidence is a required deliverable, and
        # a caller who does not know to ask for it should still get it.
        self.config = PipelineConfig(
            checkpoint=checkpoint, overlay_dir=overlay_dir, device=device,
            **config_overrides,
        )
        self._pipeline = BiTemporalPipeline(self.config)

    def describe(self) -> Dict[str, Any]:
        return describe_tool(self.config.checkpoint)

    def analyze(
        self,
        t1: ImageLike,
        t2: ImageLike,
        query: str = "",
        scene_id: Optional[str] = None,
    ):
        """Analyse a pair and return a ``BiTemporalEvidence``.

        Accepts file paths, already-loaded ``RSImage`` objects, or raw arrays.
        Paths are the expected case: the controller hands over what the user
        uploaded and nothing else.
        """
        image_1 = _coerce(t1)
        image_2 = _coerce(t2)
        if scene_id is None and isinstance(t1, str):
            scene_id = os.path.splitext(os.path.basename(t1))[0]
        result = self._pipeline.run(image_1, image_2, query, scene_id=scene_id)
        return result.to_change_evidence()

    def analyze_result(self, t1: ImageLike, t2: ImageLike, query: str = "",
                       scene_id: Optional[str] = None):
        """As :meth:`analyze` but returns the full internal ``ChangeResult``.

        For callers inside this branch that want the class maps themselves.
        """
        return self._pipeline.run(_coerce(t1), _coerce(t2), query, scene_id=scene_id)


def analyze(t1: ImageLike, t2: ImageLike, query: str = "",
            checkpoint: Optional[str] = None, **kwargs):
    """One-shot convenience wrapper. Builds a specialist and runs one query.

    Prefer :class:`BiTemporalSpecialist` for more than a single call; this
    reloads the checkpoint every time.
    """
    return BiTemporalSpecialist(checkpoint=checkpoint, **kwargs).analyze(t1, t2, query)


def _coerce(image: ImageLike) -> RSImage:
    """Accept what a controller is likely to have, refuse what it cannot mean."""
    if isinstance(image, RSImage):
        return image
    if isinstance(image, str):
        return load_rsimage(image)
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = array[None]
        elif array.ndim == 3 and array.shape[-1] in (3, 4) and array.shape[0] > 4:
            # Channels-last, as PIL and OpenCV produce.
            array = np.transpose(array, (2, 0, 1))
        elif array.ndim != 3:
            raise ValueError(
                f"expected a 2-D or 3-D array; got shape {image.shape}"
            )
        names = ["red", "green", "blue"] if array.shape[0] == 3 else None
        return RSImage(
            array=array.astype(np.float32), crs=None, transform=None,
            modality="optical", band_names=names, gsd_m=None,
        )
    raise TypeError(
        f"cannot interpret {type(image).__name__} as an image; pass a file "
        f"path, an RSImage, or a numpy array"
    )
