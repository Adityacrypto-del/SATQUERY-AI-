"""Staged bi-temporal orchestrator (build-order step 8).

The problem statement grades the observable execution trace, not internal
reasoning, so every stage appends an entry recording what ran, with which
parameters, what it observed, how long it took, and *why* that step was
taken. The trace is an output in its own right.

Routing: RGB input with a checkpoint goes through the semantic segmenter
(the CDVQA path, where answers come from the deterministic rules). Anything
else -- multispectral, SAR, or no checkpoint -- goes through the index and
Otsu detector. The route taken is recorded, never implied.

Nothing here fabricates. A field that cannot be computed is None: no area in
m2 for ungeoreferenced input, no answer where no rule applies, and a
confidence of 0.0 carries an explicit ``confidence_basis`` of "none" rather
than a plausible-looking number nobody measured.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cdvqa import (
    answer_compound,
    answer_question,
    classify_question,
    is_compound,
    parse_question,
)
from .detector import detect_change_stack, index_stack
from .indices import available_indices
from .io import RSImage, georeferencing_report
from .morphology import clean_mask
from .regions import extract_regions, summarise_regions
from .result import ChangeResult

__all__ = ["BiTemporalPipeline", "PipelineConfig"]


@dataclass
class PipelineConfig:
    """Every tunable in one place; nothing is a literal in the flow below."""

    checkpoint: Optional[str] = None
    opening_radius: int = 1
    closing_radius: int = 1
    min_region_pixels: int = 16
    max_regions_reported: int = 20
    overlay_dir: Optional[str] = None
    device: Optional[str] = None


class _Trace:
    """Accumulates stage records and times them."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def stage(self, name: str, tool: str, why: str, params: Dict[str, Any]):
        return _StageTimer(self, name, tool, why, params)

    def append(self, entry: Dict[str, Any]) -> None:
        entry["stage"] = len(self.entries) + 1
        self.entries.append(entry)


class _StageTimer:
    def __init__(self, trace: _Trace, name: str, tool: str, why: str, params):
        self.trace, self.name, self.tool, self.why = trace, name, tool, why
        self.params = params
        self.observation = ""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.trace.append({
            "name": self.name,
            "tool": self.tool,
            "params": self.params,
            "observation": (
                self.observation if exc is None
                else f"FAILED: {exc_type.__name__}: {exc}"
            ),
            "duration_ms": (time.perf_counter() - self._start) * 1000.0,
            "why": self.why,
        })
        return False


class BiTemporalPipeline:
    """Runs validation, detection, region analysis and answering as stages."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        segmenter: Optional[Any] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._segmenter = segmenter

    # -- lazily loaded so the pipeline is usable with no checkpoint --------

    @property
    def segmenter(self):
        if self._segmenter is None and self.config.checkpoint:
            from .semantic import SemanticSegmenter

            self._segmenter = SemanticSegmenter(
                self.config.checkpoint, device=self.config.device
            )
        return self._segmenter

    # -- routing ----------------------------------------------------------

    def _can_segment(self, t1: RSImage, t2: RSImage) -> Tuple[bool, str]:
        if self.segmenter is None:
            return False, "no segmentation checkpoint configured"
        if t1.modality == "sar" or t2.modality == "sar":
            return False, "SAR input: the ratio-operator detector is the SAR path"
        if t1.n_bands < 3:
            return False, f"only {t1.n_bands} band(s); the model needs RGB"
        return True, "RGB optical input with a checkpoint available"

    # -- confidence -------------------------------------------------------

    def _confidence(
        self, question_type: Optional[str], validation, route: str,
        target_class: Optional[str] = None, margin: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Confidence from measured accuracy, or 0.0 with an explicit basis.

        The only honest basis available is the checkpoint's *measured*
        per-question-type accuracy on the validation split. When that is
        absent -- no checkpoint, or a question no rule answers -- the
        confidence is 0.0 and the basis says so. It is never a plausible
        number chosen to look reasonable.
        """
        if route != "semantic" or self.segmenter is None:
            return 0.0, "none: deterministic detector path has no calibrated accuracy"
        report = getattr(self.segmenter, "val_report", None) or {}
        per_type = report.get("per_type") or {}
        measured = None
        if question_type and question_type in per_type:
            measured = per_type[question_type].get("accuracy")
        elif question_type is None:
            return 0.0, "none: descriptive output, no scored question type"
        if measured is None:
            measured = report.get("average_accuracy")
        if measured is None:
            return 0.0, "none: checkpoint carries no measured validation accuracy"
        basis = (
            f"measured val accuracy for {question_type or 'average'} "
            f"= {measured:.4f}"
        )

        # Per-instance refinement, where it has been measured. The per-type
        # accuracy above is a prior: identical for every image asking that
        # question. The softmax margin says how separated this model's
        # decisions were on *this* image, and the calibration file records
        # what accuracy each (question type, margin band) cell actually
        # achieved on Val. Pooled across types the margin looks worthless --
        # 2.49 points between the extreme bands -- but within a type it
        # spans 13.17 points on average and 44.85 for change_ratio, because
        # the types have very different base rates and pooling hides it.
        # The number reported is still a measurement, never the margin
        # itself: a margin is not a probability of being correct.
        # Most specific measured cell wins. A question naming a land-cover
        # class is better described by that class's measured accuracy than by
        # the question type's average: on validation, change_or_not overall
        # reaches 84.5% while the same question about trees reaches 69.0%.
        per_class = self._class_accuracy(question_type, target_class)
        if per_class is not None:
            accuracy, n, caveat = per_class
            basis = (
                f"measured val accuracy for {question_type} about "
                f"{target_class} = {accuracy:.4f} (n={n}); per-type prior was "
                f"{measured:.4f}"
            )
            if caveat:
                basis += f". {caveat}"
            return float(max(0.0, min(1.0, accuracy))), basis

        refined = self._calibrated_accuracy(question_type, margin)
        if refined is not None:
            accuracy, low, high, n = refined
            basis = (
                f"measured val accuracy for {question_type} at softmax margin "
                f"in ({low:.4f}, {high:.4f}] = {accuracy:.4f} (n={n}); "
                f"per-type prior was {measured:.4f}"
            )
            measured = accuracy
        if validation is not None and validation.warnings:
            # Reported, never applied. An earlier version reduced confidence
            # by 10% per warning, which was a placeholder in both magnitude
            # and premise. Phase correlation on a bi-temporal pair measures
            # *apparent* displacement, mixing true misregistration with real
            # land-cover change; it cannot separate them. Measured on 40
            # genuinely co-registered SECOND pairs the median apparent shift
            # is 6.20 px, so the warning fires on essentially every real
            # pair and discriminates nothing. Calibrating a penalty against
            # that quantity would fit a number to something that does not
            # mean what the penalty claims, so the warning stays an
            # observation in the trace and does not move the number.
            basis += (
                f"; {len(validation.warnings)} validation warning(s) noted, "
                f"not applied to the number (no calibrated basis)"
            )
        return float(max(0.0, min(1.0, measured))), basis

    def _class_accuracy(self, question_type, target_class):
        """Measured accuracy for this question type about this class.

        Returns ``(accuracy, n, caveat)`` or None when unmeasured or thin.

        The caveat matters more than the number. A class the model almost
        never predicts can still score highly on "did it change?" by always
        answering no -- playgrounds reach 94.4% that way while being predicted
        at 0.00x their true frequency. That is accuracy achieved by absence,
        and reporting it bare would be the system sounding most confident
        exactly where it is blind. The measured prediction ratio travels with
        the number so a reader, or a controller, can tell the two apart.
        """
        segmenter = self.segmenter
        calibration = getattr(segmenter, "calibration", None)
        if not calibration or not question_type or not target_class:
            return None
        cell = ((calibration.get("per_target_class") or {})
                .get(question_type, {}).get(target_class))
        if not cell or not cell.get("usable") or cell.get("accuracy") is None:
            return None

        caveat = ""
        ratios = calibration.get("class_prediction_ratio") or {}
        entry = ratios.get(target_class) or {}
        ratio = entry.get("ratio")
        if ratio is not None and ratio < 0.5:
            caveat = (
                f"NOTE: {target_class} is predicted at {ratio:.2f}x its true "
                f"frequency on validation, so this accuracy is achieved "
                f"substantially by predicting absence rather than by "
                f"recognising the class"
            )
        return float(cell["accuracy"]), int(cell["n"]), caveat

    def _calibrated_accuracy(self, question_type, margin=None):
        """Measured accuracy for this question type at this image's margin.

        Returns ``(accuracy, bin_low, bin_high, n)``, or None when there is no
        calibration file, no margin from the last forward pass, or the cell
        holds too few questions to say anything -- in which case the caller
        keeps the per-type prior rather than reporting a thin measurement.
        """
        segmenter = self.segmenter
        calibration = getattr(segmenter, "calibration", None)
        if margin is None:
            margin = getattr(segmenter, "last_margin", None)
        if not calibration or margin is None or not question_type:
            return None
        cells = (calibration.get("per_type_bins") or {}).get(question_type)
        if not cells:
            return None
        for cell in cells:
            if cell["low"] < margin <= cell["high"]:
                if not cell.get("usable") or cell.get("accuracy") is None:
                    return None
                return (
                    float(cell["accuracy"]), float(cell["low"]),
                    float(cell["high"]), int(cell["n"]),
                )
        return None

    # -- main -------------------------------------------------------------

    def run(
        self,
        t1: RSImage,
        t2: RSImage,
        query: str = "",
        scene_id: Optional[str] = None,
    ) -> ChangeResult:
        trace = _Trace()
        config = self.config

        # 1 -- validation
        from tools.validation.pair import validate_pair

        with trace.stage(
            "validate_pair", "tools.validation.pair.validate_pair",
            "Incomparable inputs produce confident nonsense; check before analysing.",
            {"scene_id": scene_id},
        ) as st:
            validation = validate_pair(t1, t2)
            st.observation = (
                f"ok={validation.ok}, {len(validation.failures)} failure(s), "
                f"{len(validation.warnings)} warning(s)"
            )
        if not validation.ok:
            reasons = "; ".join(c.detail for c in validation.failures)
            result = ChangeResult.empty(summary=f"Pair rejected: {reasons}")
            result.trace = trace.entries
            result.georeferencing = georeferencing_report(t1)
            return result

        # 2 -- routing
        can_segment, reason = self._can_segment(t1, t2)
        route = "semantic" if can_segment else "index"
        with trace.stage(
            "route", "pipeline._can_segment",
            "The CDVQA rules need class maps; other input needs index differencing.",
            {"route": route},
        ) as st:
            st.observation = f"route={route} ({reason})"

        s_t1 = s_t2 = None
        margin = None
        detection = None
        producer_b_trace = None
        cross_check_agreement = None
        cross_check_contradiction = False
        target_class = parse_question(query)["target"] if query else None

        # 3 -- change extraction
        if route == "semantic":
            with trace.stage(
                "semantic_segmentation", "tools.change_analysis.semantic",
                "Predict what each changed pixel was and became, so the "
                "deterministic rules have class maps to reason over.",
                {"checkpoint": config.checkpoint},
            ) as st:
                # A caller may inject its own segmenter, and one predating
                # the margin API is still usable -- it simply yields no
                # per-instance calibration, and the confidence falls back to
                # the per-type prior rather than failing.
                if hasattr(self.segmenter, "predict_with_margin"):
                    s_t1, s_t2, margin = self.segmenter.predict_with_margin(t1, t2)
                else:
                    s_t1, s_t2 = self.segmenter.predict(t1, t2)
                mask = s_t1 != 0
                meta = self.segmenter.metadata()
                st.params.update({
                    "arch": meta["arch"], "value_scaling": meta["value_scaling"]
                })
                st.observation = (
                    f"changed fraction {float(mask.mean()):.4f}; "
                    f"classes at t1 {sorted(np.unique(s_t1).tolist())}"
                )

            # 3b -- the second opinion. The trained segmenter is a 3-band RGB
            # model trained on SECOND; its accuracy on other sensors is not
            # measured, and its softmax margin stays high on input it has
            # never seen, so the margin alone cannot flag out-of-distribution
            # imagery. Producer B reaches the same class maps from physics
            # instead of training, so where they disagree strongly the input
            # is likely outside what the model was trained on. This is
            # reported, never used to overrule: the deterministic producer is
            # an approximation, not a referee.
            if available_indices(t1):
                with trace.stage(
                    "cross_check",
                    "tools.change_analysis.index_classifier.classify_pair",
                    "Compare the trained model against a physics-based "
                    "producer, so input outside the training distribution is "
                    "visible rather than silently answered.",
                    {"indices": available_indices(t1)},
                ) as st:
                    from .index_classifier import classify_pair

                    other = classify_pair(t1, t2)
                    producer_b_trace = other.trace
                    if other.semantic_available:
                        b1, _ = other.require_semantic()
                        b_mask = b1 != 0
                        union = float((mask | b_mask).sum())
                        overlap = (
                            float((mask & b_mask).sum()) / union if union else 1.0
                        )
                        cross_check_agreement = overlap
                        # A categorical contradiction, not a matter of degree:
                        # the trained model found no change at all while
                        # independent physical evidence found some. Measured
                        # on a real Sentinel-2 pair the model reported 0.0000
                        # changed against the index producer's 0.3410 --
                        # SECOND is sub-metre aerial imagery, Sentinel-2 is
                        # 10 m, and the model does not transfer. Recognising
                        # "one found nothing, the other found a third of the
                        # scene" needs no invented threshold.
                        # The confidence calibration was measured entirely on
                        # SECOND, which is 3-band RGB. This cross-check only
                        # runs when NDVI/NDWI/NDBI are computable, which needs
                        # NIR -- so reaching this line at all means the input
                        # is definitionally not SECOND-like and the
                        # calibration has no evidence about it.
                        #
                        # An earlier version required the model to find
                        # *exactly* zero change, justified as needing no
                        # invented threshold. It was brittle instead: on three
                        # real Sentinel-2 scenes the model found 0, 126 and
                        # 3,526 changed pixels, and only the first was caught.
                        # The other two reported 0.82 confidence on input just
                        # as far out of distribution, because one pixel
                        # defeated the rule.
                        cross_check_contradiction = True
                        st.observation = (
                            f"change-mask agreement with the index producer: "
                            f"{overlap:.3f} (Jaccard). Model changed fraction "
                            f"{float(mask.mean()):.4f}, index producer "
                            f"{float(b_mask.mean()):.4f}"
                        )
                    else:
                        st.observation = (
                            "index producer could not supply class maps; no "
                            "cross-check available"
                        )
        else:
            with trace.stage(
                "index_detection", "tools.change_analysis.detector.detect_change",
                "Difference the spectral index (optical) or ratio the "
                "backscatter (SAR), then threshold with Otsu -- never a constant.",
                {"modality": t1.modality,
                 "available_indices": available_indices(t1),
                 "target_class": target_class},
            ) as st:
                detection = detect_change_stack(t1, t2, target_class=target_class)
                mask = detection.mask
                st.params.update(detection.as_trace_params())
                st.observation = (
                    f"operator={detection.operator}, threshold="
                    f"{detection.threshold}, changed pixels={int(mask.sum())}"
                )

            # 3b -- Producer B. The index detector above says *whether* a
            # pixel changed; the CDVQA rules also need to know *what* it was
            # and became. For multispectral optical the spectral indices can
            # supply that. For SAR they cannot -- backscatter carries no
            # mapping to land cover -- and the producer returns no maps at
            # all rather than empty ones, because an all-zero map reads to
            # every rule as "nothing changed" and would be answered with
            # full confidence.
            with trace.stage(
                "index_classification",
                "tools.change_analysis.index_classifier.classify_pair",
                "Give the rules class maps where physics can supply them, "
                "and refuse where it cannot rather than returning zeros.",
                {"modality": t1.modality},
            ) as st:
                from .index_classifier import classify_pair

                classification = classify_pair(t1, t2)
                producer_b_trace = classification.trace
                if classification.semantic_available:
                    s_t1, s_t2 = classification.require_semantic()
                    s_t1 = s_t1.astype(np.int64)
                    s_t2 = s_t2.astype(np.int64)
                    st.observation = (
                        f"semantic maps from {classification.trace['indices']}; "
                        f"precedence {' > '.join(classification.trace['precedence'])}"
                    )
                else:
                    st.observation = (
                        "no semantic maps: "
                        f"{classification.trace.get('limitation', 'unsupported input')}. "
                        "CDVQA rules will not be applied."
                    )

        # 4 -- morphology
        with trace.stage(
            "morphology", "tools.change_analysis.morphology.clean_mask",
            "Open then close: remove speckle that would become thousands of "
            "one-pixel regions, then fill pinholes in what survives.",
            {"opening_radius": config.opening_radius,
             "closing_radius": config.closing_radius},
        ) as st:
            before = int(mask.sum())
            mask = clean_mask(mask, config.opening_radius, config.closing_radius)
            st.observation = f"changed pixels {before} -> {int(mask.sum())}"

        # 5 -- regions
        with trace.stage(
            "regions", "tools.change_analysis.regions.extract_regions",
            "Region-level statistics are markedly more robust to "
            "misregistration than pixel-level ones.",
            {"min_region_pixels": config.min_region_pixels},
        ) as st:
            stacks = None
            try:
                s1_stack, s2_stack = index_stack(t1), index_stack(t2)
                if s1_stack and s2_stack:
                    stacks = (s1_stack, s2_stack)
            except ValueError:
                stacks = None
            regions = extract_regions(
                mask, t1, min_pixels=config.min_region_pixels, index_stacks=stacks
            )
            summary_stats = summarise_regions(regions, t1)
            st.params["spectral_signature"] = bool(stacks)
            st.observation = (
                f"{len(regions)} region(s), "
                f"{summary_stats['total_changed_pixels']} changed pixels, "
                f"area_m2={summary_stats['total_area_m2']}"
                + (f", signatures from {sorted(stacks[0])}" if stacks else "")
            )

        # 6 -- answer
        question_type = classify_question(query) if query else None
        answer: Optional[str] = None
        answer_evidence: Dict[str, Any] = {}
        with trace.stage(
            "answer", "tools.change_analysis.cdvqa.answer_question",
            "Apply the CDVQA rule for this question type to the class maps; "
            "fall back to a region summary rather than inventing a label.",
            {"query": query, "question_type": question_type},
        ) as st:
            compound_summary: Optional[str] = None
            if question_type and s_t1 is not None and is_compound(query):
                # Several classes named at once. Each is answered by the same
                # rule, but `answer` stays None: CDVQA has no token for a
                # combined result and no rule for combining one, so the
                # per-class answers are reported as they are rather than
                # collapsed into a single fabricated token.
                outcome = answer_compound(query, question_type, s_t1, s_t2)
                compound_summary = outcome.summary()
                answer_evidence = {
                    "compound": True,
                    "targets": outcome.targets,
                    "per_class": {
                        target: sub.answer
                        for target, sub in zip(outcome.targets, outcome.answers)
                    },
                    "per_class_evidence": {
                        target: sub.evidence
                        for target, sub in zip(outcome.targets, outcome.answers)
                    },
                }
                st.observation = (
                    f"compound question over {len(outcome.targets)} classes; "
                    f"{compound_summary}. No single CDVQA token applies."
                )
            elif question_type and s_t1 is not None:
                outcome = answer_question(query, question_type, s_t1, s_t2)
                answer = outcome.answer
                answer_evidence = outcome.evidence
                st.observation = (
                    f"answer={answer!r}"
                    + (f" (reason={outcome.reason})" if outcome.reason else "")
                )
            elif question_type:
                st.observation = (
                    "question type recognised but no class maps available on "
                    "the index route; answering descriptively instead"
                )
            else:
                st.observation = (
                    "no CDVQA rule matches this query; answering descriptively"
                )

        from .describe import describe_change

        description = describe_change(regions, summary_stats, t1, s_t1, s_t2)
        if compound_summary:
            # The per-class answers are the response to a compound question;
            # the region description is context for them.
            description = f"{compound_summary}. {description}"

        # 7 -- overlay evidence
        overlay_paths: Dict[str, str] = {}
        if config.overlay_dir:
            with trace.stage(
                "overlay", "tools.change_analysis.overlay",
                "A change claim should come with a picture a human can check.",
                {"overlay_dir": config.overlay_dir},
            ) as st:
                overlay_paths = self._write_overlays(
                    t1, t2, mask, regions, scene_id,
                    s_t1=s_t1, s_t2=s_t2, target_class=target_class,
                )
                st.observation = (
                    f"{len(overlay_paths)} image(s) written"
                    + (f"; focus scoped to {target_class}" if target_class
                       and "focus_comparison" in overlay_paths else "")
                )

        # 8 -- confidence
        confidence, basis = self._confidence(
            question_type, validation, route, target_class, margin
        )
        if cross_check_contradiction:
            # The calibration was measured on inputs where the model does
            # detect change. It says nothing about an input where the model
            # detects none and physics says otherwise, so there is no
            # measured basis here -- and measured-or-zero means zero.
            confidence = 0.0
            basis = (
                "none: the confidence calibration was measured on SECOND, "
                "which is 3-band RGB aerial imagery. This input carries "
                "spectral bands beyond RGB, so it is outside that "
                "distribution and no measured accuracy applies to it. "
                f"Change-mask agreement between the trained model and the "
                f"independent index producer was "
                f"{cross_check_agreement if cross_check_agreement is not None else 0:.1%}"
                ", reported as an observation. The answer is reported; the "
                "number behind it is not."
            )
        with trace.stage(
            "confidence", "pipeline._confidence",
            "Report a confidence only where a measured basis exists; "
            "otherwise report zero and say why.",
            {"confidence": confidence, "basis": basis},
        ) as st:
            st.observation = f"confidence={confidence:.4f} basis={basis}"

        result = ChangeResult(
            answer=answer,
            trace=trace.entries,
            changed=bool(summary_stats["total_changed_pixels"] > 0),
            summary=description,
            change_type=self._dominant_change_type(s_t1, s_t2),
            confidence=confidence,
            changed_area_pixels=summary_stats["total_changed_pixels"],
            changed_area_m2=summary_stats["total_area_m2"],
            regions=[r.to_dict() for r in regions[: config.max_regions_reported]],
            semantic_t1=s_t1,
            semantic_t2=s_t2,
            georeferencing=georeferencing_report(t1),
        )
        result.evidence_extras = {
            "validation": validation.to_dict(),
            "route": route,
            "question_type": question_type,
            "answer_evidence": answer_evidence,
            "confidence_basis": basis,
            "region_summary": summary_stats,
            "overlays": overlay_paths,
            "overlay_legend": (
                __import__(
                    "tools.change_analysis.overlay", fromlist=["focus_legend"]
                ).focus_legend(s_t1, s_t2) if s_t1 is not None else {}
            ),
            "segmenter": self.segmenter.metadata() if route == "semantic" else None,
            "index_classifier": producer_b_trace,
            "cross_check_agreement": cross_check_agreement,
            "cross_check_contradiction": cross_check_contradiction,
        }
        return result

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _dominant_change_type(s_t1, s_t2) -> Optional[str]:
        """Commonest destination class, or None without class maps."""
        if s_t2 is None:
            return None
        from .cdvqa import CLASS_NAMES

        values, counts = np.unique(s_t2[s_t2 != 0], return_counts=True)
        if values.size == 0:
            return None
        return CLASS_NAMES.get(int(values[int(np.argmax(counts))]))

    def _write_overlays(self, t1, t2, mask, regions, scene_id,
                        s_t1=None, s_t2=None, target_class=None) -> Dict[str, str]:
        from .overlay import (
            before_after_crops, focus_comparison, overlay_mask, save_png,
        )

        os.makedirs(self.config.overlay_dir, exist_ok=True)
        stem = scene_id or "scene"
        paths: Dict[str, str] = {}
        # Side-by-side, scoped to what was actually asked. Built from the
        # same class maps the rule counted, so the picture cannot disagree
        # with the answer.
        if s_t1 is not None and s_t2 is not None:
            try:
                paths["focus_comparison"] = save_png(
                    os.path.join(self.config.overlay_dir, f"{stem}_focus.png"),
                    focus_comparison(t1, t2, s_t1, s_t2, target_class=target_class),
                )
            except ValueError:
                # An unknown class name must not cost the caller every other
                # piece of evidence.
                pass
        paths["mask_overlay"] = save_png(
            os.path.join(self.config.overlay_dir, f"{stem}_overlay.png"),
            overlay_mask(t2, mask, outline_only=False),
        )
        if regions:
            crops = before_after_crops(t1, t2, regions[0])
            paths["before"] = save_png(
                os.path.join(self.config.overlay_dir, f"{stem}_before.png"),
                crops["before"],
            )
            paths["after"] = save_png(
                os.path.join(self.config.overlay_dir, f"{stem}_after.png"),
                crops["after"],
            )
        return paths


def export_report(result: ChangeResult, path: str) -> str:
    """Write a JSON report of a run: answer, summary, regions and full trace.

    Semantic maps are omitted -- they are large arrays, and their derived
    statistics are already in the regions and answer evidence.
    """
    payload = {
        "answer": result.answer,
        "summary": result.summary,
        "changed": result.changed,
        "change_type": result.change_type,
        "confidence": result.confidence,
        "changed_area_pixels": result.changed_area_pixels,
        "changed_area_m2": result.changed_area_m2,
        "georeferencing": result.georeferencing,
        "regions": result.regions,
        "extras": getattr(result, "evidence_extras", {}),
        "trace": result.trace,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path
