"""Downstream validation: score the model on CDVQA Val, not on mIoU.

mIoU is a proxy and not obviously the right one. Every CDVQA rule is an
*area comparison*, so a model with mediocre boundaries but well-calibrated
per-class areas can answer well, while one with crisp boundaries that
systematically under-predicts a rare class fails every question about it.
Checkpoint selection therefore uses CDVQA Val average accuracy; mIoU is
reported alongside as a diagnostic only.

Also reports head consistency. In ground truth a pixel is unchanged in both
maps or classed in both -- `white(label1) == white(label2)` holds at exactly
100% across all 4,662 SECOND pairs. Nothing in the architecture enforces
that, so head 1 can call a pixel unchanged while head 2 calls it buildings.
The rules have no defined behaviour for that state, so its frequency is
measured before deciding whether it needs structural repair.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from eval.run_cdvqa import CDVQA_GROUPS, QUESTION_TYPES, group_of, load_cdvqa_split
from tools.change_analysis.cdvqa import N_CLASSES, answer_question, scene_stats

__all__ = [
    "CDVQAValidator",
    "head_disagreement",
]


def head_disagreement(p1: torch.Tensor, p2: torch.Tensor) -> float:
    """Fraction of pixels where the heads disagree on changed-vs-unchanged.

    Ground-truth agreement is exactly 100%, so any value above zero is a
    model artefact rather than a property of the data.
    """
    return float(((p1 == 0) != (p2 == 0)).float().mean().item())


class CDVQAValidator:
    """Runs the full pipeline on CDVQA Val and scores it against gold.

    Question/answer data is loaded once and reused every epoch; only the
    forward pass repeats.
    """

    def __init__(
        self,
        cdvqa_root: str,
        second_root: str,
        split: str = "Val",
        scenes: Optional[Sequence[str]] = None,
    ) -> None:
        samples = load_cdvqa_split(cdvqa_root, split)
        self.by_scene: Dict[str, List] = defaultdict(list)
        for sample in samples:
            self.by_scene[sample.pair_id].append(sample)
        self.scenes = list(scenes) if scenes is not None else sorted(self.by_scene)
        self.second_root = second_root
        self.split = split

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        loader,
        device: torch.device,
        amp: bool = True,
    ) -> Dict[str, Any]:
        """Score ``model``. ``loader`` must yield batches in ``self.scenes`` order."""
        model.eval()

        hits: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
        totals: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
        unanswered: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
        group_hits: Dict[str, int] = {g: 0 for g in CDVQA_GROUPS}
        group_totals: Dict[str, int] = {g: 0 for g in CDVQA_GROUPS}

        matrix = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)
        disagreements: List[float] = []

        index = 0
        for x, y1, y2 in loader:
            x = x.to(device, non_blocking=True)
            y1d, y2d = y1.to(device), y2.to(device)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                o1, o2 = model(x)
            p1, p2 = o1.argmax(1), o2.argmax(1)

            for k in (0, 1):
                pred, target = (p1, y1d) if k == 0 else (p2, y2d)
                matrix += torch.bincount(
                    (target * N_CLASSES + pred).view(-1),
                    minlength=N_CLASSES * N_CLASSES,
                ).view(N_CLASSES, N_CLASSES)

            disagreements.append(head_disagreement(p1, p2))

            cpu1 = p1.to(torch.uint8).cpu().numpy()
            cpu2 = p2.to(torch.uint8).cpu().numpy()
            for row in range(cpu1.shape[0]):
                if index >= len(self.scenes):
                    break
                scene = self.scenes[index]
                index += 1
                s1, s2 = cpu1[row], cpu2[row]
                stats = scene_stats(s1, s2)
                for sample in self.by_scene.get(scene, ()):
                    question_type = sample.question_type
                    totals[question_type] = totals.get(question_type, 0) + 1
                    group = group_of(question_type)
                    group_totals[group] = group_totals.get(group, 0) + 1
                    predicted = answer_question(
                        sample.question, question_type, s1, s2, stats
                    ).answer
                    if predicted is None:
                        # No answer produced. Counted as a miss -- silence is
                        # not a correct answer -- but tracked separately so a
                        # rule-coverage problem stays distinguishable from a
                        # segmentation problem.
                        unanswered[question_type] = unanswered.get(question_type, 0) + 1
                        continue
                    if predicted == sample.answer:
                        hits[question_type] = hits.get(question_type, 0) + 1
                        group_hits[group] = group_hits.get(group, 0) + 1

        per_type = {
            t: {
                "n": totals[t],
                "correct": hits.get(t, 0),
                "unanswered": unanswered.get(t, 0),
                "accuracy": (hits.get(t, 0) / totals[t]) if totals[t] else None,
            }
            for t in totals
        }
        measured = [v["accuracy"] for v in per_type.values() if v["accuracy"] is not None]
        total_n = sum(totals.values())

        per_group = {
            g: {
                "n": group_totals[g],
                "accuracy": (group_hits[g] / group_totals[g]) if group_totals[g] else None,
            }
            for g in group_totals
        }
        group_measured = [v["accuracy"] for v in per_group.values() if v["accuracy"] is not None]

        cpu_matrix = matrix.cpu().double()
        intersection = cpu_matrix.diag()
        union = cpu_matrix.sum(0) + cpu_matrix.sum(1) - intersection
        present = union > 0
        per_class_iou = (intersection / union.clamp(min=1)).tolist()
        miou = (
            float((intersection[present] / union[present]).mean().item())
            if present.any() else float("nan")
        )

        return {
            "split": self.split,
            "scenes_scored": index,
            "per_type": per_type,
            "average_accuracy": (sum(measured) / len(measured)) if measured else None,
            "overall_accuracy": (
                sum(hits.values()) / total_n if total_n else None
            ),
            "per_group": per_group,
            "group_average_accuracy": (
                sum(group_measured) / len(group_measured) if group_measured else None
            ),
            "n_questions": total_n,
            "unanswered_total": sum(unanswered.values()),
            "miou": miou,
            "per_class_iou": per_class_iou,
            "pixel_accuracy": float(
                (cpu_matrix.diag().sum() / cpu_matrix.sum().clamp(min=1)).item()
            ),
            "head_disagreement": float(np.mean(disagreements)) if disagreements else 0.0,
            "head_disagreement_max": float(np.max(disagreements)) if disagreements else 0.0,
        }


def format_validation(report: Dict[str, Any]) -> str:
    lines = [
        f"CDVQA {report['split']} -- downstream accuracy "
        f"({report['scenes_scored']} scenes, {report['n_questions']} questions)",
        "-" * 72,
    ]
    for question_type, entry in report["per_type"].items():
        accuracy = entry["accuracy"]
        rendered = "NOT_MEASURED" if accuracy is None else f"{accuracy * 100:6.2f}%"
        lines.append(
            f"  {question_type:<22} {rendered:>13}  "
            f"(n={entry['n']:>6}, unanswered={entry['unanswered']})"
        )
    for label, key in (
        ("Average Accuracy", "average_accuracy"),
        ("Overall Accuracy", "overall_accuracy"),
    ):
        value = report[key]
        rendered = "NOT_MEASURED" if value is None else f"{value * 100:6.2f}%"
        lines.append(f"  {label:<22} {rendered:>13}")
    lines.append("")
    lines.append(f"  mIoU (diagnostic)      {report['miou'] * 100:6.2f}%")
    lines.append(f"  pixel accuracy         {report['pixel_accuracy'] * 100:6.2f}%")
    lines.append(
        f"  head disagreement      {report['head_disagreement'] * 100:6.3f}%"
        f"  (max batch {report['head_disagreement_max'] * 100:.3f}%)"
    )
    return "\n".join(lines)
