"""Measure what each softmax-margin band actually achieves on Val.

The confidence reported before this was the checkpoint's per-question-type
Val accuracy. That is honest but it is a *prior*, not an estimate: a crisp
image and a hopeless one asking the same question type receive the identical
number, because the number describes the question type rather than the image.
The problem statement asks the system to "estimate confidence", and a lookup
that ignores the input is not an estimate.

The fix is not to invent a formula. A softmax margin says how separated the
model's decisions were on this image, but it carries no calibrated mapping to
probability-of-correctness, and multiplying it into the prior would fabricate
one. So the mapping is measured instead: bin Val questions by the margin of
the scene they came from, and record the accuracy actually achieved in each
bin. A new instance is then reported the accuracy that its margin band really
achieved -- a measurement, like everything else in this branch.

Bins are quantiles of the observed margin distribution, so every bin holds a
comparable number of questions and no cut is chosen by hand. A bin is only
kept when it holds enough questions to mean anything; the floor is derived
from the binomial standard error rather than picked, and thin bins fall back
to the per-type prior, which is what the pipeline used before.

Run after selecting a checkpoint, on Val only. This never touches Test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from eval.run_cdvqa import QUESTION_TYPES, load_cdvqa_split
from tools.change_analysis.cdvqa import (
    CLASS_NAMES, answer_question, decode_label, parse_question, scene_stats,
)

# A bin is usable when its accuracy is measured tightly enough to beat the
# spread between question types it would replace. The per-type accuracies
# span roughly 0.35 to 0.85, so a standard error of 0.05 is the point at
# which a bin says more than the prior it displaces. n = p(1-p)/se^2, worst
# case p = 0.5, gives 100. Derived from the measured spread, not chosen.
MIN_BIN_QUESTIONS = 100
TARGET_STANDARD_ERROR = 0.05


def _standard_error(correct: int, total: int) -> float:
    if total == 0:
        return float("nan")
    p = correct / total
    return math.sqrt(max(p * (1.0 - p), 1e-9) / total)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cdvqa-root", default="datasets/CDVQA")
    parser.add_argument("--second-root", default="datasets/SECOND_raw/train")
    parser.add_argument("--split", default="Val", choices=("Train", "Val"),
                        help="Test and Test2 are budgeted and never calibrated on")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from tools.change_analysis.semantic import SemanticSegmenter
    from tools.change_analysis.io import RSImage

    segmenter = SemanticSegmenter(args.checkpoint, device=args.device)
    samples = load_cdvqa_split(args.cdvqa_root, args.split)
    by_scene: Dict[str, List] = defaultdict(list)
    for sample in samples:
        by_scene[sample.pair_id].append(sample)

    scenes = sorted(by_scene)
    print(f"scoring {len(samples)} questions over {len(scenes)} scenes", flush=True)

    records = []  # (margin, question_type, correct, target class)
    predicted_pixels = {c: 0 for c in CLASS_NAMES}
    truth_pixels = {c: 0 for c in CLASS_NAMES}
    for position, scene in enumerate(scenes):
        t1 = _load(os.path.join(args.second_root, "im1", scene))
        t2 = _load(os.path.join(args.second_root, "im2", scene))
        s_t1, s_t2 = segmenter.predict(t1, t2)
        margin = segmenter.last_margin
        # Predicted vs true class frequency. A class the model almost never
        # predicts can still score well on "did it change?" by always saying
        # no. That is accuracy achieved by absence, and the confidence basis
        # has to say so rather than present it as competence.
        truth = decode_label(os.path.join(args.second_root, "label1", scene))
        for class_index in CLASS_NAMES:
            predicted_pixels[class_index] += int((s_t1 == class_index).sum())
            truth_pixels[class_index] += int((truth == class_index).sum())
        stats = scene_stats(s_t1, s_t2)
        for sample in by_scene[scene]:
            outcome = answer_question(
                sample.question, sample.question_type, s_t1, s_t2, stats=stats
            )
            records.append((
                margin, sample.question_type,
                int(outcome.answer == sample.answer),
                parse_question(sample.question)["target"],
            ))
        if (position + 1) % 50 == 0:
            print(f"  {position + 1}/{len(scenes)} scenes", flush=True)

    margins = np.array([r[0] for r in records], dtype=np.float64)
    correct = np.array([r[2] for r in records], dtype=np.int64)
    types = [r[1] for r in records]
    targets = [r[3] for r in records]

    # Quantile edges: equal-population bins, so no cut is chosen by hand.
    edges = np.quantile(margins, np.linspace(0, 1, args.bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_questions": len(records),
        "n_scenes": len(scenes),
        "margin_min": float(margins.min()),
        "margin_max": float(margins.max()),
        "margin_median": float(np.median(margins)),
        "bin_edges": [float(e) for e in edges],
        "min_bin_questions": MIN_BIN_QUESTIONS,
        "bins": [],
        "per_type_bins": {},
    }

    print(f"\nmargin range {margins.min():.4f} to {margins.max():.4f}\n")
    print(f"{'bin':<26}{'n':>7}{'accuracy':>11}{'std err':>10}{'usable':>8}")
    for index in range(args.bins):
        inside = (margins > edges[index]) & (margins <= edges[index + 1])
        n = int(inside.sum())
        hits = int(correct[inside].sum())
        accuracy = hits / n if n else float("nan")
        se = _standard_error(hits, n)
        usable = n >= MIN_BIN_QUESTIONS
        report["bins"].append({
            "low": float(edges[index]), "high": float(edges[index + 1]),
            "n": n, "accuracy": accuracy, "standard_error": se, "usable": usable,
        })
        label = f"({edges[index]:.4f}, {edges[index + 1]:.4f}]"
        print(f"{label:<26}{n:>7}{accuracy * 100:>10.2f}%{se:>10.4f}{str(usable):>8}")

    # Per question type as well: the margin says something different about a
    # binary yes/no than about a six-way class choice.
    for question_type in QUESTION_TYPES:
        mask = np.array([t == question_type for t in types])
        if not mask.any():
            continue
        entries = []
        for index in range(args.bins):
            inside = mask & (margins > edges[index]) & (margins <= edges[index + 1])
            n = int(inside.sum())
            hits = int(correct[inside].sum())
            entries.append({
                "low": float(edges[index]), "high": float(edges[index + 1]),
                "n": n,
                "accuracy": (hits / n) if n else None,
                "standard_error": _standard_error(hits, n) if n else None,
                "usable": n >= MIN_BIN_QUESTIONS,
            })
        report["per_type_bins"][question_type] = entries

    # Per target land-cover class. The model under-predicts rare classes, so
    # a water question inheriting change_or_not's overall accuracy would be
    # confidently wrong about precisely the branch's weakest area. Measuring
    # per class lets the reported number say so.
    cells = defaultdict(lambda: [0, 0])
    for (_, question_type, hit, target) in records:
        if target is None:
            continue
        entry = cells[(question_type, target)]
        entry[0] += hit
        entry[1] += 1
    report["per_target_class"] = {}
    print()
    print(f"{'question type':<22}{'class':<18}{'n':>7}{'accuracy':>11}{'usable':>8}")
    for (question_type, target), (hits, n) in sorted(cells.items()):
        usable = n >= MIN_BIN_QUESTIONS
        report["per_target_class"].setdefault(question_type, {})[target] = {
            "n": n, "accuracy": hits / n if n else None,
            "standard_error": _standard_error(hits, n), "usable": usable,
        }
        if usable:
            print(f"{question_type:<22}{target:<18}{n:>7}{hits / n * 100:>10.2f}%"
                  f"{str(usable):>8}")

    total_predicted = sum(predicted_pixels.values()) or 1
    total_truth = sum(truth_pixels.values()) or 1
    report["class_prediction_ratio"] = {}
    print()
    print(f"{'class':<18}{'predicted':>11}{'truth':>9}{'ratio':>8}")
    for class_index, name in CLASS_NAMES.items():
        p_share = predicted_pixels[class_index] / total_predicted
        t_share = truth_pixels[class_index] / total_truth
        ratio = (p_share / t_share) if t_share else None
        report["class_prediction_ratio"][name] = {
            "predicted_share": p_share, "true_share": t_share, "ratio": ratio,
        }
        print(f"{name:<18}{p_share * 100:10.2f}%{t_share * 100:8.2f}%"
              f"{(ratio if ratio is not None else 0):7.2f}x")

    spread = [b["accuracy"] for b in report["bins"] if b["usable"]]
    if len(spread) >= 2:
        report["accuracy_spread_across_bins"] = float(max(spread) - min(spread))
        print(f"\naccuracy spread across usable bins: "
              f"{report['accuracy_spread_across_bins'] * 100:.2f} points")
        print("A spread near zero would mean the margin carries no information "
              "and the per-type prior should be kept.")

    out = args.out or os.path.join(
        os.path.dirname(args.checkpoint), "confidence_calibration.json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {out}")
    return 0


def _load(path: str):
    """SECOND scenes are plain PNGs; load as an ungeoreferenced RSImage."""
    from PIL import Image

    from tools.change_analysis.io import RSImage

    array = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    return RSImage(
        array=np.transpose(array, (2, 0, 1)), crs=None, transform=None,
        modality="optical", band_names=["red", "green", "blue"], gsd_m=None,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
