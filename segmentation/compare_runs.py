"""Side-by-side comparison of two (or more) training runs.

Reads each run's ``history.json`` and ``best_val_report.json`` from its
output directory and prints:

*   per-question-type Val accuracy at each run's best checkpoint, side by side,
*   per-class IoU side by side, so the rare classes (trees, water,
    playgrounds) that drive smallest_change / change_to_what are directly
    visible,
*   the head-disagreement trend per epoch for each run, not just the final
    value -- a plateau above the bar across two independent runs is a much
    stronger signal than one run's number.

No selection or decision logic here: it only lays the numbers next to each
other so the checkpoint and architecture calls can be made from evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

CLASS_NAMES = [
    "unchanged", "NVG", "low_veg", "trees", "buildings", "water", "playgrounds"
]


def _load_run(out_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(os.path.join(out_dir, "best_val_report.json"), encoding="utf-8") as f:
        best = json.load(f)
    history_path = os.path.join(out_dir, "history.json")
    history = None
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
    return best, history


def _fmt_pct(value, width: int = 7) -> str:
    if value is None:
        return "n/a".rjust(width)
    return f"{value * 100:.2f}%".rjust(width)


def compare(labels: List[str], dirs: List[str]) -> str:
    runs = [_load_run(d) for d in dirs]
    lines: List[str] = []

    # -- headline --------------------------------------------------------
    lines.append("BEST CHECKPOINT (selected on Val average accuracy)")
    lines.append("-" * 72)
    header = "  " + "metric".ljust(24) + "".join(l.rjust(16) for l in labels)
    lines.append(header)
    for key, name in (
        ("average_accuracy", "Average Accuracy"),
        ("overall_accuracy", "Overall Accuracy"),
        ("miou", "mIoU (diagnostic)"),
        ("pixel_accuracy", "pixel accuracy"),
        ("head_disagreement", "head disagreement"),
    ):
        row = "  " + name.ljust(24)
        for best, _ in runs:
            row += _fmt_pct(best.get(key), 16)
        lines.append(row)
    row = "  " + "best epoch".ljust(24)
    for _, history in runs:
        row += (str(history["best"]["epoch"]) if history else "?").rjust(16)
    lines.append(row)
    row = "  " + "unanswered / total".ljust(24)
    for best, _ in runs:
        row += f"{best['unanswered_total']}/{best['n_questions']}".rjust(16)
    lines.append(row)

    # -- per question type ----------------------------------------------
    lines.append("")
    lines.append("PER QUESTION TYPE -- Val accuracy at best checkpoint")
    lines.append("-" * 72)
    lines.append("  " + "type".ljust(24) + "".join(l.rjust(16) for l in labels)
                 + "     delta")
    q_types = list(runs[0][0]["per_type"].keys())
    for q_type in q_types:
        row = "  " + q_type.ljust(24)
        values = []
        for best, _ in runs:
            entry = best["per_type"].get(q_type, {})
            acc = entry.get("accuracy")
            values.append(acc)
            unanswered = entry.get("unanswered", 0)
            cell = _fmt_pct(acc)
            if unanswered:
                cell += f"*{unanswered}"
            row += cell.rjust(16)
        if len(values) == 2 and None not in values:
            delta = (values[1] - values[0]) * 100
            row += f"  {delta:+6.2f}"
        lines.append(row)
    lines.append("  (* = questions the rules could not answer, model predicted"
                 " the class absent)")

    # -- per class IoU -------------------------------------------------
    lines.append("")
    lines.append("PER-CLASS IoU (diagnostic) -- rare classes drive the weak types")
    lines.append("-" * 72)
    lines.append("  " + "class".ljust(24) + "".join(l.rjust(16) for l in labels)
                 + "     delta")
    for index, name in enumerate(CLASS_NAMES):
        row = "  " + name.ljust(24)
        values = []
        for best, _ in runs:
            iou = best["per_class_iou"][index] if index < len(best["per_class_iou"]) else None
            values.append(iou)
            row += _fmt_pct(iou)
        if len(values) == 2 and None not in values:
            row += f"  {(values[1] - values[0]) * 100:+6.2f}"
        lines.append(row)

    # -- disagreement trend ------------------------------------------
    lines.append("")
    lines.append("HEAD-DISAGREEMENT TREND (% pixels, changed-vs-unchanged)")
    lines.append("-" * 72)
    have_history = all(h is not None for _, h in runs)
    if not have_history:
        lines.append("  (history.json missing for at least one run -- "
                     "trend unavailable, run still in progress?)")
    else:
        max_epochs = max(len(h["history"]) for _, h in runs)
        lines.append("  epoch " + "".join(l.rjust(16) for l in labels)
                     + "".join((l + " AA").rjust(16) for l in labels))
        for epoch in range(0, max_epochs, max(max_epochs // 20, 1)):
            row = f"  {epoch:>5} "
            for _, history in runs:
                h = history["history"]
                cell = f"{h[epoch]['head_disagreement'] * 100:.3f}%" if epoch < len(h) else ""
                row += cell.rjust(16)
            for _, history in runs:
                h = history["history"]
                cell = f"{h[epoch]['val_average_accuracy'] * 100:.2f}%" if epoch < len(h) else ""
                row += cell.rjust(16)
            lines.append(row)
        lines.append("")
        for label, (_, history) in zip(labels, runs):
            series = [e["head_disagreement"] * 100 for e in history["history"]]
            tail = series[len(series) // 2:]
            lines.append(
                f"  {label}: min {min(series):.2f}%  max {max(series):.2f}%  "
                f"second-half mean {sum(tail) / len(tail):.2f}%  "
                f"(bar = 2.00%)"
            )

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", nargs=2, metavar=("LABEL", "DIR"),
        required=True, help="a run label and its output directory; repeatable",
    )
    parser.add_argument("--out", default=None, help="also write the report here")
    args = parser.parse_args(argv)

    labels = [label for label, _ in args.run]
    dirs = [d for _, d in args.run]
    report = compare(labels, dirs)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
