"""CDVQA evaluation harness (build-order step 3).

CDVQA is a 19-way classification task, not text generation, so scoring is
exact string match against a closed vocabulary. This module exists before any
model does, because a majority-class baseline run today gives every later
change a measured delta to be judged against.

Reported metrics match the paper's table format:

*   per-question-type accuracy, for all five types
*   ``average_accuracy`` -- the unweighted mean over types that have samples
*   ``overall_accuracy`` -- micro-averaged over samples

The two differ whenever the type distribution is unbalanced, which in CDVQA
it is; reporting only one of them is how benchmark comparisons go wrong.

Nothing here invents a number. If the dataset is absent the harness raises;
if a split contains an answer outside the 19, it raises rather than scoring
against a vocabulary we know to be wrong.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "CDVQA_ANSWERS",
    "QUESTION_TYPES",
    "CDVQASample",
    "assert_no_pair_leakage",
    "evaluate",
    "load_split",
    "majority_class_baseline",
    "per_type_majority_baseline",
]

# The 19 answers, exactly as CLAUDE.md section 2 lists them. Order is the
# paper's frequency order and is preserved for comparability.
CDVQA_ANSWERS = (
    "no",
    "yes",
    "0%-10%",
    "0",
    "NVG surface",
    "buildings",
    "low vegetation",
    "10%-20%",
    "trees",
    "20%-30%",
    "water",
    "80%-90%",
    "30%-40%",
    "90%-100%",
    "70%-80%",
    "40%-50%",
    "60%-70%",
    "50%-60%",
    "playgrounds",
)

QUESTION_TYPES = (
    "change_or_not",
    "increase_or_decrease",
    "change_to_what",
    "largest_smallest_change",
    "change_ratio",
)

# Accepted JSON key spellings, most explicit first. The public CDVQA release
# and our own exports do not agree on these, so the loader adapts rather than
# forcing a preprocessing step that would be easy to forget.
_PAIR_ID_KEYS = ("pair_id", "image_id", "img_id", "image", "id")
_QUESTION_KEYS = ("question", "question_text")
_ANSWER_KEYS = ("answer", "answer_text", "gt_answer")
_TYPE_KEYS = ("question_type", "type", "q_type")

_ANSWER_LOOKUP = {answer.strip().lower(): answer for answer in CDVQA_ANSWERS}


@dataclass(frozen=True)
class CDVQASample:
    """One CDVQA question, tied to the image pair it was asked about."""

    pair_id: str
    question: str
    answer: str
    question_type: str


def normalise_answer(answer: str) -> str:
    """Canonical form of an answer string, for comparison only.

    Case and surrounding whitespace vary across exports; the underlying class
    does not. Returns the lowercased form, which is what scoring compares.
    """
    return str(answer).strip().lower()


def _pick(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in record:
            return record[key]
    return None


def load_split(path: str) -> List[CDVQASample]:
    """Load one CDVQA split.

    Raises
    ------
    FileNotFoundError:
        the split is not on disk. Callers must report NOT_YET_MEASURED
        rather than substituting a zero.
    ValueError:
        the records are keyed unexpectedly, or contain an answer outside the
        19-way vocabulary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CDVQA split not found: {path!r}. "
            "Accuracy is NOT_YET_MEASURED until the official split is present."
        )

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        # Some exports wrap the list under a top-level key.
        for key in ("questions", "data", "annotations"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        raise ValueError(
            f"expected a list of records in {path!r}, got {type(payload).__name__}"
        )

    samples: List[CDVQASample] = []
    for index, record in enumerate(payload):
        pair_id = _pick(record, _PAIR_ID_KEYS)
        question = _pick(record, _QUESTION_KEYS)
        answer = _pick(record, _ANSWER_KEYS)
        question_type = _pick(record, _TYPE_KEYS)

        if pair_id is None or question is None or answer is None:
            raise ValueError(
                f"record {index} in {path!r} is missing required fields. "
                f"Found keys {sorted(record)!r}; expected a pair id from "
                f"{list(_PAIR_ID_KEYS)!r}, a question from "
                f"{list(_QUESTION_KEYS)!r} and an answer from "
                f"{list(_ANSWER_KEYS)!r}."
            )

        key = normalise_answer(answer)
        if key not in _ANSWER_LOOKUP:
            raise ValueError(
                f"record {index} in {path!r} has answer {answer!r}, which is "
                f"outside the 19-way CDVQA vocabulary. Scoring against a "
                f"vocabulary we know to be wrong would misreport accuracy."
            )

        samples.append(
            CDVQASample(
                pair_id=str(pair_id),
                question=str(question),
                answer=_ANSWER_LOOKUP[key],
                question_type=str(question_type) if question_type else "unknown",
            )
        )

    return samples


def assert_no_pair_leakage(
    train: Sequence[CDVQASample], test: Sequence[CDVQASample]
) -> None:
    """Fail loudly if any image pair appears in both splits.

    CDVQA splits by image pair, so a single shared pair invalidates every
    number downstream. This is an assertion, not a warning, on purpose.
    """
    train_pairs = {sample.pair_id for sample in train}
    test_pairs = {sample.pair_id for sample in test}
    overlap = sorted(train_pairs & test_pairs)
    assert not overlap, (
        f"{len(overlap)} image pair(s) appear in both the training and test "
        f"splits, which invalidates every reported metric. "
        f"Offending pair ids (first 10): {overlap[:10]!r}"
    )


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

Predictor = Callable[[CDVQASample], str]


def majority_class_baseline(train: Sequence[CDVQASample]) -> Predictor:
    """Always predict the single commonest answer in the training split."""
    if not train:
        raise ValueError("cannot fit a baseline on an empty training split")
    commonest = Counter(sample.answer for sample in train).most_common(1)[0][0]

    def predict(sample: CDVQASample) -> str:
        return commonest

    return predict


def per_type_majority_baseline(train: Sequence[CDVQASample]) -> Predictor:
    """Predict the commonest training answer *for the question's type*.

    A stronger and fairer floor than the global majority: question type is
    given at inference time, so using it costs nothing and any model that
    cannot beat this has learned nothing about the imagery.
    """
    if not train:
        raise ValueError("cannot fit a baseline on an empty training split")

    by_type: Dict[str, Counter] = {}
    for sample in train:
        by_type.setdefault(sample.question_type, Counter())[sample.answer] += 1

    commonest_by_type = {
        question_type: counts.most_common(1)[0][0]
        for question_type, counts in by_type.items()
    }
    overall = Counter(sample.answer for sample in train).most_common(1)[0][0]

    def predict(sample: CDVQASample) -> str:
        return commonest_by_type.get(sample.question_type, overall)

    return predict


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def evaluate(
    samples: Sequence[CDVQASample], predictions: Sequence[str]
) -> Dict[str, Any]:
    """Score predictions against ground truth, per question type and overall.

    ``accuracy`` is None for a question type with no samples -- reporting 0.0
    would be a measurement that never happened, and would drag the average
    down by a fifth per empty type.
    """
    if len(samples) != len(predictions):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(samples)} samples; "
            "a length mismatch would silently score a subset"
        )

    totals: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    hits: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}

    for sample, prediction in zip(samples, predictions):
        question_type = sample.question_type
        totals.setdefault(question_type, 0)
        hits.setdefault(question_type, 0)
        totals[question_type] += 1
        if normalise_answer(prediction) == normalise_answer(sample.answer):
            hits[question_type] += 1

    per_type: Dict[str, Dict[str, Any]] = {}
    for question_type, total in totals.items():
        per_type[question_type] = {
            "n": total,
            "correct": hits[question_type],
            "accuracy": (hits[question_type] / total) if total else None,
        }

    measured = [
        entry["accuracy"]
        for entry in per_type.values()
        if entry["accuracy"] is not None
    ]
    total_n = sum(totals.values())

    return {
        "per_type": per_type,
        "average_accuracy": (sum(measured) / len(measured)) if measured else None,
        "overall_accuracy": (
            sum(hits.values()) / total_n if total_n else None
        ),
        "n": total_n,
    }


def format_report(report: Dict[str, Any], title: str) -> str:
    """Render a report in the paper's table shape, for the terminal."""
    lines = [title, "-" * len(title)]
    for question_type in QUESTION_TYPES:
        entry = report["per_type"].get(question_type, {"n": 0, "accuracy": None})
        accuracy = entry["accuracy"]
        rendered = (
            "NOT_YET_MEASURED" if accuracy is None else f"{accuracy * 100:6.2f}%"
        )
        lines.append(f"  {question_type:<26} {rendered:>16}  (n={entry['n']})")
    for label, key in (("Average Accuracy", "average_accuracy"),
                       ("Overall Accuracy", "overall_accuracy")):
        value = report[key]
        rendered = (
            "NOT_YET_MEASURED" if value is None else f"{value * 100:6.2f}%"
        )
        lines.append(f"  {label:<26} {rendered:>16}")
    lines.append(f"  {'Samples':<26} {report['n']:>16}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a predictor on the official CDVQA test split."
    )
    parser.add_argument("--train", required=True, help="training split JSON")
    parser.add_argument("--test", required=True, help="test split JSON")
    parser.add_argument(
        "--baseline",
        choices=("majority", "per_type_majority"),
        default="per_type_majority",
        help="which baseline to score (default: per_type_majority)",
    )
    parser.add_argument(
        "--out",
        default="outputs/cdvqa_predictions.json",
        help="where to write raw predictions",
    )
    args = parser.parse_args(argv)

    train = load_split(args.train)
    test = load_split(args.test)
    assert_no_pair_leakage(train, test)

    fit = (
        majority_class_baseline
        if args.baseline == "majority"
        else per_type_majority_baseline
    )
    predict = fit(train)
    predictions = [predict(sample) for sample in test]
    report = evaluate(test, predictions)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline": args.baseline,
                "train_split": args.train,
                "test_split": args.test,
                "report": report,
                "predictions": [
                    {**asdict(sample), "prediction": prediction}
                    for sample, prediction in zip(test, predictions)
                ],
            },
            handle,
            indent=2,
        )

    print(format_report(report, f"CDVQA -- baseline: {args.baseline}"))
    print(f"\nRaw predictions written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
