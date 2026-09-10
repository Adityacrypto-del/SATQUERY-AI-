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
    "CDVQA_GROUPS",
    "QUESTION_TYPES",
    "QUESTION_TYPE_GROUPS",
    "group_of",
    "load_cdvqa_split",
    "CDVQASample",
    "assert_no_pair_leakage",
    "evaluate",
    "load_split",
    "majority_class_baseline",
    "per_type_majority_baseline",
]

# The 19 answers, verbatim as they appear in the released CDVQA JSON, in
# training-set frequency order. CLAUDE.md section 2 paraphrases these with
# percentage signs and spaces ("0%-10%", "NVG surface"); the dataset itself
# uses underscores. Scoring compares against the dataset, so these strings
# are the authority and the paraphrase is not.
CDVQA_ANSWERS = (
    "no",
    "yes",
    "NVG_surface",
    "0",
    "0_to_10",
    "buildings",
    "low_vegetation",
    "trees",
    "10_to_20",
    "water",
    "80_to_90",
    "20_to_30",
    "90_to_100",
    "70_to_80",
    "30_to_40",
    "60_to_70",
    "40_to_50",
    "playgrounds",
    "50_to_60",
)

# The eight question types the dataset actually declares.
QUESTION_TYPES = (
    "change_or_not",
    "increase_or_not",
    "decrease_or_not",
    "change_to_what",
    "largest_change",
    "smallest_change",
    "change_ratio",
    "change_ratio_types",
)

# The five rule families in CLAUDE.md section 2 are a grouping of those eight:
# increase/decrease are asked separately, as are largest/smallest, and the
# ratio rule covers both the whole-scene and the per-class question.
# Reported both ways because it is not established which granularity the
# paper's table uses, and picking one silently would make our numbers
# look comparable when they might not be.
QUESTION_TYPE_GROUPS = {
    "change_or_not": "change_or_not",
    "increase_or_not": "increase_or_decrease",
    "decrease_or_not": "increase_or_decrease",
    "change_to_what": "change_to_what",
    "largest_change": "largest_smallest_change",
    "smallest_change": "largest_smallest_change",
    "change_ratio": "change_ratio",
    "change_ratio_types": "change_ratio",
}

CDVQA_GROUPS = (
    "change_or_not",
    "increase_or_decrease",
    "change_to_what",
    "largest_smallest_change",
    "change_ratio",
)


def group_of(question_type: str) -> str:
    """Map a native question type onto its CLAUDE.md rule family."""
    return QUESTION_TYPE_GROUPS.get(question_type, question_type)

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


def load_cdvqa_split(root: str, split: str) -> List[CDVQASample]:
    """Load one official CDVQA split from its released three-file form.

    The release stores a split as ``<Split>_images.json`` /
    ``_questions.json`` / ``_answers.json`` and joins them by id. An image
    entry is a *question group*, not a scene: roughly sixteen entries share
    one ``file_name``. The scene filename is what identifies the image pair,
    so that -- not the entry id -- becomes ``pair_id`` and therefore the key
    the leakage check works on.
    """
    paths = {
        part: os.path.join(root, f"{split}_{part}.json")
        for part in ("images", "questions", "answers")
    }
    for part, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CDVQA {split} {part} not found: {path!r}. "
                "Accuracy is NOT_YET_MEASURED until the official split is present."
            )

    def _read(part: str) -> List[Dict[str, Any]]:
        with open(paths[part], "r", encoding="utf-8") as handle:
            return json.load(handle)[part]

    file_name_by_entry = {
        entry["id"]: entry["file_name"] for entry in _read("images")
    }
    answer_by_id = {row["id"]: row["answer"] for row in _read("answers")}

    samples: List[CDVQASample] = []
    for question in _read("questions"):
        answer_ids = question.get("answers_ids") or []
        if len(answer_ids) != 1:
            raise ValueError(
                f"question {question['id']} in {split} has {len(answer_ids)} "
                "answers; CDVQA is single-answer classification"
            )
        answer = answer_by_id[answer_ids[0]]
        key = normalise_answer(answer)
        if key not in _ANSWER_LOOKUP:
            raise ValueError(
                f"question {question['id']} in {split} has answer {answer!r}, "
                "which is outside the 19-way CDVQA vocabulary."
            )
        pair_id = file_name_by_entry[question["img_id"]]
        samples.append(
            CDVQASample(
                pair_id=pair_id,
                question=question["question"],
                answer=_ANSWER_LOOKUP[key],
                question_type=question["type"],
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
    samples: Sequence[CDVQASample],
    predictions: Sequence[str],
    by: str = "type",
) -> Dict[str, Any]:
    """Score predictions against ground truth, per question type and overall.

    ``by="type"`` keys on the eight native CDVQA types; ``by="group"`` keys
    on the five rule families from CLAUDE.md section 2.

    ``accuracy`` is None for a question type with no samples -- reporting 0.0
    would be a measurement that never happened, and would drag the average
    down by one slot per empty type.
    """
    if len(samples) != len(predictions):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(samples)} samples; "
            "a length mismatch would silently score a subset"
        )
    if by not in ("type", "group"):
        raise ValueError(f"by must be 'type' or 'group'; got {by!r}")

    universe = QUESTION_TYPES if by == "type" else CDVQA_GROUPS
    key_of = (lambda s: s.question_type) if by == "type" else (
        lambda s: group_of(s.question_type)
    )

    totals: Dict[str, int] = {t: 0 for t in universe}
    hits: Dict[str, int] = {t: 0 for t in universe}

    for sample, prediction in zip(samples, predictions):
        question_type = key_of(sample)
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
    for question_type in report["per_type"]:
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
    parser.add_argument(
        "--root", help="directory holding the official CDVQA JSON files"
    )
    parser.add_argument("--train", help="training split JSON (flat format)")
    parser.add_argument("--test", help="test split JSON (flat format)")
    parser.add_argument(
        "--train-split", default="Train", help="official split name to fit on"
    )
    parser.add_argument(
        "--test-split",
        default="Test",
        help="official split name to score. Test and Test2 cover the same 968 "
             "scenes with different question sets, so the choice is explicit.",
    )
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

    if args.root:
        train = load_cdvqa_split(args.root, args.train_split)
        test = load_cdvqa_split(args.root, args.test_split)
    elif args.train and args.test:
        train = load_split(args.train)
        test = load_split(args.test)
    else:
        parser.error("pass --root, or both --train and --test")
    assert_no_pair_leakage(train, test)

    fit = (
        majority_class_baseline
        if args.baseline == "majority"
        else per_type_majority_baseline
    )
    predict = fit(train)
    predictions = [predict(sample) for sample in test]
    report = evaluate(test, predictions, by="type")
    grouped = evaluate(test, predictions, by="group")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline": args.baseline,
                "train_split": args.train_split if args.root else args.train,
                "test_split": args.test_split if args.root else args.test,
                "report": report,
                "report_grouped": grouped,
                "predictions": [
                    {**asdict(sample), "prediction": prediction}
                    for sample, prediction in zip(test, predictions)
                ],
            },
            handle,
            indent=2,
        )

    print(format_report(report, f"CDVQA native types -- baseline: {args.baseline}"))
    print()
    print(format_report(grouped, "CDVQA rule families (CLAUDE.md section 2)"))
    print(f"\nRaw predictions written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
