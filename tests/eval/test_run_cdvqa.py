"""Tests for the CDVQA evaluation harness (build-order step 3).

The point of this step is a real measured number on day one. So the tests
here are weighted toward the things that would make that number a lie:
leakage between splits, a metric that silently divides by zero, and any path
that reports an accuracy when no evaluation actually ran.
"""

from __future__ import annotations

import json

import pytest

from eval.run_cdvqa import (
    CDVQA_ANSWERS,
    QUESTION_TYPES,
    CDVQASample,
    assert_no_pair_leakage,
    evaluate,
    load_split,
    per_type_majority_baseline,
    majority_class_baseline,
)


def _sample(pair_id, question_type, answer, question="q?"):
    return CDVQASample(
        pair_id=pair_id,
        question=question,
        answer=answer,
        question_type=question_type,
    )


def _write_split(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# The answer vocabulary is fixed at 19 -- this is classification, not
# generation, and a 20th string would silently break comparability.
# --------------------------------------------------------------------------


def test_answer_vocabulary_is_the_official_nineteen():
    assert len(CDVQA_ANSWERS) == 19
    assert len(set(CDVQA_ANSWERS)) == 19
    for expected in ("no", "yes", "0", "water", "playgrounds", "90%-100%"):
        assert expected in CDVQA_ANSWERS


def test_all_five_question_types_are_declared():
    assert len(QUESTION_TYPES) == 5


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_per_type_and_overall_accuracy_are_computed_separately():
    samples = [
        _sample("p1", "change_or_not", "yes"),
        _sample("p2", "change_or_not", "no"),
        _sample("p3", "change_ratio", "0%-10%"),
        _sample("p4", "change_ratio", "10%-20%"),
        _sample("p5", "change_ratio", "20%-30%"),
        _sample("p6", "change_ratio", "30%-40%"),
    ]
    # change_or_not: 1/2 correct. change_ratio: 3/4 correct.
    predictions = ["yes", "yes", "0%-10%", "10%-20%", "20%-30%", "0%-10%"]

    report = evaluate(samples, predictions)

    assert report["per_type"]["change_or_not"]["accuracy"] == pytest.approx(0.5)
    assert report["per_type"]["change_ratio"]["accuracy"] == pytest.approx(0.75)
    # Overall is micro-averaged over samples: 4 of 6.
    assert report["overall_accuracy"] == pytest.approx(4 / 6)
    # Average is the unweighted mean over types, as the paper reports it.
    assert report["average_accuracy"] == pytest.approx(0.625)


def test_question_type_with_no_samples_does_not_divide_by_zero():
    samples = [_sample("p1", "change_or_not", "yes")]

    report = evaluate(samples, ["yes"])

    absent = report["per_type"]["change_ratio"]
    assert absent["n"] == 0
    assert absent["accuracy"] is None
    # A type with no samples must not drag the average toward zero.
    assert report["average_accuracy"] == pytest.approx(1.0)


def test_answers_are_compared_case_and_whitespace_insensitively():
    samples = [_sample("p1", "change_or_not", "Yes ")]

    report = evaluate(samples, ["yes"])

    assert report["overall_accuracy"] == pytest.approx(1.0)


def test_prediction_count_must_match_sample_count():
    """A silent zip() truncation would report accuracy on a subset."""
    samples = [_sample("p1", "change_or_not", "yes")]

    with pytest.raises(ValueError):
        evaluate(samples, ["yes", "no"])


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_majority_class_baseline_predicts_the_commonest_training_answer():
    train = [
        _sample("p1", "change_or_not", "no"),
        _sample("p2", "change_or_not", "no"),
        _sample("p3", "change_ratio", "0"),
    ]

    predict = majority_class_baseline(train)

    assert predict(_sample("t1", "change_ratio", "water")) == "no"


def test_per_type_majority_baseline_is_conditioned_on_question_type():
    train = [
        _sample("p1", "change_or_not", "no"),
        _sample("p2", "change_or_not", "no"),
        _sample("p3", "change_ratio", "0"),
        _sample("p4", "change_ratio", "0"),
        _sample("p5", "change_ratio", "10%-20%"),
    ]

    predict = per_type_majority_baseline(train)

    assert predict(_sample("t1", "change_or_not", "yes")) == "no"
    assert predict(_sample("t2", "change_ratio", "yes")) == "0"


# --------------------------------------------------------------------------
# Leakage (CLAUDE.md section 2 -- fail loudly, never warn quietly)
# --------------------------------------------------------------------------


def test_pair_id_shared_between_splits_fails_loudly():
    train = [_sample("pair_7", "change_or_not", "yes")]
    test = [_sample("pair_7", "change_ratio", "0")]

    with pytest.raises(AssertionError) as excinfo:
        assert_no_pair_leakage(train, test)

    assert "pair_7" in str(excinfo.value)


def test_disjoint_splits_pass_the_leakage_check():
    train = [_sample("pair_1", "change_or_not", "yes")]
    test = [_sample("pair_2", "change_ratio", "0")]

    assert_no_pair_leakage(train, test) is None


# --------------------------------------------------------------------------
# Loading, and refusing to invent a number (HARD RULE 3)
# --------------------------------------------------------------------------


def test_load_split_reads_records(tmp_path):
    path = _write_split(
        tmp_path / "test.json",
        [
            {
                "pair_id": "p1",
                "question": "is there any change?",
                "answer": "yes",
                "question_type": "change_or_not",
            }
        ],
    )

    samples = load_split(path)

    assert len(samples) == 1
    assert samples[0].pair_id == "p1"
    assert samples[0].answer == "yes"


def test_missing_dataset_raises_rather_than_reporting_zero(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(str(tmp_path / "absent.json"))


def test_unrecognised_answer_in_the_data_fails_loudly(tmp_path):
    """A 20th answer string means our vocabulary is wrong; do not score it."""
    path = _write_split(
        tmp_path / "test.json",
        [
            {
                "pair_id": "p1",
                "question": "q",
                "answer": "purple",
                "question_type": "change_or_not",
            }
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        load_split(path)

    assert "purple" in str(excinfo.value)


def test_unrecognised_field_names_report_what_was_actually_found(tmp_path):
    """The public CDVQA release may key its JSON differently; say so clearly."""
    path = _write_split(
        tmp_path / "test.json", [{"img": "p1", "q": "q", "a": "yes"}]
    )

    with pytest.raises(ValueError) as excinfo:
        load_split(path)

    message = str(excinfo.value)
    assert "img" in message and "pair_id" in message
