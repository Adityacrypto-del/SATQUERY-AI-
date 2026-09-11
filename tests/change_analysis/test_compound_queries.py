"""Tests for compound (multi-class) questions.

CDVQA questions name exactly one land-cover class. A user asking "what did
low vegetation and water change into" is asking two CDVQA questions at once,
and the benchmark's 19-answer vocabulary has no token for a combined answer.

So the design decision under test is a refusal as much as a feature: a
compound question yields one answer *per named class*, never a single
fabricated answer. There is no defined rule for combining "yes" and "no" into
one token -- AND and OR are both defensible and CDVQA specifies neither -- and
picking one would be inventing benchmark semantics (HARD RULE 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.change_analysis.cdvqa import (
    answer_compound,
    is_compound,
    parse_targets,
)


def _maps():
    """t1: left half low vegetation, right half water. t2: all buildings."""
    s1 = np.zeros((8, 8), dtype=np.int64)
    s1[:, :4] = 2   # low_vegetation
    s1[:, 4:] = 5   # water
    s2 = np.full((8, 8), 4, dtype=np.int64)  # buildings
    return s1, s2


# --------------------------------------------------------------------------
# Parsing every named class, not just the first
# --------------------------------------------------------------------------


def test_parses_both_classes_in_order_of_appearance():
    assert parse_targets("what did low vegetation and water change into") == [
        "low_vegetation", "water"
    ]


def test_singular_and_plural_phrases_resolve_to_one_class_each():
    """"buildings" must not also match "building" and double-count."""
    assert parse_targets("have buildings and playgrounds changed?") == [
        "buildings", "playgrounds"
    ]


def test_longer_phrase_wins_over_a_shorter_one_inside_it():
    """"low vegetation" must not be read as a bare vegetation match, and
    "non-vegetated ground surface" must not be split apart."""
    assert parse_targets(
        "did non-vegetated ground surface and trees change?"
    ) == ["NVG_surface", "trees"]


def test_a_repeated_class_is_named_once():
    assert parse_targets("did water increase, and did water decrease?") == ["water"]


def test_single_class_questions_are_not_compound():
    assert parse_targets("have the areas of water changed?") == ["water"]
    assert is_compound("have the areas of water changed?") is False


def test_a_question_naming_no_class_yields_no_targets():
    """"vegetation" alone is ambiguous between low vegetation and trees, so
    it maps to nothing rather than to a guess."""
    assert parse_targets("what changed and where?") == []
    assert parse_targets("how much vegetation is there?") == []
    assert is_compound("what changed and where?") is False


# --------------------------------------------------------------------------
# Answering: one answer per class, never a combined one
# --------------------------------------------------------------------------


def test_compound_change_to_what_answers_each_class_separately():
    s1, s2 = _maps()

    result = answer_compound(
        "what did low vegetation and water change into", "change_to_what", s1, s2
    )

    assert [a.answer for a in result.answers] == ["buildings", "buildings"]
    assert result.targets == ["low_vegetation", "water"]


def test_each_sub_answer_is_a_real_cdvqa_answer_with_its_own_evidence():
    """The per-class answers must be the same objects the single-class path
    produces, so a compound question is not a second, weaker implementation
    of the rules."""
    from eval.run_cdvqa import CDVQA_ANSWERS
    from tools.change_analysis.cdvqa import answer_question

    s1, s2 = _maps()
    result = answer_compound(
        "have low vegetation and water changed?", "change_or_not", s1, s2
    )

    for target, sub in zip(result.targets, result.answers):
        assert sub.answer in CDVQA_ANSWERS
        assert sub.evidence
        direct = answer_question(
            f"have the areas of {target.replace('_', ' ')} changed?",
            "change_or_not", s1, s2,
        )
        assert sub.answer == direct.answer


def test_no_single_combined_answer_is_invented():
    """The whole point. A compound result exposes per-class answers and does
    not collapse them into one of the 19 tokens."""
    s1, s2 = _maps()
    s1[:, 4:] = 0  # water no longer changes; the two classes now disagree

    result = answer_compound(
        "have low vegetation and water changed?", "change_or_not", s1, s2
    )

    assert [a.answer for a in result.answers] == ["yes", "no"]
    assert not hasattr(result, "answer")


def test_compound_result_summarises_itself_for_the_trace():
    s1, s2 = _maps()

    result = answer_compound(
        "what did low vegetation and water change into", "change_to_what", s1, s2
    )
    text = result.summary()

    assert "low vegetation" in text and "water" in text and "buildings" in text


def test_a_single_target_still_returns_one_answer_not_a_special_case():
    s1, s2 = _maps()

    result = answer_compound("have buildings changed?", "change_or_not", s1, s2)

    assert result.targets == ["buildings"]
    assert len(result.answers) == 1
