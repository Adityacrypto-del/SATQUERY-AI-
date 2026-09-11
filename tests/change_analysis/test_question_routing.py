"""Tests for free-text question -> CDVQA question-type routing."""

from __future__ import annotations

import pytest

from tools.change_analysis.cdvqa import classify_question


@pytest.mark.parametrize("text,expected", [
    ("Have the areas of water changed?", "change_or_not"),
    ("Did the regions of buildings increase?", "increase_or_not"),
    ("Have the areas of trees decreased?", "decrease_or_not"),
    ("What have the areas of low vegetation mainly changed to?", "change_to_what"),
    ("What is the largest change?", "largest_change"),
    ("What type of change is the smallest?", "smallest_change"),
    ("What is the percentage of changed areas?", "change_ratio"),
    ("What is the change ratio of buildings in the pre-event image?",
     "change_ratio_types"),
])
def test_cdvqa_shaped_questions_route_correctly(text, expected):
    assert classify_question(text) == expected


def test_change_to_what_is_not_shadowed_by_the_generic_change_pattern():
    """Ordering matters: 'changed to' contains 'chang'."""
    assert classify_question(
        "What have the regions of buildings changed into?"
    ) == "change_to_what"


def test_ratio_splits_on_whether_a_class_is_named():
    assert classify_question("How much of the area has changed?") == "change_ratio"
    assert classify_question(
        "What is the change ratio of water?"
    ) == "change_ratio_types"


def test_classless_question_falls_through_rather_than_guessing():
    """'Did it increase?' names no class, so no rule can answer it."""
    assert classify_question("Did it increase?") != "increase_or_not"


@pytest.mark.parametrize("text", [
    "", "   ", "Describe this scene.", "What is the weather like?",
    "Show me the imagery.",
])
def test_non_cdvqa_questions_return_none(text):
    """None routes to the region summary; it must never be a guessed type."""
    assert classify_question(text) is None
