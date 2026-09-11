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


# --------------------------------------------------------------------------
# Natural phrasing, not only the benchmark's
# --------------------------------------------------------------------------


def test_natural_phrasings_route_the_way_a_person_would_expect():
    """CDVQA asks "Have the areas of water changed?". A person asks "are
    there any water bodies erased?" -- and that used to fall through to the
    descriptive path even though the target class parsed correctly. The
    benchmark number is unaffected either way, but a specialist a controller
    routes real user queries to has to understand more than one dialect.
    """
    cases = {
        "are there any water bodies erased?": "decrease_or_not",
        "did the water disappear?": "decrease_or_not",
        "was any water lost?": "decrease_or_not",
        "were any trees cleared?": "decrease_or_not",
        "have buildings gone up?": "increase_or_not",
        "did new buildings appear?": "change_or_not",
        "what did the trees turn into?": "change_to_what",
        "Which class changed the most?": "largest_change",
        "Which class changed the least?": "smallest_change",
    }
    for question, expected in cases.items():
        assert classify_question(question) == expected, question


def test_broadening_did_not_steal_from_the_general_case():
    """largest_change is tried before change_or_not, so a bare "most" would
    capture a question that is really about whether a class changed."""
    assert classify_question("most of the water changed, right?") == "change_or_not"


def test_a_question_naming_no_class_still_refuses_to_route():
    """The guard the broadening must not weaken. "What changed and where?"
    has no answer among CDVQA's 19 tokens; the honest output is a
    description, not a class invented to fill the slot."""
    assert classify_question("what changed and where?") is None
    assert classify_question("is there less vegetation now?") is None, (
        "bare 'vegetation' is ambiguous between low vegetation and trees"
    )


def test_every_cdvqa_question_still_routes_to_its_declared_type():
    """The regression that matters: broadening a pattern can only steal a
    question from a later pattern, so this asserts none was stolen. Measured
    across the full validation split, not a sample."""
    from eval.run_cdvqa import load_cdvqa_split

    try:
        samples = load_cdvqa_split("datasets/CDVQA", "Val")
    except Exception:  # noqa: BLE001 - dataset absent in a bare checkout
        import pytest

        pytest.skip("CDVQA dataset not available")

    wrong = [s for s in samples if classify_question(s.question) != s.question_type]
    assert not wrong, (
        f"{len(wrong)} of {len(samples)} re-routed, e.g. "
        f"{wrong[0].question!r} -> {classify_question(wrong[0].question)}"
    )
