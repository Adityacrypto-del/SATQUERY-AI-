"""Tests for the CDVQA rule implementation (oracle).

These lock in rules that were fitted empirically against gold answers rather
than assumed, so a regression here would silently move the ceiling.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eval.oracle_cdvqa import (
    PALETTE,
    decode_label,
    oracle_answer,
    parse_question,
    question_side,
    ratio_bin,
)


def _maps(pairs, size=10):
    """Build (s1, s2) where ``pairs`` is a list of (class1, class2, count)."""
    s1 = np.zeros(size * size, dtype=np.uint8)
    s2 = np.zeros(size * size, dtype=np.uint8)
    at = 0
    for c1, c2, count in pairs:
        s1[at:at + count] = c1
        s2[at:at + count] = c2
        at += count
    return s1.reshape(size, size), s2.reshape(size, size)


def _answer(qtype, question, s1, s2):
    return oracle_answer(qtype, parse_question(question, qtype), s1, s2)


# --------------------------------------------------------------------------
# Palette decoding
# --------------------------------------------------------------------------


def test_decode_maps_the_seven_palette_colours(tmp_path):
    colours = list(PALETTE)
    rgb = np.array(colours, dtype=np.uint8).reshape(1, 7, 3)
    path = tmp_path / "label.png"
    Image.fromarray(rgb).save(path)

    decoded = decode_label(str(path))

    assert decoded.tolist() == [[PALETTE[c] for c in colours]]


def test_decode_rejects_off_palette_colour(tmp_path):
    """An unmapped colour would silently become a wrong class."""
    rgb = np.array([[[7, 7, 7]]], dtype=np.uint8)
    path = tmp_path / "bad.png"
    Image.fromarray(rgb).save(path)

    with pytest.raises(ValueError):
        decode_label(str(path))


# --------------------------------------------------------------------------
# Ratio binning
# --------------------------------------------------------------------------


def test_zero_is_its_own_bin_and_decades_are_half_open():
    assert ratio_bin(0.0) == "0"
    assert ratio_bin(0.01) == "0_to_10"
    assert ratio_bin(10.0) == "0_to_10"
    assert ratio_bin(10.01) == "10_to_20"
    assert ratio_bin(100.0) == "90_to_100"


# --------------------------------------------------------------------------
# Which map a question refers to -- the highest-leverage rule detail
# --------------------------------------------------------------------------


def test_question_side_detects_pre_post_and_absence():
    assert question_side("What is the largest change in the second image?") == "post"
    assert question_side("What is the largest change in the pre-event image?") == "pre"
    assert question_side("What is the largest change?") == "none"


def test_largest_change_uses_the_named_map_only():
    # NVG (1) dominates at t1; buildings (4) dominate at t2.
    s1, s2 = _maps([(1, 4, 60), (4, 1, 10)])

    assert _answer("largest_change", "What is the largest change in the first image?", s1, s2) == "NVG_surface"
    assert _answer("largest_change", "What is the largest change in the second image?", s1, s2) == "buildings"


def test_unqualified_largest_change_uses_both_maps():
    """No image named means the union, not a silent default to t1."""
    s1, s2 = _maps([(1, 4, 30), (3, 3, 45)])

    # trees (3) totals 90 across both maps; NVG 30 and buildings 30.
    assert _answer("largest_change", "What is the largest change?", s1, s2) == "trees"


def test_absent_classes_are_not_candidates_for_smallest_change():
    s1, s2 = _maps([(1, 4, 50), (4, 1, 5)])

    # water/trees/playgrounds have zero changed area and must not win.
    assert _answer("smallest_change", "What is the smallest change?", s1, s2) in (
        "NVG_surface", "buildings"
    )


# --------------------------------------------------------------------------
# The remaining rules
# --------------------------------------------------------------------------


def test_change_or_not_is_scoped_by_the_named_map():
    s1, s2 = _maps([(1, 4, 20)])

    q = "Did the areas of buildings change in the pre-change image?"
    assert _answer("change_or_not", q, s1, s2) == "no"
    q = "Did the areas of buildings change in the post-event image?"
    assert _answer("change_or_not", q, s1, s2) == "yes"


def test_increase_and_decrease_compare_areas_across_time():
    s1, s2 = _maps([(1, 4, 30)])

    # NVG (30 -> 0) shrank; buildings (0 -> 30) grew.
    assert _answer("increase_or_not", "Did the regions of buildings increase?", s1, s2) == "yes"
    assert _answer("decrease_or_not", "Did the regions of buildings decrease?", s1, s2) == "no"
    assert _answer("increase_or_not", "Did the regions of non-vegetated ground surface increase?", s1, s2) == "no"
    assert _answer("decrease_or_not", "Did the regions of non-vegetated ground surface decrease?", s1, s2) == "yes"


def test_change_to_what_takes_the_majority_destination():
    s1, s2 = _maps([(1, 4, 30), (1, 3, 10)])

    q = "What have the areas of non-vegetated ground surface in the first image mainly changed to?"
    assert _answer("change_to_what", q, s1, s2) == "buildings"


def test_change_ratio_inverts_for_unchanged_phrasing():
    s1, s2 = _maps([(1, 4, 25)])  # 25 of 100 pixels changed

    assert _answer("change_ratio", "What is the percentage of changed areas?", s1, s2) == "20_to_30"
    assert _answer("change_ratio", "What is the percentage of unchanged areas?", s1, s2) == "70_to_80"


def test_change_ratio_types_is_scoped_by_map():
    s1, s2 = _maps([(1, 4, 15)])

    pre = "What is the change ratio of non-vegetated ground surface in the pre-event image?"
    post = "What is the change ratio of non-vegetated ground surface in the post-event image?"
    assert _answer("change_ratio_types", pre, s1, s2) == "10_to_20"
    assert _answer("change_ratio_types", post, s1, s2) == "0"
