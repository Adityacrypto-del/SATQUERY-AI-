"""Deterministic tests for Qwen prompt templates.

These tests validate prompt construction and content.
No model loading or inference is required.
"""

from modules.qwen.prompts import (
    bitemporal_prompt,
    final_reasoning_prompt,
    single_image_prompt,
)


def test_single_image_prompt_contains_query():
    prompt = single_image_prompt("What land cover is visible?")

    assert "What land cover is visible?" in prompt
    assert "remote-sensing" in prompt.lower()


def test_single_image_prompt_warns_against_inventing():
    prompt = single_image_prompt("Describe this area.")

    assert "Do not invent" in prompt


def test_bitemporal_prompt_contains_query():
    prompt = bitemporal_prompt("What changed between the images?")

    assert "What changed between the images?" in prompt


def test_bitemporal_prompt_distinguishes_temporal_order():
    prompt = bitemporal_prompt("Describe changes.")

    assert "T1" in prompt
    assert "T2" in prompt
    assert "EARLIER" in prompt or "earlier" in prompt
    assert "LATER" in prompt or "later" in prompt


def test_bitemporal_prompt_warns_against_confusion():
    prompt = bitemporal_prompt("What changed?")

    assert "Do not confuse" in prompt


def test_bitemporal_prompt_warns_against_inventing():
    prompt = bitemporal_prompt("What changed?")

    assert "Do not invent" in prompt


def test_final_reasoning_prompt_contains_all_inputs():
    prompt = final_reasoning_prompt(
        query="Did vegetation increase?",
        qwen_observation="I see more green in T2.",
        specialist_summary="NDVI increased in region A.",
        measurements="changed_pixels=11940, area=47760 m2",
    )

    assert "Did vegetation increase?" in prompt
    assert "I see more green in T2." in prompt
    assert "NDVI increased in region A." in prompt
    assert "changed_pixels=11940" in prompt


def test_final_reasoning_prompt_enforces_specialist_authority():
    prompt = final_reasoning_prompt(
        query="test",
        qwen_observation="obs",
        specialist_summary="sum",
        measurements="meas",
    )

    lower = prompt.lower()
    assert "authoritative" in lower
    assert "do not" in lower
    assert "override" in lower or "invent" in lower
