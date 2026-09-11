"""Track D: the LoRA training manifest must share zero images with any eval manifest.

Identity is checked by path, then by content hash (a renamed copy is still the
same image), then by dataset key where records carry one (BigEarthNet patch
IDs). ``prepare_instructions`` must refuse to write ``instructions.json`` when
the check fails -- the original bug was that its fallback trained on the
exact manifest ``evaluate_vrsbench`` scores against.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from satquery.adaptation import prepare_instructions
from satquery.adaptation.overlap_guard import (
    TrainEvalOverlapError,
    assert_no_overlap,
    check_overlap,
)


def _img(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def _manifest(path: Path, records: list[dict]) -> str:
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def test_disjoint_train_and_eval_pass(tmp_path):
    train = [{"image_path": _img(tmp_path / "tr" / "a.jpg", b"train-a")}]
    ev = _manifest(tmp_path / "ev.json", [{"image_path": _img(tmp_path / "ev" / "b.jpg", b"eval-b")}])
    report = check_overlap(train, {"vrsbench": ev})
    assert report.ok
    assert report.checked == ["vrsbench"]


def test_same_path_is_detected(tmp_path):
    shared = _img(tmp_path / "x.jpg", b"same")
    ev = _manifest(tmp_path / "ev.json", [{"image_path": shared}])
    report = check_overlap([{"image_path": shared}], {"vrsbench": ev})
    assert not report.ok
    assert len(report.path_overlaps) == 1


def test_renamed_copy_is_detected_by_content_hash(tmp_path):
    original = _img(tmp_path / "ev" / "orig.jpg", b"identical-bytes")
    copy = tmp_path / "tr" / "renamed.jpg"
    copy.parent.mkdir()
    shutil.copy(original, copy)
    ev = _manifest(tmp_path / "ev.json", [{"image_path": original}])
    report = check_overlap([{"image_path": str(copy)}], {"vrsbench": ev})
    assert not report.ok
    assert report.path_overlaps == []
    assert len(report.hash_overlaps) == 1


def test_dataset_key_overlap_is_detected(tmp_path):
    ev = _manifest(tmp_path / "ev.json", [{"image_key": "ben:S2A_patch_1"}])
    report = check_overlap([{"image_key": "ben:S2A_patch_1"}], {"bigearthnet_bench": ev})
    assert not report.ok
    assert len(report.key_overlaps) == 1


def test_missing_eval_manifest_is_reported_never_silently_passed(tmp_path):
    train = [{"image_path": _img(tmp_path / "a.jpg", b"a")}]
    report = check_overlap(train, {"rsvqa": str(tmp_path / "does_not_exist.json")})
    assert report.unchecked == ["rsvqa"]
    assert not report.ok
    with pytest.raises(TrainEvalOverlapError):
        assert_no_overlap(train, {"rsvqa": str(tmp_path / "does_not_exist.json")})


def test_assert_no_overlap_raises_with_counts(tmp_path):
    shared = _img(tmp_path / "x.jpg", b"same")
    ev = _manifest(tmp_path / "ev.json", [{"image_path": shared}])
    with pytest.raises(TrainEvalOverlapError, match="1"):
        assert_no_overlap([{"image_path": shared}], {"vrsbench": ev})


def test_prepare_refuses_to_write_when_train_is_the_eval_manifest(tmp_path):
    """Regression for the original bug: VRSBench manifest used as both train and eval."""
    img = _img(tmp_path / "v" / "0.jpg", b"vrs-0")
    manifest = _manifest(tmp_path / "manifest.json", [{"image_path": img, "caption": "a field"}])
    out = tmp_path / "out"
    with pytest.raises((TrainEvalOverlapError, SystemExit)):
        prepare_instructions.main([
            "--source", "vrsbench", "--vrs-manifest", manifest,
            "--eval-manifest", f"vrsbench={manifest}", "--out", str(out),
        ])
    assert not (out / "instructions.json").exists()


def test_hf_source_ids_are_the_verified_ones():
    assert prepare_instructions.RS_INSTRUCTIONS_REPO == "BigData-KSU/RS-instructions-dataset"
    assert "EarthVQA/rsvqa-lr" not in json.dumps(prepare_instructions.HF_SOURCES)
