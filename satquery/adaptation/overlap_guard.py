"""Train/eval disjointness guard for LoRA adaptation data.

A training record and an eval record are the same image if any of these match:
1. resolved file path;
2. SHA-256 of the file contents (a renamed or re-located copy is still identity);
3. a dataset key (``image_key``), for sources whose images live in a store
   rather than as files -- e.g. ``ben:<BigEarthNet patch_id>``.

An eval manifest that cannot be read is reported as *unchecked* and the report
is not ok: a guard that passes because the eval data was missing proves nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


class TrainEvalOverlapError(RuntimeError):
    pass


@dataclass
class OverlapReport:
    n_train: int
    n_train_unidentified: int
    checked: List[str] = field(default_factory=list)
    unchecked: List[str] = field(default_factory=list)
    n_eval: Dict[str, int] = field(default_factory=dict)
    path_overlaps: List[dict] = field(default_factory=list)
    hash_overlaps: List[dict] = field(default_factory=list)
    key_overlaps: List[dict] = field(default_factory=list)

    @property
    def n_overlaps(self) -> int:
        return len(self.path_overlaps) + len(self.hash_overlaps) + len(self.key_overlaps)

    @property
    def ok(self) -> bool:
        return not self.unchecked and self.n_overlaps == 0

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok, "n_overlaps": self.n_overlaps}


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.realpath(p))


_hash_cache: Dict[str, str] = {}


def _sha256(path: str) -> str:
    key = _norm_path(path)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def _index(records: Iterable[dict]) -> tuple[dict, dict, dict, int]:
    """Return (path -> rec, hash -> rec, key -> rec, n_unidentified)."""
    by_path, by_hash, by_key, unidentified = {}, {}, {}, 0
    for rec in records:
        path, key = rec.get("image_path"), rec.get("image_key")
        if key:
            by_key.setdefault(str(key), rec)
        if path and Path(path).is_file():
            by_path.setdefault(_norm_path(path), rec)
            by_hash.setdefault(_sha256(path), rec)
        elif not key:
            unidentified += 1
    return by_path, by_hash, by_key, unidentified


def _load_manifest(path: str) -> Optional[List[dict]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def check_overlap(train_records: List[dict], eval_manifests: Mapping[str, str]) -> OverlapReport:
    t_path, t_hash, t_key, unidentified = _index(train_records)
    report = OverlapReport(n_train=len(train_records), n_train_unidentified=unidentified)

    for name, manifest in eval_manifests.items():
        records = _load_manifest(manifest)
        if records is None:
            report.unchecked.append(name)
            continue
        report.checked.append(name)
        report.n_eval[name] = len(records)
        e_path, e_hash, e_key, _ = _index(records)
        for p in t_path.keys() & e_path.keys():
            report.path_overlaps.append({"eval": name, "path": p})
        path_hits = {_sha256(p) for p in t_path.keys() & e_path.keys()}
        for h in (t_hash.keys() & e_hash.keys()) - path_hits:
            report.hash_overlaps.append({
                "eval": name, "sha256": h,
                "train_path": t_hash[h].get("image_path"), "eval_path": e_hash[h].get("image_path"),
            })
        for k in t_key.keys() & e_key.keys():
            report.key_overlaps.append({"eval": name, "image_key": k})
    return report


def assert_no_overlap(train_records: List[dict], eval_manifests: Mapping[str, str]) -> OverlapReport:
    report = check_overlap(train_records, eval_manifests)
    if not report.ok:
        raise TrainEvalOverlapError(
            f"train/eval overlap guard FAILED: {report.n_overlaps} overlapping image(s) "
            f"(path {len(report.path_overlaps)}, content-hash {len(report.hash_overlaps)}, "
            f"key {len(report.key_overlaps)}); unchecked eval manifests: {report.unchecked or 'none'}"
        )
    return report
