"""Prepare instruction data for LoRA adaptation, with a hard train/eval overlap guard.

Sources (``--source``):
  bigearthnet_txt  (default) BigEarthNet.txt ``split == "train"`` rows -- the PS's
                   named adaptation corpus. Images live in BigEarthNet v2.0 and are
                   referenced by patch ID (``image_key = "ben:<patch_id>"``); the
                   cloud job materialises them. The benchmark / test / validation
                   patches are written to an eval key manifest and guarded against.
  rs_instructions  ``BigData-KSU/RS-instructions-dataset`` (verified to exist and load).
                   NOTE: this repo ships JSON only -- its ``image`` field is a relative
                   path into RSICD / NWPU / UCM / RSVQA-LR / DOTA, not image bytes.
  vrsbench         a local VRSBench-format manifest. Kept only for explicit use: it is
                   the path that used to train on the exact manifest
                   ``evaluate_vrsbench`` scores, and the guard now refuses that.

Nothing is written to ``instructions.json`` unless the overlap guard passes against
every eval manifest. A missing eval manifest fails the guard unless it is named in
``--allow-unchecked`` (recorded in the report).

Usage:
    python -m satquery.adaptation.prepare_instructions --source bigearthnet_txt \\
        --ben-parquet data/bigearthnet_txt/BigEarthNet.txt.parquet --out data/ben_instructions
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from satquery.adaptation.overlap_guard import TrainEvalOverlapError, check_overlap

RS_INSTRUCTIONS_REPO = "BigData-KSU/RS-instructions-dataset"
# Only verified sources. "BigData-KSU/RS-instructions" (no suffix) and
# "EarthVQA/rsvqa-lr" both return 401/not-found on the Hub and were removed.
HF_SOURCES = [(RS_INSTRUCTIONS_REPO, "train")]

# The eval harnesses' own default manifests: training data is always checked
# against these unless explicit --eval-manifest arguments replace them.
DEFAULT_EVAL_MANIFESTS = {
    "vrsbench": "data/vrsbench/sample/manifest.json",
    "rsvqa": "data/rsvqa_lr/manifest.json",
}

# BigEarthNet.txt splits that are evaluation-only. Train must never touch these patches.
BEN_EVAL_SPLITS = ("bench", "test", "validation")
BEN_TEXT_TYPES = ("captioning", "binary", "mcq")


def from_rs_instructions(n: int, token: str) -> List[dict]:
    from datasets import load_dataset

    repo_id, split = HF_SOURCES[0]
    print(f"[prepare] Streaming {repo_id} ({split}) ...")
    ds = load_dataset(repo_id, split=split, streaming=True, token=token or None)
    records = []
    for row in itertools.islice(ds, n):
        turns = row.get("conversations") or []
        for q, a in zip(turns[0::2], turns[1::2]):
            records.append({
                "image_path": None,
                "image_key": f"rsinst:{row['image']}",
                "question": str(q.get("value", "")).replace("<image>", "").strip()[:512],
                "answer": str(a.get("value", ""))[:512],
            })
    return records


def from_vrsbench_manifest(vrs_manifest: str, n: int) -> List[dict]:
    with open(vrs_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    records = []
    for rec in manifest:
        img_path = rec.get("image_path")
        if not img_path or not Path(img_path).exists():
            continue
        if rec.get("caption"):
            records.append({"image_path": img_path, "question": "Describe this satellite image.",
                            "answer": rec["caption"]})
        for qa in (rec.get("qa_pairs") or [])[:3]:
            q, a = qa.get("question") or qa.get("Q", ""), qa.get("answer") or qa.get("A", "")
            if q and a:
                records.append({"image_path": img_path, "question": str(q), "answer": str(a)})
        if len(records) >= n:
            break
    return records[:n]


def from_bigearthnet_txt(
    parquet: str, n: int, seed: int, out_dir: Path, n_patches: int = 0
) -> tuple[List[dict], str]:
    """Train rows only, plus an eval key manifest of every bench/test/validation patch.

    ``n_patches`` samples whole patches (all of a patch's text rows stay together), which is what
    bounds image download cost. Streams record batches so the 9.6M-row parquet is never fully
    materialised in RAM.
    """
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    rng = np.random.default_rng(seed)
    dataset = ds.dataset(parquet)
    eval_patches = set()
    for batch in dataset.to_batches(columns=["patch_id"], filter=ds.field("split").isin(BEN_EVAL_SPLITS)):
        eval_patches.update(pc.unique(batch["patch_id"]).to_pylist())
    eval_manifest = out_dir / "ben_eval_patch_keys.json"
    eval_manifest.write_text(
        json.dumps([{"image_key": f"ben:{p}"} for p in sorted(eval_patches)]), encoding="utf-8"
    )

    flt = (ds.field("split") == "train") & ds.field("type").isin(BEN_TEXT_TYPES)
    if n_patches:
        train_patches = set()
        for batch in dataset.to_batches(columns=["patch_id"], filter=flt):
            train_patches.update(pc.unique(batch["patch_id"]).to_pylist())
        if n_patches < len(train_patches):
            keep = rng.choice(sorted(train_patches), size=n_patches, replace=False).tolist()
            flt = flt & ds.field("patch_id").isin(keep)

    cols = ["ID", "patch_id", "s1_name", "input", "output", "type", "category"]
    records = []
    for batch in dataset.to_batches(columns=cols, filter=flt):
        for r in batch.to_pylist():
            records.append({
                "image_path": None,
                "image_key": f"ben:{r['patch_id']}",
                "patch_id": r["patch_id"],
                "s1_name": r["s1_name"],
                "question": r["input"],
                "answer": r["output"],
                "type": r["type"],
                "category": r["category"],
                "source_id": int(r["ID"]),
            })
    if n and n < len(records):
        keep_idx = np.sort(rng.choice(len(records), size=n, replace=False))
        records = [records[i] for i in keep_idx]
    return records, str(eval_manifest)


def _parse_eval_manifests(values: Optional[List[str]]) -> Dict[str, str]:
    if not values:
        return dict(DEFAULT_EVAL_MANIFESTS)
    out = {}
    for v in values:
        name, sep, path = v.partition("=")
        if not sep:
            raise SystemExit(f"--eval-manifest expects NAME=PATH, got {v!r}")
        out[name] = path
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare RS instruction data for LoRA")
    parser.add_argument("--source", choices=["bigearthnet_txt", "rs_instructions", "vrsbench"],
                        default="bigearthnet_txt")
    parser.add_argument("--out", default="data/ben_instructions")
    parser.add_argument("--ben-parquet", default="data/bigearthnet_txt/BigEarthNet.txt.parquet")
    parser.add_argument("--vrs-manifest", default="data/vrsbench/sample/manifest.json")
    parser.add_argument("--n", type=int, default=0, help="Cap on records (0 = all).")
    parser.add_argument("--n-patches", type=int, default=0,
                        help="bigearthnet_txt only: sample this many train patches, keeping all their rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-manifest", action="append",
                        help="NAME=PATH; repeatable. Replaces the harness defaults when given.")
    parser.add_argument("--allow-unchecked", action="append", default=[],
                        help="Eval manifest NAME that may be missing; recorded in the report.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_manifests = _parse_eval_manifests(args.eval_manifest)

    if args.source == "bigearthnet_txt":
        records, ben_eval = from_bigearthnet_txt(args.ben_parquet, args.n, args.seed, out_dir, args.n_patches)
        eval_manifests["bigearthnet_eval_splits"] = ben_eval
    elif args.source == "rs_instructions":
        records = from_rs_instructions(args.n or 300, args.hf_token)
    else:
        records = from_vrsbench_manifest(args.vrs_manifest, args.n or 300)

    if not records:
        raise SystemExit(f"[prepare] source {args.source!r} produced 0 records; nothing written.")

    report = check_overlap(records, eval_manifests)
    waived = [u for u in report.unchecked if u in args.allow_unchecked]
    verdict = report.to_dict() | {
        "source": args.source,
        "eval_manifests": eval_manifests,
        "unchecked_waived": waived,
        "passed": report.n_overlaps == 0 and set(report.unchecked) <= set(waived),
    }
    if not verdict["passed"]:
        (out_dir / "overlap_guard_FAILED.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        raise TrainEvalOverlapError(
            f"[prepare] refusing to write instructions.json: {report.n_overlaps} train/eval overlap(s), "
            f"unchecked eval manifests {sorted(set(report.unchecked) - set(waived)) or 'none'}. "
            f"See {out_dir / 'overlap_guard_FAILED.json'}"
        )

    (out_dir / "overlap_guard_report.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    manifest_path = out_dir / "instructions.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[prepare] guard passed; saved {len(records)} records -> {manifest_path}")


if __name__ == "__main__":
    main()
