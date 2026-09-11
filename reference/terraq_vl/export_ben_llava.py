"""Track D (prepared, NOT run): turn guarded BigEarthNet.txt instructions into TerraQ-VL's training layout.

Input: ``instructions.json`` + ``overlap_guard_report.json`` written by
``python -m satquery.adaptation.prepare_instructions --source bigearthnet_txt``. Refuses to run
unless that report says the guard passed.

Output (the layout ``build_vrsbench_trainset.py`` emits and TerraQ-VL's ``train.py`` reads):
    <out>/train.json   LLaVA single-turn records {id, image, conversations:[human, gpt]}
    <out>/val.json     same, from BigEarthNet.txt ``validation`` rows (sampled patches)
    <out>/images/<patch_id>.png   RGB render of S2 B04/B03/B02 via Branch 1's to_model_rgb

Task tags follow TerraQ-VL's VRSBench format: ``[caption]`` for captioning rows, ``[vqa]`` for
binary / mcq rows.

Images are read from a BigEarthNet v2.0 LMDB -- a local path (fast; download the LMDB or build it
with rico-hdl) or the HF mirror URL (HTTP range reads, ~0.5 MB and a few seconds per patch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "data_access"), str(HERE.parents[1])]

LMDB_MIRROR = "https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB/resolve/main/BENv2.lmdb/data.mdb"
TAG = {"captioning": "[caption]", "binary": "[vqa]", "mcq": "[vqa]"}


def llava(rec: dict, i: int) -> dict:
    return {
        "id": f"ben_{rec.get('source_id', i)}",
        "image": f"{rec['patch_id']}.png",
        "conversations": [
            {"from": "human", "value": f"<image>\n{TAG[rec['type']]} {rec['question']}"},
            {"from": "gpt", "value": str(rec["answer"])},
        ],
    }


def patch_reader(lmdb_source: str):
    from safetensors.numpy import load as st_load

    if lmdb_source.startswith("http"):
        import threading

        from httprange import HTTPRangeFile, RemoteLMDB

        local = threading.local()

        def get(pid: str) -> dict:
            if not hasattr(local, "db"):
                local.db = RemoteLMDB(HTTPRangeFile(lmdb_source))
            return st_load(local.db.get(pid.encode()))
    else:
        import lmdb

        env = lmdb.open(lmdb_source, readonly=True, lock=False, readahead=False)

        def get(pid: str) -> dict:
            with env.begin(buffers=True) as txn:
                return st_load(bytes(txn.get(pid.encode())))
    return get


def render(bands: dict, png: Path) -> None:
    import numpy as np
    from PIL import Image

    from satquery.preprocessing.multispectral import to_model_rgb

    arr = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1).astype("float32")
    rgb, _ = to_model_rgb(arr, {"band_descriptions": ["B04", "B03", "B02"]})
    Image.fromarray(np.rint(rgb * 255).astype("uint8")).save(png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions-dir", required=True, help="prepare_instructions --out dir")
    ap.add_argument("--ben-parquet", required=True)
    ap.add_argument("--out", default="datasets/ben_llava")
    ap.add_argument("--lmdb", default=LMDB_MIRROR, help="Local LMDB dir or the HF mirror URL.")
    ap.add_argument("--val-patches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    ins = Path(args.instructions_dir)
    report = json.loads((ins / "overlap_guard_report.json").read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("source") != "bigearthnet_txt":
        raise SystemExit("overlap guard report missing, failed, or not from bigearthnet_txt; refusing to export")
    train = json.loads((ins / "instructions.json").read_text(encoding="utf-8"))

    import numpy as np
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    dset = ds.dataset(args.ben_parquet)
    vflt = (ds.field("split") == "validation") & ds.field("type").isin(list(TAG))
    vpatches = set()
    for b in dset.to_batches(columns=["patch_id"], filter=vflt):
        vpatches.update(pc.unique(b["patch_id"]).to_pylist())
    keep = np.random.default_rng(args.seed).choice(sorted(vpatches), size=args.val_patches, replace=False).tolist()
    val = [
        {"source_id": int(r["ID"]), "patch_id": r["patch_id"], "question": r["input"],
         "answer": r["output"], "type": r["type"]}
        for b in dset.to_batches(columns=["ID", "patch_id", "input", "output", "type"],
                                 filter=vflt & ds.field("patch_id").isin(keep))
        for r in b.to_pylist()
    ]

    train_p, val_p = {r["patch_id"] for r in train}, {r["patch_id"] for r in val}
    assert not train_p & val_p, "train/val patch overlap"

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "train.json").write_text(json.dumps([llava(r, i) for i, r in enumerate(train)], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "val.json").write_text(json.dumps([llava(r, i) for i, r in enumerate(val)], ensure_ascii=False, indent=2), encoding="utf-8")

    get = patch_reader(args.lmdb)
    todo = [p for p in sorted(train_p | val_p) if not (out / "images" / f"{p}.png").exists()]

    def one(pid: str) -> None:
        render(get(pid), out / "images" / f"{pid}.png")

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 500 == 0:
                print(f"{i}/{len(todo)} patches rendered", flush=True)
    print(f"train {len(train)} recs / {len(train_p)} patches; val {len(val)} recs / {len(val_p)} patches -> {out}")


if __name__ == "__main__":
    main()
