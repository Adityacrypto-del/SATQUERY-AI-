"""Near-duplicate check between CDVQA's own splits.

The ID-based leakage assertions already in place guarantee that no pair ID
appears in two splits. That is necessary and not sufficient. SECOND's tiles
are cut from a handful of cities, so two 512x512 tiles from the same city
block are near-duplicates with different IDs, and training on one while
testing on the other is leakage no ID check can see. The PNGs carry no
georeferencing, so the comparison has to be visual.

Method is the one already used for the spare-scene screen: a 64-bit
difference hash per image, Hamming distance between every Train scene and
every held-out scene, minimum taken over both dates. No threshold is baked
in -- the full distribution is reported alongside a calibration distribution
built from pairs known to be distinct, so the cut is chosen from evidence.

This matters more than the spare-scene screen did: it bears directly on
whether a reported Test number reflects generalisation or memorisation.
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from eval.run_cdvqa import load_cdvqa_split
from scripts.check_scene_overlap import _describe, hamming_distances, load_hashes


def split_scenes(cdvqa_root: str, split: str):
    return sorted({s.pair_id for s in load_cdvqa_split(cdvqa_root, split)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdvqa-root", default="datasets/CDVQA")
    parser.add_argument("--second-root", default="datasets/SECOND_raw/train")
    parser.add_argument("--out", default="outputs/split_leakage.json")
    args = parser.parse_args(argv)

    train = split_scenes(args.cdvqa_root, "Train")
    print(f"Train scenes: {len(train)}", flush=True)
    train_hashes = load_hashes(args.second_root, train)

    report = {"n_train": len(train), "splits": {}}
    for split in ("Val", "Test", "Test2"):
        held = split_scenes(args.cdvqa_root, split)
        held_hashes = load_hashes(args.second_root, held)
        # hamming_distances already reduces to the per-scene minimum.
        nearest = hamming_distances(held_hashes, train_hashes)

        # Calibration: split the held-out set in half and compare one half
        # against the other. Those are distinct pairs by construction, so
        # this says what "a different scene" looks like on this data and the
        # cut is chosen from evidence rather than invented.
        half = len(held_hashes) // 2
        calib_nearest = hamming_distances(held_hashes[:half], held_hashes[half:])

        entry = {
            "n_scenes": len(held),
            "nearest_min": int(nearest.min()),
            "nearest_median": float(np.median(nearest)),
            "nearest_p5": float(np.percentile(nearest, 5)),
            "calibration_min": int(calib_nearest.min()),
            "calibration_median": float(np.median(calib_nearest)),
            "counts_at_or_below": {
                str(t): int((nearest <= t).sum()) for t in (0, 2, 4, 6, 8, 10, 12)
            },
            "fraction_at_or_below_12": float((nearest <= 12).mean()),
            "suspect_scenes": [
                held[i] for i in np.argsort(nearest)[:15] if nearest[i] <= 12
            ],
        }
        report["splits"][split] = entry
        print(f"\n{split}: {len(held)} scenes")
        print(_describe(f"  nearest Train distance", nearest))
        print(_describe(f"  calibration (held-out vs itself)", calib_nearest))
        print(f"  at or below 12: {entry['counts_at_or_below']['12']} "
              f"({entry['fraction_at_or_below_12'] * 100:.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
