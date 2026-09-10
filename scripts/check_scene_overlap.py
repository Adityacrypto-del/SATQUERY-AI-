"""Near-duplicate check between spare SECOND scenes and CDVQA test scenes.

CDVQA uses 2,968 of SECOND's 4,662 pairs, all drawn from SECOND's ``train/``
folder. The remaining 1,694 in SECOND's ``test/`` folder carry no CDVQA
questions, so they are useful only as extra segmentation training data.

The ID-based leakage check passes trivially for them -- the filenames are
disjoint. But SECOND's tiles come from a handful of cities, and two 512x512
tiles cut from the same city block are near-duplicates. Training on one and
testing on the other is leakage that no ID check can catch, and the PNGs
carry no georeferencing to check spatially. So this compares them visually.

Method: a 64-bit difference hash per image, Hamming distance between every
spare scene and every CDVQA test scene, minimum taken over both dates. No
threshold is baked in. The script reports the full distance distribution
alongside a *calibration* distribution built from pairs known to be
distinct, so the cut can be chosen from evidence rather than from a
number someone liked the look of.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["dhash", "hamming_distances", "load_hashes"]


def dhash(path: str, size: int = 8) -> np.uint64:
    """64-bit difference hash of an image.

    Compares each pixel with its right-hand neighbour in a downscaled
    greyscale image, so the hash encodes coarse structure and is insensitive
    to brightness and small radiometric differences -- which is what we want,
    since two crops of the same block may differ in exposure.
    """
    from PIL import Image

    with Image.open(path) as handle:
        small = handle.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
    value = np.uint64(0)
    for index, bit in enumerate(bits):
        if bit:
            value |= np.uint64(1) << np.uint64(index)
    return value


def load_hashes(root: str, scenes: Sequence[str], folders=("im1", "im2")) -> np.ndarray:
    """Hash each scene's images. Returns ``(n_scenes, n_folders)`` uint64."""
    out = np.zeros((len(scenes), len(folders)), dtype=np.uint64)
    for row, scene in enumerate(scenes):
        for column, folder in enumerate(folders):
            out[row, column] = dhash(os.path.join(root, folder, scene))
    return out


def _popcount(values: np.ndarray) -> np.ndarray:
    """Bit count of each uint64, via byte-wise table lookup."""
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    view = values.view(np.uint8).reshape(values.shape + (8,))
    return table[view].sum(axis=-1)


def hamming_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum Hamming distance between each row of ``a`` and each of ``b``.

    Both are ``(n, k)`` hashes with ``k`` dates. The distance between two
    scenes is the minimum over dates: matching on either date is enough to
    make them near-duplicates for training purposes.
    """
    best = np.full(a.shape[0], 64, dtype=np.int16)
    for column in range(a.shape[1]):
        for other_column in range(b.shape[1]):
            xor = a[:, column, None] ^ b[None, :, other_column]
            distances = _popcount(xor).min(axis=1)
            best = np.minimum(best, distances.astype(np.int16))
    return best


def _describe(name: str, values: np.ndarray) -> str:
    percentiles = [0, 1, 5, 25, 50]
    parts = [f"{name}: n={values.size}"]
    parts.append("  min=%d" % values.min())
    for p in percentiles[1:]:
        parts.append(f"  p{p}={np.percentile(values, p):.0f}")
    parts.append(f"  median={np.median(values):.0f}")
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdvqa-root", default="datasets/CDVQA")
    parser.add_argument("--second-train", default="datasets/SECOND_raw/train")
    parser.add_argument("--spare-root", default="datasets/SECOND_raw/test_extract/test")
    parser.add_argument("--out", default="outputs/scene_overlap.json")
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Hamming distance at or below which a spare scene is dropped. "
             "Omit to report the distribution without committing to a cut.",
    )
    args = parser.parse_args(argv)

    from segmentation.dataset import cdvqa_split_scenes

    test_scenes = sorted(
        set(cdvqa_split_scenes(args.cdvqa_root, "Test"))
        | set(cdvqa_split_scenes(args.cdvqa_root, "Test2"))
    )
    val_scenes = cdvqa_split_scenes(args.cdvqa_root, "Val")
    held_out = sorted(set(test_scenes) | set(val_scenes))
    spare_scenes = sorted(os.listdir(os.path.join(args.spare_root, "im1")))

    print(f"hashing {len(spare_scenes)} spare scenes ...", flush=True)
    spare = load_hashes(args.spare_root, spare_scenes)
    print(f"hashing {len(held_out)} held-out CDVQA scenes ...", flush=True)
    held = load_hashes(args.second_train, held_out)

    distances = hamming_distances(spare, held)

    # Calibration: distances among held-out scenes themselves, excluding the
    # self-match. These pairs are known-distinct scenes, so their distance
    # distribution is what "not a duplicate" looks like for this data.
    print("calibrating against known-distinct pairs ...", flush=True)
    calibration = []
    step = max(len(held_out) // 400, 1)
    for row in range(0, len(held_out), step):
        xor = held[row, 0] ^ held[:, 0]
        counts = _popcount(np.asarray(xor, dtype=np.uint64))
        counts[row] = 64  # drop the self-match
        calibration.append(counts.min())
    calibration = np.array(calibration, dtype=np.int16)

    print()
    print(_describe("spare-vs-heldout ", distances))
    print(_describe("heldout-vs-heldout (known distinct)", calibration))
    print()

    header = f"{'threshold':>10}  {'spare dropped':>14}  {'calibration flagged':>20}"
    print(header)
    print("-" * len(header))
    for threshold in (0, 2, 4, 6, 8, 10, 12):
        dropped = int((distances <= threshold).sum())
        false_positive = int((calibration <= threshold).sum())
        print(f"{threshold:>10}  {dropped:>14}  {false_positive:>20}")

    payload: Dict[str, object] = {
        "n_spare": len(spare_scenes),
        "n_held_out": len(held_out),
        "distance_min": int(distances.min()),
        "distance_median": float(np.median(distances)),
        "calibration_min": int(calibration.min()),
        "calibration_median": float(np.median(calibration)),
        "counts_by_threshold": {
            str(t): int((distances <= t).sum()) for t in (0, 2, 4, 6, 8, 10, 12)
        },
        "calibration_by_threshold": {
            str(t): int((calibration <= t).sum()) for t in (0, 2, 4, 6, 8, 10, 12)
        },
        # Persisted so a threshold can be applied later without re-hashing
        # 6,000-odd images. Only the non-trivial tail is kept: anything above
        # the calibration floor is uncontroversially distinct.
        "distances": {
            scene: int(distance)
            for scene, distance in zip(spare_scenes, distances)
            if distance <= 20
        },
    }

    if args.threshold is not None:
        safe = [s for s, d in zip(spare_scenes, distances) if d > args.threshold]
        payload["threshold"] = args.threshold
        payload["safe_scenes"] = safe
        payload["dropped_scenes"] = [
            s for s, d in zip(spare_scenes, distances) if d <= args.threshold
        ]
        print()
        print(
            f"threshold {args.threshold}: {len(safe)} of {len(spare_scenes)} "
            f"spare scenes are safe to train on "
            f"({len(spare_scenes) - len(safe)} dropped)"
        )
    else:
        print()
        print("No --threshold given, so no safe list was written. Choose a cut "
              "from the table above, where calibration flagged should stay at "
              "or near zero.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
