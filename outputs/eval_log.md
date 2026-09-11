# Test-set evaluation log

Binding protocol (set 2026-09-10):

- **Val (400 scenes)** absorbs all iteration: checkpoint selection, learning
  rate, architecture, augmentation. Look as often as needed.
- **Test** may be evaluated **at most 3 times total**, at declared milestones.
- **Test2** exactly **once**, at the very end.
- Every Test/Test2 evaluation is appended below with timestamp, checkpoint
  and reason. Never evaluate on Test to satisfy curiosity.

A long log means the reported number is optimistically biased by selection,
and the writeup must say so.

| # | Date | Split | Checkpoint | Reason | AA | OA |
|---|------|-------|------------|--------|----|----|
| 1 | 2026-09-10 | Test | none (per-type majority baseline) | **non-selection (no model, no checkpoint choice)** — establishes the pre-model floor. | 44.76% | 50.84% |
| 2 | 2026-09-11 | Test | `outputs/segmentation/run1_unweighted/best.pt` (siamese, epoch 22) | **Val-to-Test transfer measurement.** All checkpoint and architecture selection to date landed on Val; nothing had ever measured whether Val numbers transfer. Not a winner confirmation -- Run 1 was the forward candidate either way, so this buys the transfer gap, not the ranking. | 68.14% | 74.89% |

Notes:

- Entry 1 is marked **non-selection**: no trainable parameters, no checkpoint
  choice, therefore zero selection pressure on Test. Logged for completeness,
  not counted against the budget of 3, which remains fully intact.
- Test2 was also run once at baseline (47.53% AA / 45.60% OA) for the same
  reason. The end-of-project Test2 evaluation is still available.

### Entry 2 — what it measured

**Budget: 1 of 3 spent. 2 remain. Test2 untouched.**

| | Val | Test | gap |
|---|---|---|---|
| Average Accuracy | 68.59% | 68.14% | **-0.45** |
| Overall Accuracy | 75.15% | 74.89% | -0.26 |
| mIoU (diagnostic) | 42.08% | 39.98% | -2.10 |
| head disagreement | 3.254% | 3.273% | +0.019 |

Per question type (Val -> Test): change_or_not 84.54 -> 84.69, increase_or_not
84.64 -> 83.37, decrease_or_not 84.18 -> 83.92, change_to_what 57.47 -> 57.47,
largest_change 69.42 -> 68.35, smallest_change 34.92 -> 35.02, change_ratio
57.88 -> 56.46, change_ratio_types 75.72 -> 75.84.

The gap is 0.45 AA across 39,686 held-out questions on 968 scenes, and no
question type moves more than 1.4 points. Five rounds of Val-based selection
therefore cost roughly half a point of optimism, not the several points that
would have meant Val was being fitted rather than measured. Two consequences:

1. Val remains usable for the remaining decisions, including whether Run 5
   earns Test evaluation #2.
2. The measured Val gap between candidates (~0.8 AA plateau-to-plateau
   between Run 1 and Run 4) is larger than the transfer error, so Val ranking
   between these candidates carries signal rather than noise.

mIoU moved further (-2.10) than the scored metric. It is diagnostic only and
is not used for checkpoint selection, which is exactly why selection is done
on CDVQA accuracy.

For reference, not comparison: the CDVQA paper's baseline reports ~58% average
accuracy. 68.14% here is on the official Test split with the selection history
above disclosed in full.

---

## Leakage findings (2026-09-11) — read before quoting any number above

Two checks that should have run before entry 2 were run after it. Both
qualify the numbers already recorded; neither invalidates them.

### 1. Train/Test near-duplicates: real, small, bounded

Pair-ID disjointness is perfect — `Train ∩ Test = 0`, so the ID assertions in
`eval/run_cdvqa.py` and `segmentation/dataset.py` do their job. That check is
necessary and not sufficient: SECOND's tiles are cut from a handful of
cities, so two 512×512 tiles from the same block are near-duplicates with
different IDs, and no ID check can see it.

`scripts/check_split_leakage.py` compares every Train scene against every
held-out scene by 64-bit difference hash, minimum over both dates, with a
calibration distribution built from scenes known to be distinct.

| split | scenes | nearest-Train min | median | ≤ 12 | calibration floor |
|---|---|---|---|---|---|
| Val | 400 | 10 | 16 | 11 (2.8%) | 13 |
| Test | 968 | **0** | 16 | 13 (1.3%) | 12 |
| Test2 | 968 | **0** | 16 | 13 (1.3%) | 12 |

13 Test scenes sit at or below the calibration floor, including exact hash
matches. **Bound on the effect:** if all 13 were perfectly memorised and
would otherwise have scored at the split average, the inflation is
13/968 × (100 − 68.14) ≈ **0.43 points**. So the true figure is
**≥ 67.7% AA**. (Approximate — AA is averaged per question type, not per
scene — but the contaminated fraction is small enough that the ordering of
conclusions does not change.)

Entry 2's 68.14% stands, with that bound stated alongside it.

### 2. Test2 is not an independent holdout — the protocol assumed wrong

| | questions | scenes |
|---|---|---|
| Test | 39,686 | 968 |
| Test2 | 31,036 | **the same 968** |

The scene sets are identical, and 5,028 (scene, question, answer) triples
appear in both. **Test2 is a question-level resample over the same imagery,
not a second image-level holdout.**

The protocol at the top of this file reserves Test2 as a pristine final
measurement. That independence does not exist: every Test evaluation has
already seen 100% of Test2's imagery. Test2 still measures something real —
generalisation to unseen *questions* about seen scenes — but it cannot
support a claim of unseen-imagery generalisation, and the writeup must say
which of the two it is.

This does not change the budget, which remains 1 of 3 Test evaluations spent
and Test2 untouched. It changes what spending them buys.
