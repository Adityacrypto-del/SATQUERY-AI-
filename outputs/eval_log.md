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
