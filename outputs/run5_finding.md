# Run 5 — unchanged-pixel class supervision (INCOMPLETE, conclusion not established)

**Status: stopped at epoch 23 of 40. Not a completed experiment. Do not cite
it as evidence that the approach failed.**

## What it changed

Exactly one knob against Run 4: the class heads gained a seventh "unchanged"
logit, supervised on a derived fraction (0.0376) of unchanged pixels, where
Run 4 masked them out via `ignore_index`. Binary change weight stayed at 0.4.
`predict` reads only the six land-cover channels, so the change head still
owns the changed/unchanged decision — head disagreement stayed 0.000% for all
23 epochs.

The hypothesis it tested: Run 4's masking gave the class heads ~5x less
supervision and implicitly rebalanced the six classes, and that this cost the
common-class question types.

## Why it was stopped, and why that reason was wrong

It was stopped on the argument that closing the gap to Run 1 needed +2.5 AA
against a best-observed late gain of +1.02. That arithmetic used 66.10, a
figure the run had already left behind: it reached 67.46 two epochs later, so
the real gap was 1.13 — within the range both other runs achieved late. The
justification was stale when it was made.

## What the numbers actually show

Checkpoint provenance matters and the first comparison written up got it
wrong. Run 1's and Run 4's best checkpoints are at their peaks; Run 5's is
mid-climb.

| | Run 1 | Run 4 | Run 5 |
|---|---|---|---|
| best checkpoint | epoch 22 (peak) | epoch 32 (peak) | epoch 22 (**still rising**) |
| Average Accuracy | 68.59% | 68.00% | 67.46% |
| Overall Accuracy | 75.15% | 73.79% | 73.28% |
| mIoU | 42.08% | 43.67% | **43.97%** |
| head disagreement | 3.25% | 0.000% | 0.000% |

At **matched epoch 22**, which is the only fair comparison available:

| | Run 1 | Run 4 | Run 5 |
|---|---|---|---|
| AA at epoch 22 | 68.59 (peak) | 66.43 | 67.46 |

Run 5 is +1.03 ahead of Run 4 at the same epoch and already above Run 4's
entire last-10 plateau mean of 67.23. Run 4 went on to gain +1.57 from epoch
22 to its peak; an equivalent late gain would put Run 5 near 69, above Run 1.

That is not a prediction. It is the reason the experiment is unresolved.

The lead over Run 4 is also thin — Run 5 trailed Run 4 at epochs 15-20 and
led only at 21 and 22:

```
ep15 -1.16  ep16 -1.29  ep17 -1.74  ep18 -1.48
ep19 -0.44  ep20 -1.36  ep21 +1.41  ep22 +1.03
```

So the honest reading is unresolved in both directions, not quietly
favourable.

## Per-type, with the provenance caveat attached

Against the pre-committed decision rule (Run 5 must recover the four
common-class types `change_or_not`, `increase_or_not`, `decrease_or_not`,
`change_ratio_types`), comparing Run 5's mid-climb checkpoint against two
peak checkpoints:

| type | Run 1 (ep22 peak) | Run 4 (ep32 peak) | Run 5 (ep22 rising) |
|---|---|---|---|
| change_or_not * | 84.54% | 82.59% | 82.04% |
| increase_or_not * | 84.64% | 80.68% | 80.78% |
| decrease_or_not * | 84.18% | 83.04% | 82.47% |
| change_ratio_types * | 75.72% | 72.93% | 71.56% |
| change_to_what | 57.47% | 61.90% | **64.73%** |
| largest_change | 69.42% | 69.75% | 69.75% |
| smallest_change | 34.92% | 32.50% | 31.00% |
| change_ratio | 57.88% | 60.62% | 57.38% |

Run 5 is roughly level with Run 4 on the four starred types while ten epochs
younger, so on a per-epoch basis this is not evidence the mechanism
hypothesis is refuted. It is not evidence it is confirmed either.

`change_to_what` at 64.73% is the best any run has produced, and unanswered
dead ends fell to 92 (Run 1: 216, Run 4: 106).

## The finding that IS established

Run 5 has the best mIoU of all three runs and the worst CDVQA accuracy. With
the Test evaluation showing mIoU moving -2.10 Val->Test while AA moved -0.45,
that is three independent measurements that mIoU and CDVQA accuracy come
apart on this task.

Selecting checkpoints on mIoU — the standard segmentation practice — would
have chosen the worst-scoring model here, repeatedly. Selection runs on CDVQA
average accuracy for this reason, and the reason is measured rather than
assumed.

## If resumed

There is no resume path: `train()` always builds a fresh model, so completing
this means ~52 minutes from scratch. Command:

```
python -m segmentation.train --arch shared --change-weight-power 0.4 \
  --unchanged-class-supervision --epochs 40 --workers 2 \
  --out-dir outputs/segmentation/run5_unchanged_supervision
```

The decision rule stands: beat Run 1 on Val AA **and** recover the four
starred types, keeping 0.000% disagreement. Aggregate win without those four
is MIXED, which means stop rather than promote.
