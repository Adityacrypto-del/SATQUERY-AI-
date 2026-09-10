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

Notes:

- Entry 1 is marked **non-selection**: no trainable parameters, no checkpoint
  choice, therefore zero selection pressure on Test. Logged for completeness,
  not counted against the budget of 3, which remains fully intact.
- Test2 was also run once at baseline (47.53% AA / 45.60% OA) for the same
  reason. The end-of-project Test2 evaluation is still available.
