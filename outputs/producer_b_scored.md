# Producer B, scored against reference labels

Until now Producer B reported `confidence = 0.0` on the grounds that it had
never been scored. OSCD ships reference change masks, so that is no longer
true for its change mask. Its **class assignment** remains unscored, because
OSCD has no land-cover labels -- only binary change.

## Method

10 real Sentinel-2 pairs (OSCD test split), 13 bands, written through the
same GeoTIFF path the pipeline uses. Producer B's `change_mask` compared
against the reference mask per scene.

## Result

| scene | truth changed | predicted | IoU | precision | recall |
|---|---|---|---|---|---|
| 0 | 5.69% | 34.10% | 0.092 | 0.098 | 0.590 |
| 1 | 1.14% | 65.77% | 0.009 | 0.009 | 0.541 |
| 2 | 0.44% | 49.85% | 0.004 | 0.004 | 0.487 |
| 3 | 7.21% | 25.56% | 0.135 | 0.152 | 0.540 |
| 4 | 6.79% | 42.24% | 0.094 | 0.099 | 0.617 |
| 5 | 2.58% | 29.03% | 0.044 | 0.046 | 0.521 |
| 6 | 1.32% | 51.44% | 0.016 | 0.016 | 0.613 |
| 7 | 9.92% | 30.08% | 0.147 | 0.171 | 0.517 |
| 8 | 0.80% | 24.12% | 0.016 | 0.016 | 0.486 |
| 9 | 7.67% | 20.17% | 0.200 | 0.230 | 0.605 |
| **mean** | | | **0.076** | **0.084** | **0.552** |

## Reading it

Recall 0.552 and precision 0.084. It finds about **half** the real change and
**over-detects by roughly six times** -- ground truth runs 0.44% to 9.92% of
a scene while it predicts 20% to 66%.

The cause is structural, not a tuning error. `classify_pair` calls a pixel
changed when its independently assigned class differs between dates, so any
index fluctuation across a class boundary flips it. Per-date classification
is noisy, and differencing two noisy classifications compounds the noise.

## What this changes

**Producer B is a cross-check, not an answer path.** An earlier note in this
repo said that for Sentinel-2-like input "the deterministic index path is the
usable one". That was wrong and is corrected here: at precision 0.084 it is
not usable as a primary answer.

What it *is* good for is exactly what it now does:

*   telling us the trained model is out of its depth -- it correctly flagged
    a flat contradiction on the Sentinel-2 pair, which now zeroes the
    reported confidence;
*   refusing SAR class questions rather than answering them;
*   supplying class maps where the trained model has no coverage, with the
    confidence honestly at 0.0.

Its `confidence` stays 0.0. These numbers are too poor to report as a
calibrated accuracy, and reporting 0.076 as a confidence would be worse than
reporting none.

## Still open

*   Class assignment is unscored and needs semantic reference labels.
*   `playgrounds` is never predicted, by design -- no spectral signature.
*   Trees vs low vegetation, and buildings vs ground, are relative rankings
    within a scene rather than absolute determinations.
*   The over-detection is fixable in principle: requiring a minimum index
    *magnitude* change, not merely a class flip, would trade recall for
    precision. Not attempted here; it is a Stage B change and needs its own
    measurement.
