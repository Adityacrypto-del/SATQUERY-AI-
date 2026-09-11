# SatQuery AI - System Report

*SIH 2026 | PS 26167 | ISRO / SAC*

An agentic vision-language assistant for remote-sensing imagery. A controller reads the query and the input configuration, selects a specialist model, and returns an evidence-grounded answer with a visual overlay and an auditable execution trace.

## 1. System architecture

`User query + image(s)  ->  Controller (task + input routing)  ->  Specialist model  ->  Evidence + trace  ->  Web UI`

| Layer | Technology | Status |
|---|---|---|
| Frontend | React 19, Vite 8, Tailwind 4 (4 pages) | Built |
| Backend API | FastAPI, single-image endpoint | Built |
| Branch 1: single image | Qwen2-VL-7B-Instruct + LoRA | Not adapted |
| Branch 2: bi-temporal | ResNet-18 U-Net + rule engine | Measured |
| Branch 3: DeltaVLM | Adapter interface only (stubs) | Not built |
| Optical-SAR fusion | Not started | Not built |

> Branches share one contract (ChangeEvidence) so the controller consumes a single object shape regardless of which specialist answered.

## 2. Branch 1 - single-image VQA and captioning

| Item | Specification |
|---|---|
| Base model | Qwen2-VL-7B-Instruct (vision-language) |
| Adaptation | LoRA / PEFT, rank 16, alpha 32, q/k/v/o projections |
| Quantisation | 4-bit (8 GB VRAM) or fp16 (16 GB) |
| Input | GeoTIFF, TIFF, PNG, JPEG via rasterio / PIL |
| Band handling | RGB by colour interpretation; Sentinel B04/B03/B02 by name; otherwise first three bands, recorded in the trace |
| Task routing | Keyword classifier -> captioning or VQA |
| Outputs | Answer or caption, model name, execution trace |
| Evaluation | RSVQA (exact match), VRSBench (BLEU, ROUGE-L, METEOR) |
| Measured accuracy | NOT YET MEASURED |

> The pipeline, LoRA configuration and evaluation scripts are complete, but no adapter has been trained and no benchmark has been run. The problem statement requires remote-sensing adaptation explicitly, so this is mandatory scope.

## 3. Branch 2 - bi-temporal change analysis

The published CDVQA baseline reaches about 58% because it never uses semantic labels. This branch predicts what each changed pixel was and became, then applies the benchmark's rules deterministically.

| Item | Specification |
|---|---|
| Architecture | U-Net, two decoders, ResNet-18 ImageNet encoder |
| Parameters | 21.6 M |
| Input / output | 6 channels (2 x RGB) at 512x512 -> two 7-class maps |
| Training | 40 epochs, AMP fp16, RTX 4050 6 GB, best epoch 22 |
| Answer engine | Deterministic rules -> one of CDVQA's 19 answer tokens |
| Inference | 4 s model load, then about 200 ms per query |
| Interface | One call in, evidence object out; registry descriptor included |
| Tests | 312 passing |

### Measured results

| Metric | Value | Note |
|---|---|---|
| CDVQA Test average accuracy | 68.14% | 968 scenes, 39,686 questions |
| CDVQA Test overall accuracy | 74.89% | official held-out split |
| Published CDVQA baseline | ~58% | we are about 10 points above |
| Majority-class floor | 44.91% | what guessing alone scores |
| Rule ceiling (oracle) | 99.90% | rules on ground-truth maps |
| Segmentation mIoU | 42.08% | diagnostic, not the scored metric |

> The oracle figure proves the rules reproduce the benchmark's own answers almost exactly, so every remaining error is segmentation quality rather than reasoning. Near-duplicate screening bounds the true test figure at 67.7% or above.

## 4. Datasets

| Dataset | Role | Used by | Scale |
|---|---|---|---|
| SECOND | Semantic change maps, training labels | Branch 2 | 4,662 pairs |
| CDVQA | Change VQA benchmark, 19 answers, 8 types | Branch 2 | 2,968 scenes |
| OSCD | Real Sentinel-2 13-band pairs, GeoTIFF path | Branch 2 | 24 pairs |
| BigEarthNet-S2 | Domain adaptation corpus for LoRA | Branch 1 | subset |
| RSVQA | Single-image VQA evaluation | Branch 1 | not yet run |
| VRSBench | Captioning evaluation | Branch 1 | not yet run |

## 5. Problem statement coverage

| Mandatory requirement | Status |
|---|---|
| Single-image VQA | Built, not measured |
| Second single-image task (captioning) | Built, not measured |
| Bi-temporal change analysis | Built and measured |
| Spatial change map | Built |
| Optical-SAR paired analysis | NOT BUILT |
| Remote-sensing adaptation (fine-tuning) | NOT DONE |
| Agentic tool orchestration | Partial |
| Confidence estimation | Built (branch 2, measured) |
| Visual evidence | Built (branch 2) |
| Execution trace | Built (both branches) |
| Downloadable report | Built (branch 2) |
| Interactive web application | Built |

## 6. What distinguishes this build

- Confidence is measured, never invented. Every number is an accuracy actually observed on validation for that question type and land-cover class. Where no measurement applies it reports 0.0 and states why.
- Out-of-distribution input is detected. A physics-based producer cross-checks the trained model; when they disagree the system withdraws its confidence rather than answering confidently.
- Refusals over fabrication. SAR carries no land-cover mapping, so class questions are refused. Ungeoreferenced input reports pixel counts, never an invented area.
- Full execution trace: every stage records the tool, parameters, observation, duration and why the step was taken.
- Visual evidence is built from the same pixels the rule counted, so the picture cannot contradict the text.

## 7. Known gaps

- Remote-sensing fine-tuning on branch 1 has not been run. Mandatory.
- Optical-SAR paired analysis is not started. Mandatory.
- Branch 1 has no measured accuracy on any benchmark.
- Branch 2 under-predicts rare classes: water at 0.21x true frequency, playgrounds never predicted. Declared in the tool descriptor and reflected in its confidence.
- Branch 2 does not transfer across sensors yet: trained on sub-metre aerial imagery, it detects nothing at Sentinel-2's 10 m and correctly reports zero confidence.
- The controller registry spanning both branches is not yet wired.

---

All figures are measured on the stated splits. Nothing in this report is estimated or projected.