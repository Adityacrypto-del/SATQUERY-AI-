# OpticalSAR Specialist — SatQuery AI

A self-supervised fusion model for Sentinel-2 (optical) + Sentinel-1 (SAR) imagery, producing a joint 512-dimensional embedding for downstream reasoning in the SatQuery AI pipeline.

---

## 1. What is SEN12MS-CR?

SEN12MS-CR (Sentinel-1/2 Multi-Season Cloud-Removal) is a large-scale remote sensing dataset containing geographically co-registered image triplets:

- **Sentinel-2 cloudy** — 13-band multispectral optical imagery, possibly partially cloud-covered
- **Sentinel-2 cloud-free** — the same geographic patch without clouds (ground truth for cloud removal)
- **Sentinel-1 SAR** — 2-channel (VV + VH) synthetic aperture radar backscatter

Images are 256×256 pixel patches at 10 m/pixel ground sampling distance, covering multiple seasons and biomes globally.

The dataset was originally released for cloud removal research. We use it for a different purpose.

---

## 2. Why SEN12MS-CR for optical-SAR representation learning?

The dataset provides something rare and valuable: **exactly co-registered optical and SAR patches of the same geographic location**.

- Every (S2, S1) pair sees the same ground truth.
- Co-registration is physically guaranteed (same sensor overpass area).
- SAR penetrates clouds, so the pair captures complementary physical phenomena.
- Global coverage means diverse land cover types.

This makes the co-registered pairs ideal **positive pairs** for contrastive learning. Two patches from different locations in the same batch become **negative pairs** automatically.

We do **not** use the cloud-free target image at all.

---

## 3. Training objective vs. cloud removal

| Aspect | Cloud removal (original) | This project |
|--------|--------------------------|--------------|
| **Goal** | Reconstruct cloud-free optical image | Learn joint optical+SAR embedding |
| **Input** | Cloudy optical + SAR | Optical + SAR |
| **Output** | Cloud-free image | 512-dim embedding vector |
| **Supervision** | Cloud-free image (pixel MSE) | Co-registration (contrastive) |
| **Loss** | L1 / perceptual loss | InfoNCE (CLIP-style) |
| **Use case** | Image enhancement | Feature extraction for reasoning |

---

## 4. Model Architecture

```
Sentinel-2 (13 channels)          Sentinel-1 (2 channels)
        │                                   │
  OpticalEncoder                       SAREncoder
  (ResNet18, conv1                 (ResNet18, conv1
  patched for 13ch)                patched for 2ch)
        │                                   │
  [B, 512]                            [B, 512]
        │                                   │
 OpticalProjectionHead            SARProjectionHead
  (2-layer MLP)                    (2-layer MLP)
        │                                   │
  [B, 256]                            [B, 256]
        ├───────────── InfoNCE Loss ────────┤
        │                                   │
        └───────────── Fusion ──────────────┘
                           │
                  ConcatMLPFusion (default)
               OR CrossAttentionFusion (optional)
                           │
                      [B, 512]
                   fused_embedding   ← Primary output for SatQuery AI
```

### Component roles

| Component | Role |
|-----------|------|
| `OpticalEncoder` | ResNet18 with 13-channel conv1; extracts spectral-spatial features from all 13 S2 bands |
| `SAREncoder` | ResNet18 with 2-channel conv1; extracts SAR texture features (VV, VH) |
| `ProjectionHead` | 2-layer MLP; maps encoder output to contrastive loss space |
| `ConcatMLPFusion` | Concatenate + MLP; fast, interpretable |
| `CrossAttentionFusion` | Bidirectional cross-attention; more expressive, optional |

---

## 5. Sentinel-2 13-band input processing

Raw Sentinel-2 values are integer DN (digital numbers) in the range `[0, 10000]`, representing top-of-atmosphere reflectance scaled by 10,000.

**Pipeline:**
1. Divide by 10,000 → approximate reflectance `[0, 1]`
2. Clip to `[0, 1.5]` (allow slight over-saturation in bright targets)
3. Replace NaN/Inf with 0
4. Z-score normalise per band: `(x - mean_b) / std_b`

**Critical:** All 13 bands are processed independently. The optical encoder genuinely processes all 13 channels via a 13-channel `Conv2d` as the first layer. We do not reduce to RGB.

---

## 6. Sentinel-1 2-channel input processing

SAR backscatter is stored in linear scale (not dB). Values are `float32` in approximately `[1e-5, 1.0]`.

**Pipeline:**
1. Replace NaN/Inf with `1e-6` (the physical noise floor)
2. Clip to `[1e-6, 1.0]`
3. Z-score normalise per channel (VV separately from VH)

The SAR encoder's first `Conv2d` has `in_channels=2` and is trained from scratch (no pretrained weights for this layer, since ImageNet has no SAR data).

---

## 7. How contrastive learning works

We use an **InfoNCE (CLIP-style) loss**:

Given a batch of B paired samples `{(optical_i, sar_i)}`:
- `optical_i` and `sar_i` are from the same geographic location → **positive pair**
- `optical_i` and `sar_j` (where `i ≠ j`) → **negative pairs** (in-batch negatives)

The loss:

```
L = -0.5 * (
    mean_i[ log(exp(sim(opt_i, sar_i)/τ) / Σ_j exp(sim(opt_i, sar_j)/τ)) ]   # opt→sar
  + mean_i[ log(exp(sim(sar_i, opt_i)/τ) / Σ_j exp(sim(sar_j, opt_i)/τ)) ]   # sar→opt
)
```

Where `sim` is cosine similarity and `τ` is the temperature (default 0.07).

This forces the model to:
1. Map co-registered optical and SAR patches to nearby points in embedding space
2. Map non-co-registered pairs far apart

An **auxiliary consistency loss** (weight 0.1) pushes the fused embedding close to both modality embeddings, preventing fusion collapse.

---

## 8. Fusion strategy

**Default: ConcatMLPFusion**
- Concatenate `[optical_emb | sar_emb]` → shape `[B, 1024]`
- Two-layer MLP with GELU + BatchNorm → shape `[B, 512]`
- Fast, stable, interpretable

**Optional: CrossAttentionFusion**
- Optical embedding queries SAR key/value → optical-attended
- SAR embedding queries optical key/value → SAR-attended
- Bidirectional: both modalities attend to each other
- Enable with `--cross_attn` flag or `cfg.use_cross_attention_fusion = True`

---

## 9. Quick start

### Installation

```bash
pip install -r requirements.txt
```

### Step 1: Inspect the dataset

Always run this first. It confirms field names, shapes, and value ranges without downloading the full dataset:

```bash
python inspect_dataset.py --save_viz sample.png
```

Expected output:
```
S2 field key:    's2' (or 'optical', etc.)
S1 field key:    's1' (or 'sar', etc.)
S2 shape:        (13, 256, 256)
S1 shape:        (2, 256, 256)
S2 raw range:    [0, 10000]
S1 raw range:    [0.0001, 0.98]
```

If field names differ from `s2`/`s1`, update `_S2_CANDIDATES` and `_SAR_CANDIDATES` in `dataset.py`.

---

## 10. Training

### Quick experiment (streaming, 2000 samples, 5 epochs)

```bash
python train.py --max_samples 2000 --epochs 5 --streaming --batch_size 32
```

### Full training

```bash
python train.py --epochs 50 --batch_size 64
```

### With cross-attention fusion

```bash
python train.py --cross_attn --epochs 50
```

### Disable AMP (for debugging or CPU)

```bash
python train.py --no_amp
```

---

## 11. Resuming training

```bash
python train.py --resume checkpoints/last.pt
```

The checkpoint contains model weights, optimizer state, scheduler state, epoch number, and validation metrics. Training resumes from the next epoch.

---

## 12. Evaluation

Evaluation runs automatically at the end of each training epoch. To run it standalone on a saved checkpoint:

```bash
python -c "
from config import ModelConfig
from dataset import build_dataloaders
from models import OpticalSARContrastiveModel
from evaluate import evaluate_retrieval
from utils import load_checkpoint, get_device
import torch

cfg = ModelConfig(max_train_samples=500)
_, val_loader = build_dataloaders(cfg)
device = get_device()
model = OpticalSARContrastiveModel(embedding_dim=512, projection_dim=256).to(device)
load_checkpoint('checkpoints/best.pt', model, device=device)
metrics = evaluate_retrieval(model, val_loader, device)
print(metrics)
"
```

**Metrics explained:**

| Metric | Meaning |
|--------|---------|
| `val_loss` | InfoNCE loss on validation set |
| `opt_to_sar/R@1` | For each optical patch, is its paired SAR the top-1 retrieval? |
| `opt_to_sar/R@5` | Is it in the top 5? |
| `sar_to_opt/R@1` | Reverse direction |

A random model gives R@1 ≈ 1/N (where N = batch size). After training, you should see R@1 well above random, indicating cross-modal alignment.

---

## 13. Inference

```python
from inference import OpticalSARSpecialist

# Load trained model
specialist = OpticalSARSpecialist.load_checkpoint("checkpoints/best.pt")

# Encode a pair
result = specialist.encode_pair(
    optical=optical_tensor,   # (13, H, W) or (B, 13, H, W)
    sar=sar_tensor,           # (2, H, W)  or (B, 2,  H, W)
)

# Primary output for SatQuery AI's reasoning model
fused_embedding = result["fused_embedding"]  # numpy (B, 512)

# Other outputs
optical_emb = result["optical_embedding"]         # (B, 512)
sar_emb     = result["sar_embedding"]             # (B, 512)
similarity  = result["cross_modal_similarity"]    # (B,) cosine sim in [-1, 1]
confidence  = result["confidence"]                # (B,) mapped to [0, 1]
```

### With raw (un-preprocessed) arrays

```python
import numpy as np
raw_s2 = np.load("patch_s2.npy")   # (13, 256, 256) raw DN values
raw_s1 = np.load("patch_s1.npy")   # (2, 256, 256)  linear backscatter

result = specialist.encode_pair(raw_s2, raw_s1, preprocess=True)
```

### JSON-serialisable output

```python
output = specialist.get_structured_output(
    optical=optical_tensor,
    sar=sar_tensor,
    include_embeddings=True,
)
# Returns dict that is json.dumps()-safe
```

---

## 14. What the fused embedding means

The `fused_embedding` is a 512-dimensional float32 vector that encodes:

1. **Cross-modal content**: spectral reflectance (optical) + backscatter texture (SAR) jointly
2. **Alignment**: nearby embeddings correspond to similar geographic/land-cover content
3. **Modality complementarity**: the fusion captures what SAR reveals that optical misses (e.g. surface roughness, moisture) and vice versa

**What it does NOT encode:**
- Geographic coordinates or metadata
- Time of acquisition
- Natural language semantics (that's the reasoning model's job)

The downstream reasoning model (Common Reasoning Model in SatQuery AI) takes this embedding and answers natural language queries about the scene.

---

## 15. GPU memory considerations

| Configuration | Approx GPU memory | Notes |
|---|---|---|
| ResNet18 + batch 64 + AMP | ~4–6 GB | Recommended default |
| ResNet18 + batch 32 + AMP | ~2–3 GB | For 4 GB GPUs |
| ResNet34 + batch 64 + AMP | ~6–8 GB | Larger encoder |
| ResNet18 + batch 64 + cross-attn | ~5–7 GB | More fusion compute |
| CPU (no GPU) | N/A | Slow but works; disable AMP |

Reduce `batch_size` if you hit OOM. Contrastive learning performs better with larger batches (more in-batch negatives), so keep it as large as your GPU allows.

---

## 16. Checkpoint structure

```
checkpoints/
    config.json           # ModelConfig used for training
    metrics.jsonl         # One JSON line per epoch
    best.pt               # Lowest validation loss
    last.pt               # Most recent epoch
    epoch_0005.pt         # Periodic saves (every 5 epochs by default)
```

Each `.pt` file contains:
```python
{
    "model_state":     ...,   # model.state_dict()
    "optimizer_state": ...,   # optimizer.state_dict()
    "scheduler_state": ...,   # scheduler.state_dict()
    "epoch":           42,    # last completed epoch
    "metrics":         {...}, # val_loss, R@1, R@5, R@10
    "config":          {...}, # full ModelConfig dict
}
```

---

## 17. Integration with SatQuery AI

```
┌──────────────────┐
│ Optical image    │  (13-channel Sentinel-2 patch)
│ SAR image        │  (2-channel Sentinel-1 patch)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  OpticalSARSpecialist    │  ← This module
│  (OpticalSAR encoder)    │
└────────┬─────────────────┘
         │  fused_embedding [512-dim]
         │  + cross_modal_similarity
         │  + confidence
         ▼
┌──────────────────────────┐
│  Common Reasoning Model  │  ← Teammate's module
│  (LLM / MLLM)           │
└────────┬─────────────────┘
         │  natural language answer
         ▼
┌──────────────────────────┐
│  SatQuery AI             │
│  (voice + GUI)           │
└──────────────────────────┘
```

The `fused_embedding` can be:
- Prepended to the reasoning model's context as a "vision token"
- Used in a cross-attention layer if the reasoning model supports it
- Converted to a text description using a vision-language adaptor

---

## 18. Project structure

```
optical_sar/
├── config.py           # All hyperparameters
├── dataset.py          # HuggingFace dataset wrapper
├── preprocessing.py    # S1/S2 preprocessing
├── losses.py           # InfoNCE + auxiliary loss
├── train.py            # Training loop
├── evaluate.py         # Retrieval metrics
├── inference.py        # Public API (OpticalSARSpecialist)
├── visualization.py    # Sample + embedding visualizations
├── utils.py            # Seed, device, checkpoint helpers
├── inspect_dataset.py  # Schema inspection (run before training)
├── models/
│   ├── optical_encoder.py   # ResNet18 patched for 13ch
│   ├── sar_encoder.py       # ResNet18 patched for 2ch
│   ├── fusion.py            # ConcatMLP + CrossAttention
│   └── contrastive_model.py # Full model
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_inference.py
├── requirements.txt
└── README.md
```

---

## 19. Running tests

```bash
# All tests (requires PyTorch)
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_model.py -v
python -m pytest tests/test_dataset.py -v
python -m pytest tests/test_inference.py -v
```

Tests use synthetic tensors — no dataset download required.

---

## 20. Future improvements

- **Larger encoders**: Replace ResNet18 with Vision Transformer (ViT-S) for richer representations. Requires more GPU memory.
- **Band-wise attention**: Add a channel attention module in the optical encoder to let the model learn which of the 13 bands matter most for a given scene.
- **Augmentation**: Add random flipping, rotation, and spectral jitter for both modalities to improve robustness.
- **Momentum encoder**: MoCo-style momentum encoder for larger effective batch sizes without OOM.
- **Temporal fusion**: Use SEN12MS-CR's multi-season samples to learn temporal invariant representations.
