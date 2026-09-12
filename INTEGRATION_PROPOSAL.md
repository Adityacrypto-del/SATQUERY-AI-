# SatQuery end-to-end wiring proposal

Written from Branch 1 (`peek/branch1-baseline`) on 2026-09-12 after inspecting every branch.
This is a **proposal**, not applied work: it touches other people's branches, so it needs their
agreement. Nothing outside Branch 1 has been modified.

## 1. What exists today (verified, not assumed)

| | Branch 1 `ayush` / `ayushFRONT` | Branch 2 `peek/bitemporal-deterministic` | Branch 3 `model3` | Controller `aditya/bi-temporal-deltavlm` |
|---|---|---|---|---|
| HTTP | `app/backend_server.py` → `/api/status`, `/api/presets`, `/api/analyze` | `app/bitemporal_server.py` → `/api/bitemporal/{health,describe,analyze,evidence/*}` | `api_server.py` → `/api/health`, `/api/optical-sar` | none |
| Entry object | `SingleImageAdapter` (this branch, today) | `BiTemporalSpecialist` (`tools/change_analysis/api.py`) | `OpticalSARBranchService` | `run_satquery()` in `modules/pipeline.py` |
| Result type | `SingleImageEvidence` | `ChangeResult` (answer, trace, changed, regions, overlays, georeferencing) | `{"success", "data"}` | `FinalResponse` via `FusedEvidence` |
| Frontend | **owns it** (React/Vite, proxy `/api` → `:8000`) | none | none | none |

`ayushFRONT` is byte-identical to `ayush`. **No branch imports or calls any other branch.** The
frontend only ever submits one image.

## 2. Four blockers

1. **Three servers, one port.** All default to `:8000`, and the Vite proxy targets one origin.
2. **The UI is single-image only.** No t1/t2 pair, no optical+SAR pair, no overlay display.
3. **Three response shapes.** `ok`/`error`/`overlays`/`confidence_basis` (B2) vs a flat dict (B1)
   vs `{"success", "data"}` (B3). The UI would need per-branch special cases.
4. **Two competing orchestrators**: `run_satquery()` versus the per-branch servers. Only one can be
   the integration point.

## 3. Proposed architecture

```
                         React (Vite :5173)  ──proxy /api──►  ONE FastAPI app (:8000)
                                                                  │
   POST /api/query  (the controller: modality routing + fusion)    │
        │                                                          ├── router: /api/*            (Branch 1)
        ├── 1 image          → SingleImageAdapter                  ├── router: /api/bitemporal/* (Branch 2)
        ├── 2 images (t1,t2) → BiTemporalSpecialist                └── router: /api/optical-sar  (Branch 3)
        ├── optical + SAR    → OpticalSARBranchService
        │
        ├── optional generalist pass (Qwen2.5-VL-3B, Aditya's modules/qwen)
        └── fusion: specialist measurements are authoritative → one envelope
```

Each branch keeps its own server for standalone use. The change per branch is mechanical:

```python
# app/<branch>_server.py
router = APIRouter()          # was: app = FastAPI(...)
@router.post("/api/bitemporal/analyze")   # decorators move from app to router
...
app = FastAPI(); app.include_router(router)   # standalone still works
```

The backbone then does `app.include_router(branch1.router)` and so on. No branch logic moves.

### 3.1 Specialist protocol (the contract to agree on)

One shared file, one owner. Every branch already nearly satisfies it:

```python
class Specialist(Protocol):
    name: str
    modality: str          # "single_image" | "bi_temporal" | "optical_sar"
    def is_available(self) -> bool: ...
    def describe(self) -> dict: ...
    def analyze(self, **inputs) -> Evidence: ...   # Evidence has .to_dict()
```

- Branch 1: `SingleImageAdapter` satisfies it already (`is_available`, `load_model`, `describe`, `analyze`).
- Branch 2: `BiTemporalSpecialist` + `describe_tool` exist; add `is_available` / `modality`.
- Branch 3: wrap `OpticalSARBranchService` (needs `is_available`, since it loads a model at import).

### 3.2 Response envelope

Every endpoint and the controller return the same keys. This is the union of what the branches
already produce, so it is a mapping exercise, not a redesign:

```json
{
  "ok": true, "status": "ok|refused|unavailable|error",
  "modality": "single_image|bi_temporal|optical_sar", "task": "vqa|captioning|spectral_index|change|...",
  "answer": "…or null", "model": "which checkpoint produced this", "is_vlm_answer": true,
  "confidence": 0.55, "confidence_basis": "heuristic_default | none: analysis did not complete",
  "measurements": {}, "regions": [], "overlays": {"mask": "/api/bitemporal/evidence/…"},
  "source_metadata": {}, "trace": [], "limitations": [], "reason": null
}
```

Rules the envelope enforces, so honesty is structural rather than a habit:
- `status != "ok"` ⇒ `answer` is null and `reason` is set. Branch 1's `SingleImageEvidence`
  already raises if that is violated.
- `measurements` holds only deterministic, physically derived numbers (spectral indices, areas from
  a geotransform). A VLM's estimate never goes there; fusion treats it as authoritative.
- `confidence_basis` must name the method. "Heuristic" is acceptable; an uncalibrated number
  presented as a probability is not.

### 3.3 Fusion

`FusedEvidence.specialist_evidence` is typed `ChangeEvidence`, so it only accepts bi-temporal
evidence. Widen it to the `Evidence` protocol (or a union) and populate
`authoritative_measurements` from `measurements`. That is the one change needed in the
controller-owned files, and it is Aditya's call.

### 3.4 Frontend (Branch 1 owns this)

1. Mode tabs in `ChatbotPage`: **Single** / **Compare (t1, t2)** / **Optical + SAR**, with a second
   uploader for the pair modes.
2. Post to `/api/query` and render the envelope: answer, `confidence_basis`, `measurements` table,
   `limitations`, and `trace`.
3. Render `overlays` as images (Branch 2 already returns URLs). This is the most visible demo win:
   a change mask beside the answer.
4. **Delete the client-side fabricated fallback** in `ChatbotPage.jsx` (invented "95.4% confidence"
   and land-use numbers when a request fails) and show the error instead. Left untouched so far on
   your instruction; Branch 1's server-side fabrication is already removed.

## 4. Merge order and layout collisions

1. **Branch 2 → Branch 1.** Clean: `tools/`, `modules/bi_temporal/`, `eval/`, `segmentation/` do not
   exist on Branch 1. This also resolves Branch 1's current dependency on `indices.py` living on
   Branch 2 (today it is found through `SATQUERY_BITEMPORAL_ROOT`).
2. **Controller files → same tree.** `modules/{qwen,fusion,pipeline}.py` from Aditya. Watch
   `modules/bi_temporal/adapter.py`, which exists on both his and Branch 2's branch.
3. **Branch 3 → last, and it needs restructuring.** It is a flat repo (`api_server.py`, `train.py`,
   `config.py`, `dataset.py` at the root) and would collide with common names. It should move under
   `modules/optical_sar/` as part of the merge.

## 5. Open decisions (not mine to make)

1. **Who owns the backbone app and the envelope schema?** One owner, or it will drift.
2. **Is the controller `run_satquery` or the backbone?** Recommendation: keep the orchestration logic
   from `run_satquery`, but expose it as `POST /api/query` in the backbone, so there is one path to
   the frontend and no second CLI-only orchestrator.
3. **Generalist VLM cost.** A Qwen2.5-VL-3B generalist pass plus a specialist will not co-reside on a
   6 GB laptop GPU (Branch 1's specialist alone measures 3.35 GiB). Either the generalist runs on the
   demo machine only, or it is lazily loaded and reports `"generalist": "unavailable"` honestly.
4. **Preset demos.** Branch 1's `/api/analyze-preset` now returns 501 instead of canned text, because
   preset images are remote URLs. If presets matter for the demo, ship 2-3 real rasters in the repo
   and analyse them for real.

## 6. Suggested first slice (smallest end-to-end path)

1. Convert the three servers to routers (mechanical, per branch).
2. Backbone app mounting all three + `/api/status` aggregating each `describe()`.
3. Envelope mappers for Branch 2 and Branch 3 responses.
4. `/api/query` with modality routing only (no generalist, no fusion yet).
5. Frontend mode tabs + overlay rendering.
6. An end-to-end smoke test: start the backbone, submit one single-image, one pair, one optical+SAR
   request, assert each returns a valid envelope — including a clean `unavailable` when weights are
   absent, which is what CI without a GPU will see.

Effort is unestimated here on purpose; steps 1-3 are mechanical, step 4-5 are the real work.
