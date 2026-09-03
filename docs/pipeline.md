# Pipeline workflow

Full paper (description, flowchart, references, attributions):
[open-vocab-3dgs-imagine-pipeline-paper.md](open-vocab-3dgs-imagine-pipeline-paper.md) ·
[PDF](open-vocab-3dgs-imagine-pipeline-paper.pdf)

Default WebGPU scene: `splats/alarm_clock_generated.splat` (compact 32-byte SH0, shipped). Serve from the **repo root**.

```bash
./setup.sh                            # once per machine
python3 -m http.server 8090 --bind 127.0.0.1
./semantic_sidecar/launch.sh          # 127.0.0.1:8766, reads XAI_API_KEY
./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
```

```
splats/alarm_clock_generated.splat
        │  HTTP 8090  (or drop a trained point_cloud.ply)
        ▼
┌─────────────────────────────────────────────────────────┐
│  WebGPU viewer  (Kerbl EWA + SH0–3, GPU sort)           │
│  GaussForge decode → float gaussians                    │
└───────────────────────────┬─────────────────────────────┘
                            │  Tag scene (Grok)
                            │  capture 1–6 orbit rasters (PNG + camera)
                            ▼
┌─────────────────────────────────────────────────────────┐
│  semantic sidecar  :8766                                │
│  grok-4.6 vision  →  JSON boxes + names                 │
│  cluster names across views                             │
│  grok-imagine-image-2.0  →  edit crops → object cards   │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  overlay boxes      tint / list IDs     Imagine studio cards
  on the canvas      (right HUD)         (edits of 3DGS crops)

Later / optional (not in the sidecar today)
        │
        ▼
  ArtiFixer (local 80GB GPU, HF download nvidia/ArtiFixer)
  RGB + opacity + cameras → 1–4 step video repair → optional 3DGRUT bake
  Then feed repaired frames to Grok vision (never Imagine-as-renderer)
```

| Stage | What | Engine |
| --- | --- | --- |
| 1 Decode | PLY (3DGS / 2DGS) / SPLAT / SPZ / … | GaussForge (vendored, offline) → `shared/splat-io.js` fallback |
| 2 Rasterize | Paper 3DGS ellipses | This WebGPU viewer |
| 2b Identity (F1) | Label per Gaussian, instance table, ID / depth / normal readback, `pick()` | `gpu-renderer.js`, HUD *Instancias* |
| 2c Superpoints (F2) | kNN + Mahalanobis/colour weights → connected components; group → instance on click; label diffusion | `shared/graph.js` in `graph-worker.js`, HUD *Grupos* |
| 2d Lift (F3) | K-buffer α·T per mask label → FlashSplat argmax → association over superpoints → diffusion → `instancias.json` + `etiquetas.u32` | `contrib-pass.js`, `shared/lift.js`, HUD *Segmentación*; masks from the current labels (test), sidecar `/segment` (Grok boxes → ellipse masks, or a SAM backend via `SAM_BACKEND`), or SAM 2 in the browser (`ml-browser.js`, prompts = projected superpoint centroids) |
| 2e Name (F4) | Isolated render per instance → sidecar `/name` (Grok VQA, JSON `nombre`/`nombre_es`/`categoria`/`confianza`; `NAME_BACKEND=mock` for tests) → panel + search + Imagine card per `id_instancia`; optional CLIP ViT-B/32 embeddings per crop in the browser → semantic search + `embedding_clip` in the export | `shared/naming.js`, `ml-browser.js`, HUD *Instancias* |
| 2f Edit / export (F5) | Selection (rect / brush / sphere / superpoint) → `asignar`; per-instance `transformar` / `duplicar` / `borrar` / `fusionar` / `renombrar` in a replayable `ops.jsonl` with undo; bake transforms → PLY (+`instance_id`, `class_id`, `confidence`), `.splat`, SPZ / compressed PLY via GaussForge; reload of that PLY restores the instances | `shared/edit-ops.js`, `shared/export-io.js`, HUD *Edición*, sidecar `/exportaciones` → `artifacts/exportaciones/` |
| 3 Capture | PNG + yaw/pitch/eye | `canvas` snapshot |
| 4 Tag | Open-vocab names + boxes | `grok-4.6` vision |
| 5 Cluster | Merge armchair/sofa/seat | name key in sidecar |
| 6 Cards | Product stills | `grok-imagine-image-2.0` **edits** |
| 7 Repair (later) | Missing / blurry views | NVIDIA ArtiFixer |

Tests: `npm test` (unit) and `npm run test:e2e` (Chromium WebGPU, offscreen); see [testing.md](testing.md).

Do **not** replace step 2 with Imagine/GPT-Image/Gemini image generation. Those APIs have no camera or opacity, so masks cannot lift back to Gaussians.
