# Pipeline workflow

Full paper (description, flowchart, references, attributions):
[open-vocab-3dgs-imagine-pipeline-paper.md](open-vocab-3dgs-imagine-pipeline-paper.md) ·
[PDF](open-vocab-3dgs-imagine-pipeline-paper.pdf)

Default WebGPU scene: `splats/model.splat` (compact 32-byte SH0). Serve from the **repo root**.

```bash
python3 -m http.server 8090 --bind 127.0.0.1
./semantic_sidecar/launch.sh          # 127.0.0.1:8766, reads XAI_API_KEY
./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
```

```
splats/model.splat
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
| 1 Decode | PLY / SPLAT / SPZ / … | GaussForge |
| 2 Rasterize | Paper 3DGS ellipses | This WebGPU viewer |
| 3 Capture | PNG + yaw/pitch/eye | `canvas` snapshot |
| 4 Tag | Open-vocab names + boxes | `grok-4.6` vision |
| 5 Cluster | Merge armchair/sofa/seat | name key in sidecar |
| 6 Cards | Product stills | `grok-imagine-image-2.0` **edits** |
| 7 Repair (later) | Missing / blurry views | NVIDIA ArtiFixer |

Do **not** replace step 2 with Imagine/GPT-Image/Gemini image generation. Those APIs have no camera or opacity, so masks cannot lift back to Gaussians.
