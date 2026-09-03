# Pipeline workflow

Full report (description, flowcharts, references, attributions and licences):
[open-vocab-3dgs-imagine-pipeline-paper.md](open-vocab-3dgs-imagine-pipeline-paper.md) ·
[PDF](open-vocab-3dgs-imagine-pipeline-paper.pdf). Roadmap and per-phase status (Spanish):
[plan-segmentacion-edicion-3dgs.md](plan-segmentacion-edicion-3dgs.md).

Default WebGPU scene: `splats/alarm_clock_generated.splat` (compact 32-byte SH0, shipped). Serve from the **repo root**.

```bash
./setup.sh                            # once per machine (dirs, .env, dependency checks)
python3 -m http.server 8090 --bind 127.0.0.1
./semantic_sidecar/launch.sh          # 127.0.0.1:8766, reads XAI_API_KEY (optional: Grok naming, Imagine cards, saving under artifacts/)
scripts/download-ml-models.sh         # optional, once: SAM 2.1 + CLIP weights into vendor/ml/ (~240 MB, gitignored)
./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
```

## Flow

![Segmentation, editing and meshing pipeline (plan F0–F6)](figures/segmentation-pipeline.png)

*Figure. Everything inside the browser runs in WebGPU or Web Workers; the sidecar only calls xAI and persists files under `artifacts/`. Source: `figures/segmentation-pipeline.mmd` (render with `node scripts/render-mermaid.mjs`).*

| Stage | What | Engine |
| --- | --- | --- |
| 1 Decode (F0) | PLY (3DGS / 2DGS) / SPLAT / SPZ / KSPLAT / SOG → float gaussians (12 floats) + SH (48 floats) | GaussForge WASM (vendored in `vendor/gaussforge/`, offline) → `shared/splat-io.js` fallback; a PLY with `instance_id` brings its instances back |
| 2 Rasterize | Paper 3DGS ellipses (EWA, SH0–3, GPU counting sort) | `gaussian_splatting_webgpu/gpu-renderer.js` |
| 2b Identity (F1) | Label per Gaussian, instance table (transform, tint, flags), colour / depth / normal / ID readback, `pick()`, `pickRect()`, `pickDisc()` | same, HUD *Instancias* |
| 2c Superpoints (F2) | kNN + Mahalanobis/colour weights → connected components; group → instance on click; label diffusion | `shared/graph.js` in `graph-worker.js`, HUD *Grupos* |
| 2d Lift (F3) | K-buffer α·T per mask label → FlashSplat argmax → association over superpoints → diffusion → `instancias.json` + `etiquetas.u32` | `contrib-pass.js`, `shared/lift.js`, HUD *Segmentación*; masks from the current labels (test), sidecar `/segment` (Grok boxes → ellipse masks, or a SAM backend via `SAM_BACKEND`), or SAM 2.1 in the browser (`ml-browser.js`, prompts = projected superpoint centroids, duplicate masks merged) |
| 2e Name (F4) | Isolated render per instance → sidecar `/name` (Grok VQA, JSON `nombre`/`nombre_es`/`categoria`/`confianza`; `NAME_BACKEND=mock` for tests) → panel + text search + Imagine card per `id_instancia`; optional CLIP ViT-B/32 embeddings per crop in the browser → semantic search + `embedding_clip` in the export | `shared/naming.js`, `ml-browser.js`, HUD *Instancias* |
| 2f Edit / export (F5) | Selection (rect / brush / sphere / superpoint) → `asignar`; per-instance `transformar` / `duplicar` / `borrar` / `fusionar` / `renombrar` in a replayable `ops.jsonl` with undo; bake transforms → PLY (+`instance_id`, `class_id`, `confidence`), `.splat`, SPZ / compressed PLY via GaussForge | `shared/edit-ops.js`, `shared/export-io.js`, HUD *Edición*, sidecar `/exportaciones` |
| 2g Mesh (F6) | Isolated instance → orbit cameras → depth (mean or 2DGS median) + colour → TSDF fusion with carving → surface nets → largest component → GLB (normals + vertex colours) | `shared/tsdf.js` in `tsdf-worker.js`, `shared/glb.js`, HUD *Malla*, sidecar `/mallas` |
| 3 Capture | PNG + yaw/pitch/eye | `canvas` snapshot |
| 4 Tag | Open-vocab names + boxes | `grok-4.6` vision (sidecar `/analyze`) |
| 5 Cluster | Merge armchair/sofa/seat | name key in sidecar |
| 6 Cards | Product stills | `grok-imagine-image-2.0` **edits** (sidecar `/card`) |
| 7 Repair (later) | Missing / blurry views | NVIDIA ArtiFixer (not in the sidecar) |

## Operator walkthrough (HUD, top to bottom)

1. **Load** a file (drop it, or `?url=…`). `?scene=synthetic` gives the deterministic two-sphere scene used by the tests.
2. **Grupos** → *Calcular grupos* (≈ 4 s for 262k Gaussians on 4 CPU cores). *Vista Grupos* colours superpoints; a click promotes one to an instance. Real scenes need a higher *Umbral* than the default 0.3 (113k superpoints on the alarm clock).
3. **Segmentación** → choose the mask source (*SAM 2 (navegador)* needs `vendor/ml/` or network), *Vistas*, *Sesgo fondo* → *Levantar máscaras* → *Exportar instancias* writes `instancias.json` + `etiquetas.u32` (download, and `artifacts/segmentaciones/<escena>/<fecha>/` through the sidecar).
4. **Instancias** → *Nombrar instancias (Grok)*, *Embeddings CLIP*, search box (tick *Búsqueda semántica* for CLIP), per-row *Aislar / Ocultar / Teñir / Nombrar / Tarjeta / Malla*.
5. **Edición** → pick a tool (*Rectángulo*, *Pincel*, *Esfera 3D*, *Superpunto*; *Añadir* / *Quitar*) and drag or click on the canvas; *Nueva instancia* / *Añadir a seleccionada* / *Quitar (fondo)*. With an instance selected: move / rotate / scale, *Duplicar*, *Borrar*, *Fusionar con #*, *Renombrar*, *Deshacer* / *Rehacer* (Ctrl+Z). *Exportar* the instance or the visible scene as PLY / .splat / SPZ / compressed PLY; *Guardar ops.jsonl*.
6. **Malla** → *Vistas*, *Vóxeles*, *Arista*, depth *media* (fast) or *mediana* (K-buffer) → *Malla de la seleccionada* → GLB download and `artifacts/mallas/<escena>/<id>.glb`.

Every panel exposes a scripting API for tests and batch runs: `window.__gsGroups`, `__gsSegment`, `__gsNames`, `__gsEdit`, `__gsMesh`, `__gsLoad`, `__gsRenderer`, `__gsInstances`, `__gsCamera`.

## Artifacts layout

Generated files never go to the repo root (`artifacts/` and `img_output/` are gitignored except their READMEs):

```text
artifacts/
  segmentaciones/<escena>/<AAAA-MM-DD_HHMMSS>/
    instancias.json          # schema: shared/schemas/instancias.schema.json (validated on export)
    etiquetas.u32            # N × uint32 little-endian, instance per gaussian (0 = fondo)
    etiquetas_base.u32       # F5: labels the ops.jsonl replays over (optional)
    ops.jsonl                # F5: one edit op per line (optional)
  exportaciones/<escena>/
    instancia-<id>.<ply|splat|spz|compressed.ply>   # or escena.<ext>; <same>.json metadata; ops.jsonl
  mallas/<escena>/<id>.glb   # + <id>.json (method, vertices, triangles, timings, warning)
  test-results/              # Playwright traces
img_output/<fecha>-<nombre>/imagine.<jpg|png>       # Imagine cards
vendor/ml/                   # SAM 2.1 + CLIP weights + transformers.js (scripts/download-ml-models.sh)
```

`instancias.json` (plan §3.3) carries `version`, `escena`, `fecha`, `fuente {formato, n_gaussianas, sh_grado}`, `metodo {mascaras, levantamiento, sesgo_fondo, umbral_iou, difusion_iter, vistas, k_buffer}`, `n_instancias`, optional `embeddings {modelo, dimension}`, and per instance `id_instancia`, `nombre`, `nombre_es`, `categoria`, `confianza`, `n_gaussianas`, `bbox`, `color`, `vistas`, `malla`, `embedding_clip`. Validate any file with `node scripts/validate-instancias.mjs <instancias.json>`.

`ops.jsonl` ops: `asignar {id_instancia, rangos}`, `transformar {id_instancia, xform[16]}`, `borrar` / `restaurar {id_instancia}`, `duplicar {id_instancia, nueva, xform?}`, `fusionar {origen, destino}`, `renombrar {id_instancia, nombre_es}`; replaying the file over `etiquetas_base.u32` reproduces the final labels and transforms (`shared/edit-ops.js`).

## Sidecar endpoints (`semantic_sidecar/server.py`, port 8766)

| Endpoint | Purpose | Needs xAI |
| --- | --- | --- |
| `GET /health` | backends, model names, output folders | no |
| `POST /analyze` | open-vocab tags + Imagine cards for captured views | yes |
| `POST /card` | Imagine edit of one crop | yes |
| `POST /segment` | masks per view: `grok-boxes` (ellipses) or `sam` (`SAM_BACKEND=module:function`) | grok-boxes only |
| `POST /name` | `nombre`/`nombre_es`/`categoria`/`confianza` per isolated instance render (`NAME_BACKEND=mock` offline) | grok only |
| `POST /segmentaciones` | save `instancias.json`, `etiquetas.u32`, `ops.jsonl`, `etiquetas_base.u32` | no |
| `POST /exportaciones` | save an exported instance / scene file + metadata | no |
| `POST /mallas` | save a GLB + metadata | no |

The browser never sees API keys; the sidecar reads `.env`. Everything but Grok naming, tags and Imagine cards works without the sidecar (downloads only).

Tests: `npm test` (unit) and `npm run test:e2e` (Chromium WebGPU, offscreen); see [testing.md](testing.md).

Do **not** replace step 2 with Imagine/GPT-Image/Gemini image generation. Those APIs have no camera or opacity, so masks cannot lift back to Gaussians.
