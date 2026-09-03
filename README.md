# Gaussian Splatting Web Viewers

Experimental browser viewers for [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

The original Three.js and A-Frame demos are still here. A new **WebGPU 3DGS viewer** uses [GaussForge](https://github.com/3dgscloud/GaussForge) to decode PLY / SPLAT / SPZ / KSPLAT / SOG, then sorts and rasterizes on the GPU. GaussForge is vendored under `vendor/gaussforge/` (Apache-2.0, single-file WASM build), so the viewer decodes every format without network access; the jsDelivr copy is only a fallback.

On a new machine:

```bash
git clone https://github.com/EdwinKestler/Gaussian-Splatting-WebViewers.git
cd Gaussian-Splatting-WebViewers
./setup.sh                 # dirs, .env template, dependency checks
./setup.sh --sidecar       # also install Pillow (Grok sidecar)
./setup.sh --tests         # also npm install (Node ≥ 22)
./setup.sh --e2e           # also Playwright Chromium
./setup.sh --all           # sidecar + tests + e2e
```

Shipped compact demo: `splats/alarm_clock_generated.splat` (~8 MB). Other files under `splats/` stay gitignored — copy a trained `point_cloud.ply` or `model.splat` locally, or drop a file in the viewer. `gaussian_splatting_webgpu/demo.ply` is a tiny SH0 sphere.

Serve the repo from the **repository root** (the WebGPU worker imports `../shared/splat-io.js`, so `file://` and serving only the subfolder will fail):

```bash
cd Gaussian-Splatting-WebViewers
python3 -m http.server 8090 --bind 127.0.0.1
```

Then open [http://127.0.0.1:8090/](http://127.0.0.1:8090/) or the WebGPU viewer at [http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat](http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat). Default scene is `splats/alarm_clock_generated.splat`. Pipeline notes: [docs/pipeline.md](docs/pipeline.md). Roadmap for per-Gaussian segmentation, classification, mesh and object editing: [docs/plan-segmentacion-edicion-3dgs.md](docs/plan-segmentacion-edicion-3dgs.md). Paper (flowchart, references, attributions): [docs/open-vocab-3dgs-imagine-pipeline-paper.md](docs/open-vocab-3dgs-imagine-pipeline-paper.md) · [PDF](docs/open-vocab-3dgs-imagine-pipeline-paper.pdf).

### WebGPU Chrome (Linux / Vulkan)

A scratch `--user-data-dir` often cannot open `chrome://newtab` (`incorrect profile type`). Use a **fresh profile** and pass a real `http://` URL:

```bash
rm -rf ~/.cache/chrome-webgpu-3dgs

google-chrome \
  --user-data-dir=$HOME/.cache/chrome-webgpu-3dgs \
  --no-first-run \
  --no-default-browser-check \
  --ignore-gpu-blocklist \
  --enable-unsafe-webgpu \
  --enable-webgpu-developer-features \
  --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE \
  --use-angle=vulkan \
  --new-window \
  'http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat'
```

Or:

```bash
./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
```

Ignore `DEPRECATED_ENDPOINT` and TensorFlow Lite lines. The new-tab profile error means you launched without a URL or reused a bad profile (do not use `/tmp/chrome-webgpu-profile`). Confirm WebGPU in `chrome://gpu` or with `!!navigator.gpu` in DevTools.

Full notes: [docs/webgpu-chrome.md](docs/webgpu-chrome.md).

## Viewers

| Path | Renderer | What it is for |
| --- | --- | --- |
| `gaussian_splatting_webgpu/` | WebGPU | Recommended 3DGS viewer. GaussForge decode (PLY/SPZ/KSPLAT/SOG/SPLAT) + GPU sort |
| `gaussian_splatting_1/` | Three.js WebGL | Compact 32-byte instanced-mesh viewer |
| `gaussian_splatting_2_three.js/` | Three.js WebGL | Covariance-buffer viewer (Kellogg-style) |
| `gaussian_splatting_2_aframe/` | A-Frame WebGL | Same splats inside an A-Frame scene |
| `splat_converter/` | — | Convert PLY ↔ 32-byte splat ↔ 44-byte splat |

Every viewer accepts `?url=` pointing at a scene file, and every WebGL/WebGPU viewer accepts a file drop. The WebGPU viewer also reads `.spz`, `.ksplat`, and `.sog` through GaussForge.

## File formats

There are two `.splat` layouts in the wild. They are **not interchangeable**, and **neither is the full 3DGS radiance field**.

| Layout | Bytes / gaussian | Contents | Used by |
| --- | --- | --- | --- |
| Compact | 32 | `xyz f32`, `scale f32`, `RGBA u8`, `quat u8` | Viewer 1, converter default, antimatter15/splat |
| Extended | 44 | `xyz f32`, `scale f32`, `RGBA u8`, `quat f32` | Original Viewer 2 |
| INRIA PLY | ~236 (SH3) | float mean, log-scale, quat, logit opacity, SH DC + `f_rest_0…44` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) `point_cloud/iteration_*/point_cloud.ply` |

A compact `.splat` has **already thrown away** spherical harmonics 1–3 and quantized rotations to 8 bits. Loading it can only show a view-independent approximation (or a point cloud if that debug toggle is on). For the paper’s radiance field, drop a trained **`point_cloud.ply`**.

Shared parser: `shared/splat-io.js`. WebGL viewers still pack to 32-byte rows. The WebGPU viewer keeps float covariance + SH0–3.

## WebGPU 3DGS viewer

`gaussian_splatting_webgpu/` is the recommended renderer. Format decoding uses **[GaussForge](https://github.com/3dgscloud/GaussForge)** (`@gaussforge/wasm`) — the same conversion IR as [3DGS Viewer](https://www.3dgsviewers.com/) — then a WebGPU compute sort + the [Kerbl et al. / INRIA](https://github.com/graphdeco-inria/gaussian-splatting) rasterizer.

- Loads **PLY**, **compressed PLY**, **SPLAT**, **KSPLAT**, **SPZ**, and **SOG**
- Keeps **float** scale / quaternion / opacity (no 8-bit packing on the GPU path)
- View-dependent **SH degree 0–3** (`computeColorFromSH` from `diff-gaussian-rasterization`)
- Paper EWA projection, 3-sigma extent, fragment `α = o · exp(-½ r²)`, front-to-back transmittance
- GPU 16-bit counting sort every frame (histogram + prefix sum + scatter)
- HUD warns when the file is compact `.splat` / SH0 so it cannot match the official viewer
- Export back through GaussForge (PLY / SPLAT / SPZ / KSPLAT / SOG)
- Falls back to `shared/splat-io.js` (`toGaussianCloud`) if WASM cannot load
- Reads 2D Gaussian Splatting PLYs (two log-scales) and plain point-cloud PLYs (`variant` reported by `describePly`)
- **Identity layer (plan F1):** a `u32` label per Gaussian plus an instance table (rigid transform, tint, visible/selected) in the shader; output modes colour / depth / normal / ID with `renderOffscreen()` readback and `pick(x, y)`; HUD panel **Instancias** (click selects, *Aislar* / *Ocultar* / *Teñir*, `Escena sintética (2 esferas)`)
- `?offscreen=1` skips the canvas context (headless SwiftShader tests); `selftest.html` renders offscreen and reports `SELFTEST_OK`
- `?scene=synthetic` loads the deterministic two-sphere scene from `shared/synthetic.js` (labels 1 = esfera A, 2 = esfera B) used by the F1 acceptance tests
- `renderOffscreen({ mode: 1 })` returns the alpha-weighted *mean* view distance; `renderContributions()` returns the exact 2DGS *median* depth (`medianDepth`) from the K-buffer
- **Mask lifting (plan F3):** `gaussian_splatting_webgpu/contrib-pass.js` is a K-buffer pass that accumulates per-Gaussian α·T mass per 2D mask label (exact: the depth-sorted list is drawn in chunks that split on overflow) and resolves the 2DGS *median* depth; `shared/lift.js` does the closed-form FlashSplat assignment, Gaga-style cross-view association over F2 superpoints, and the `instancias.json` / `etiquetas.u32` export. HUD panel **Segmentación**: views, background bias, mask source (*Etiquetas actuales (prueba)*, *Sidecar: cajas de Grok*, *Sidecar: SAM*), *Levantar máscaras*, *Exportar instancias*. Sidecar endpoints `/segment` and `/segmentaciones` (saves under `artifacts/segmentaciones/`)
- **Superpoint graph (plan F2):** `shared/graph.js` (pure JS, runs in `shared/graph-worker.js`) builds a kNN graph (k = 10, hash grid), weights edges with the symmetric Mahalanobis distance and the SH0 colour, cuts weak edges and labels connected components as superpoints; HUD panel **Grupos** (*Calcular grupos*, *Vista Grupos* = colour mode «Grupos», *Difundir etiquetas*); with the Grupos view active a click promotes the superpoint to an F1 instance. `?scene=synthetic&labels=0` loads the spheres without labels. Benchmark: `node scripts/bench-graph.mjs 1000000`

**Point cloud debug** is a diagnostic overlay. Leave it unchecked for the radiance-field rendering.

Requires Chrome 113+, Edge 113+, or another browser with WebGPU. If WebGPU is missing, the page links back to the WebGL viewers. Launch instructions: [docs/webgpu-chrome.md](docs/webgpu-chrome.md).

Optional **open-vocab tags** and **Imagine 2.0 object cards**: start `./semantic_sidecar/launch.sh` (reads `XAI_API_KEY` from `.env`, port 8766), then use **Tag scene (Grok)** in the WebGPU HUD. Tags run on captured 3DGS rasters; Imagine only edits those crops. ArtiFixer is a separate local-GPU stage (see `semantic_sidecar/README.md`).

## Tests

```bash
npm install          # @playwright/test 1.56.1 (browsers are not downloaded)
npm test             # 79 Node unit tests for shared/splat-io.js, shared/graph.js and shared/lift.js
npm run test:e2e     # 16 Playwright tests on Chromium WebGPU (SwiftShader when no GPU), offscreen rendering only
```

Details, GPU flags (`WEBGPU_ARGS`) and the SwiftShader canvas caveat: [docs/testing.md](docs/testing.md). Generated outputs go under `artifacts/` (gitignored).

## A-Frame component

```html
<script src="https://aframe.io/releases/1.4.2/aframe.min.js"></script>
<script src="gaussian-splatting.js"></script>
<a-entity gaussian-splatting="splatUrl: ./scene.ply; slider: true; splatColor: color"></a-entity>
```

| Schema | Default | Meaning |
| --- | --- | --- |
| `splatUrl` | HuggingFace train.splat | `.ply` or `.splat` URL |
| `initialPosition` | `0 0 0` | unused by the camera; put the camera entity where you want |
| `downsampleFactor` | `1` | keep 1/N gaussians |
| `vertexCount` | `1000000` | slider maximum |
| `splatSize` | `1159.58…` | focal length used as splat scale |
| `splatPixelDiscard` | `2.0` | gaussian radius cutoff |
| `slider` | `true` | on-screen vertex-count and splat-size sliders |
| `splatColor` | `color` | `color`, `grayscale`, `blackAndWhite`, `green` |

## Heritage

- Viewer 1 started from [quadjr/aframe-gaussian-splatting](https://github.com/quadjr/aframe-gaussian-splatting) (MIT, Kevin Kwok, Junya Kuwada)
- Viewer 2 started from [mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) (MIT, Mark Kellogg)
- Converter and the 32-byte layout follow [antimatter15/splat](https://github.com/antimatter15/splat) (MIT, Kevin Kwok)
- Multi-format decode uses [GaussForge](https://github.com/3dgscloud/GaussForge) (Apache-2.0)
- The WebGPU path follows the same 3DGS projection as those projects, with GPU counting sort instead of a CPU worker

## License

MIT. See `LICENSE`.
