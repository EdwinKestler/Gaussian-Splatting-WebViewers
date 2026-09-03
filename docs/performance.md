# Reproducible performance profiles

`scripts/profile-webgpu.mjs` profiles the shipped alarm-clock scene through the
same public browser APIs used by the HUD. It records the WebGPU adapter, repeated
offscreen colour/depth/contribution passes, F2 graph construction and the full
F6 mesh pipeline. Results are written to the gitignored
`artifacts/profiles/webgpu-<timestamp>.json`.

```bash
# Linux/NVIDIA: use the real display and discrete adapter.
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia npm run profile:webgpu

# Also run four-view SAM 2.1 segmentation (local vendor/ml recommended).
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  WEBGPU_PROFILE_SAM=1 WEBGPU_PROFILE_RUNS=3 npm run profile:webgpu
```

Set `WEBGPU_PROFILE_HEADLESS=1` only on platforms whose Chromium build exposes
hardware WebGPU headlessly. On the Linux/NVIDIA machine below, headless Vulkan
returned `requestAdapter() = null`; headed Chromium on `DISPLAY=:1` exposed the
real adapter. A profile is accepted as hardware evidence only when the emitted
adapter is neither empty nor `swiftshader`/`software`.

## 2026-09-03 baseline: RTX 3090 Ti

Environment:

- NVIDIA GeForce RTX 3090 Ti (24,564 MiB), driver 580.173.02;
- WebGPU adapter `vendor=nvidia`, `architecture=ampere`, `float32-blendable`;
- Intel Core i9-14900K, 32 logical CPUs, 60 GiB RAM;
- Playwright 1.56.1 / Chromium 141, Node 24.15.0;
- scene `alarm_clock_generated.splat`, 262,144 Gaussians;
- offscreen target 512 × 320; five measured runs after warm-up except the
  contribution pass, whose first run is retained.

| Stage | Result |
| --- | ---: |
| Colour readback | 5.0 ms median, 5.2 ms p95 |
| Mean-depth readback | 7.7 ms median, 7.9 ms p95 |
| Exact K-buffer contribution pass | 760.8 ms median, 937.1 ms p95 |
| F2 graph | 1,926.6 ms worker / 1,968.3 ms wall; 113,154 superpoints |
| F6 mesh, 24 views, 96³, 256 px | 908.1 ms total |
| F6 stage split | 182 ms rendering, 628 ms fusion, 74 ms extraction |
| F6 output | 12,425 vertices, 24,872 triangles, 746,996-byte GLB |

The mesh has Euler characteristic −10, so fast output is not evidence of
printability. The validation/repair gate must report boundary and non-manifold
edges before a file is called ready for fabrication.

## SAM 2.1 and association profile

With four views and eight prompts per view, the reciprocal overlap graph retained
20 SAM masks and produced **13 global instances with 7 cross-view merges**. The
previous greedy baseline documented in the paper produced 17 instances with only
2 merges. This is a useful before/after on the same alarm-clock workflow, but it
is not an accuracy score because the scene has no human instance ground truth.

The ONNX SAM 2.1 FP16 graph did not execute through WebGPU on this stack (`device
does not support fp16`) and correctly fell back to WASM. Image encoding took
12.24–12.54 s per view; the entire four-view segmentation took 69.29 s. Renderer
passes remained on the NVIDIA adapter. Optimizing or replacing the SAM export is
therefore separate from optimizing the WebGPU rasterizer.

## Interpretation

- The old 374-second SwiftShader mesh number is not representative of a real
  GPU; the same configured pipeline completed in about one second here.
- F2 is CPU/worker-bound and is already under the original 3-second target on
  this machine, but the portable `node scripts/bench-graph.mjs 1000000` benchmark
  still takes about 4.1 seconds for one million synthetic Gaussians.
- K-buffer synchronization remains the largest renderer cost and should be the
  first GPU optimization target.
- SAM inference is currently WASM-bound even when rendering uses a discrete GPU.
