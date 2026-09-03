# Docs

| File | What it is |
| --- | --- |
| [open-vocab-3dgs-imagine-pipeline-paper.md](open-vocab-3dgs-imagine-pipeline-paper.md) | Technical report: SAM 2.1 browser segmentation, reciprocal cross-view association, real-GPU profile, “From Splats to Solids” workflow, splat-to-mesh/3MF mathematics, figures, references and attributions |
| [open-vocab-3dgs-imagine-pipeline-paper.pdf](open-vocab-3dgs-imagine-pipeline-paper.pdf) | Same report as PDF (regenerate after `npm ci`: `node scripts/render-report-pdf.mjs docs/open-vocab-3dgs-imagine-pipeline-paper.md docs/open-vocab-3dgs-imagine-pipeline-paper.pdf`; equations render with KaTeX) |
| [plan-segmentacion-edicion-3dgs.md](plan-segmentacion-edicion-3dgs.md) | Plan: clustering, segmentación, clasificación, malla y edición de objetos sobre 3DGS (sin entrenamiento por escena) |
| [performance.md](performance.md) | Perfiles reproducibles WebGPU/CPU, línea base RTX 3090 Ti y límites del backend SAM 2.1 |
| [../setup.sh](../setup.sh) | One-shot machine setup (dirs, `.env`, dependency checks) |
| [../models/README.md](../models/README.md) | Optional ONNX (SAM 2 / CLIP); `main` also uses `scripts/download-ml-models.sh` → `vendor/ml/` |
| [pipeline.md](pipeline.md) | Operator workflow: stages F0–F6, HUD walkthrough, `artifacts/` layout, data files, sidecar endpoints |
| [webgpu-chrome.md](webgpu-chrome.md) | Linux WebGPU Chrome / Vulkan launch |
| [testing.md](testing.md) | Pruebas unitarias (Node) y e2e (Playwright + Chromium WebGPU): comandos, flags, SwiftShader |
| [GITHUB_PAGES.md](GITHUB_PAGES.md) | Publicación estática, allowlist del artefacto y límites del sidecar local |
| [figures/pipeline-flowchart.png](figures/pipeline-flowchart.png) | Figure 1 (tagging + Imagine cards) |
| [figures/segmentation-pipeline.png](figures/segmentation-pipeline.png) | Figure 3: segmentation, editing and meshing pipeline (plan F0–F6); source `figures/segmentation-pipeline.mmd`, rendered with `scripts/render-mermaid.mjs` |
| [figures/demo-potted-plants-pair.png](figures/demo-potted-plants-pair.png) | Figure 2 (3DGS crop vs Imagine edit) |
| `figures/workflow-alarm-clock-{splat,render,slicer}.*` | Curated real-run triptych used by the GitHub Pages hero: isolated splat, object render and complete-scene 3MF in Cura |
