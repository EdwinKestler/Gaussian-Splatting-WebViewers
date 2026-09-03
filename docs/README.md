# Docs

| File | What it is |
| --- | --- |
| [open-vocab-3dgs-imagine-pipeline-paper.md](open-vocab-3dgs-imagine-pipeline-paper.md) | Technical report: pipeline, flowchart, references, third-party attributions |
| [open-vocab-3dgs-imagine-pipeline-paper.pdf](open-vocab-3dgs-imagine-pipeline-paper.pdf) | Same report as PDF (regenerate: `npm install --no-save marked && node scripts/render-report-pdf.mjs docs/open-vocab-3dgs-imagine-pipeline-paper.md docs/open-vocab-3dgs-imagine-pipeline-paper.pdf`) |
| [plan-segmentacion-edicion-3dgs.md](plan-segmentacion-edicion-3dgs.md) | Plan: clustering, segmentación, clasificación, malla y edición de objetos sobre 3DGS (sin entrenamiento por escena) |
| [../setup.sh](../setup.sh) | One-shot machine setup (dirs, `.env`, dependency checks) |
| [pipeline.md](pipeline.md) | Operator workflow: stages F0–F6, HUD walkthrough, `artifacts/` layout, data files, sidecar endpoints |
| [webgpu-chrome.md](webgpu-chrome.md) | Linux WebGPU Chrome / Vulkan launch |
| [testing.md](testing.md) | Pruebas unitarias (Node) y e2e (Playwright + Chromium WebGPU): comandos, flags, SwiftShader |
| [figures/pipeline-flowchart.png](figures/pipeline-flowchart.png) | Figure 1 (tagging + Imagine cards) |
| [figures/segmentation-pipeline.png](figures/segmentation-pipeline.png) | Figure 3: segmentation, editing and meshing pipeline (plan F0–F6); source `figures/segmentation-pipeline.mmd`, rendered with `scripts/render-mermaid.mjs` |
| [figures/demo-potted-plants-pair.png](figures/demo-potted-plants-pair.png) | Figure 2 (3DGS crop vs Imagine edit) |
