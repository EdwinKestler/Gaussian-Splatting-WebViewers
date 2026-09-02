# Semantic sidecar (Grok vision + Imagine 2.0)

Local HTTP service for the WebGPU viewer. Captured 3DGS frames are tagged with **Grok vision** (`grok-4.6`). Optional **object cards** use **Grok Imagine Image 2.0** (`grok-imagine-image-2.0`) as *edits* of 3DGS crops — not as a replacement rasterizer.

API keys stay in the repo-root `.env` (`XAI_API_KEY`). The browser never sees them.

```bash
./semantic_sidecar/launch.sh
# http://127.0.0.1:8766/health
```

Then in the WebGPU viewer: **Tag scene (Grok)**. Default is 1 view + 2 cards. Increase views only if you need cross-view clustering.

Imagine outputs are stored under `img_output/` (source crop + `imagine.jpg` + `meta.json`).

ArtiFixer is not in this process: it needs a local 80GB-class GPU and NVIDIA non-commercial weights, not the Hugging Face inference API.
