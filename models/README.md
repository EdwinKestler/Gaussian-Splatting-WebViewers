# Local model weights

**You do not need Hugging Face models to run the current pipeline on `main`.**

That path is:

1. WebGPU 3DGS rasterizer (GaussForge is already in `vendor/gaussforge/`)
2. Grok 4.6 vision + Imagine 2.0 **API** (`XAI_API_KEY` in `.env`)

No SAM, CLIP, or ArtiFixer checkpoint is loaded.

## Optional (later F3/F4, other branch)

`scripts/download-models.sh` fetches ONNX weights into this folder (~150–250 MB) so a future in-browser SAM 2 / CLIP path can run without the Hub:

| Folder | Hub id | Use |
| --- | --- | --- |
| `sam2.1-hiera-tiny-ONNX/` | `onnx-community/sam2.1-hiera-tiny-ONNX` | Promptable masks (Apache-2.0) |
| `clip-vit-base-patch32/` | `Xenova/clip-vit-base-patch32` | Instance embeddings (MIT) |

```bash
./scripts/download-models.sh
```

Weights are gitignored. Do **not** put NVIDIA ArtiFixer here (OneWay Noncommercial, ~80 GB GPU).

`manifest.json` is written after a successful download.
