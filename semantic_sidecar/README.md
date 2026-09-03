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

## Máscaras para el levantamiento (F3)

- `POST /segment` `{views:[{png_b64,width,height}], backend:"auto"|"grok-boxes"|"sam"}` → por vista una máscara PNG de 8 bits (0 = fondo, k = objeto k) y `objects:[{id,name,confidence}]`.
  `grok-boxes` rasteriza las cajas de Grok vision como elipses (las cajas grandes primero). `sam` carga `SAM_BACKEND=paquete.modulo:funcion` desde `.env`; la función recibe `(imagen PIL, prompts)` y devuelve `(etiquetas HxW, objetos)`. Sin backend configurado responde 500 con un mensaje claro; los pesos de SAM 3 (licencia SAM) no se vendorizan.
- `POST /segmentaciones` `{escena, instancias, etiquetas_b64}` guarda `instancias.json` y `etiquetas.u32` en `artifacts/segmentaciones/<escena>/<fecha>/`.

