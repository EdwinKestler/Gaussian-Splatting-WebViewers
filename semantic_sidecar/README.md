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
  `grok-boxes` rasteriza las cajas de Grok vision como elipses (las cajas grandes primero). `sam` carga `SAM_BACKEND=paquete.modulo:funcion` desde `.env`; la función recibe `(imagen PIL, prompts)` y devuelve `(etiquetas HxW, objetos)`. Sin backend configurado responde 500 con un mensaje claro; los pesos de SAM 3 (licencia SAM) no se vendorizan. Alternativa sin sidecar: la fuente «SAM 2 (navegador)» del visor ejecuta SAM 2.1 con transformers.js (`gaussian_splatting_webgpu/ml-browser.js`, pesos locales con `scripts/download-ml-models.sh`).
- `POST /segmentaciones` `{escena, instancias, etiquetas_b64, ops_jsonl?, etiquetas_base_b64?}` guarda `instancias.json` y `etiquetas.u32` en `artifacts/segmentaciones/<escena>/<fecha>/`; con F5 también `ops.jsonl` y `etiquetas_base.u32` (las etiquetas sobre las que se reproduce el registro).

## Mallas por instancia (F6)

- `POST /mallas` `{escena, id_instancia, glb_b64, metadatos?}` guarda `artifacts/mallas/<escena>/<id_instancia>.glb` (+ `<id>.json`). La malla la calcula el visor (TSDF + surface nets en `shared/tsdf.js`); el sidecar sólo la persiste. Open3D (`ScalableTSDFVolume`, Poisson) queda como backend opcional futuro: `pip install open3d` arrastra dependencias de visualización (plotly, dash) y no se instala aquí.

## Exportaciones por objeto (F5)

- `POST /exportaciones` `{escena, id_instancia|null, formato, bytes_b64, metadatos?, ops_jsonl?}` guarda `artifacts/exportaciones/<escena>/instancia-<id>.<formato>` (o `escena.<formato>`) más `<mismo nombre>.json` y `ops.jsonl`. Formatos: `ply`, `splat`, `spz`, `compressed.ply`, `ksplat`, `sog` (los bytes los produce el visor).

## Nombres por instancia (F4)

- `POST /name` `{instances:[{id, hint, png_b64}], backend:"grok"|"mock"}` → por instancia `{id_instancia, ok, nombre, nombre_es, categoria, confianza, descripcion_es}`. El visor envía el render aislado de cada instancia (fondo blanco). `NAME_BACKEND=mock` en `.env` devuelve nombres deterministas sin clave (para pruebas); una instancia fallida no pierde las demás (`ok:false` + `error`).

