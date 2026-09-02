# artifacts/

Carpeta para **salidas generadas** por los visores, el sidecar y las pruebas. Todo lo que hay
aquí se ignora en git (regla `artifacts/**` en `.gitignore`), salvo este README, para que la
raíz del repositorio no se llene de ficheros producidos por scripts.

## Subcarpetas previstas

| Carpeta | Contenido |
| --- | --- |
| `segmentaciones/` | Etiquetas por gaussiana (`etiquetas.u32`), `instancias.json`, registros `ops.jsonl` de edición y máscaras 2D levantadas. |
| `mallas/` | Mallas por instancia (`.glb`, `.ply`) generadas por TSDF/Poisson, con sus metadatos (`*.meta.json`). |
| `exportaciones/` | Escenas o instancias exportadas (`.ply`, `.splat`, `.spz`, `.sog`, …) más `instancias.json`. |
| `test-results/` | Salida de Playwright (trazas, capturas y vídeos de las pruebas e2e). Se regenera en cada ejecución. |

Las subcarpetas se crean bajo demanda; no hace falta versionarlas vacías.

## Convenciones

- Nombrar las salidas con la escena y una marca de tiempo, p. ej.
  `segmentaciones/model_20260902-1530/instancias.json`.
- No guardar aquí datos de entrada (nubes `.ply`/`.splat` originales van en `splats/`, también
  ignorado) ni dependencias (`vendor/`).
- Los scripts de un solo uso que escriben aquí viven en `scripts/` y deben aceptar la carpeta
  de salida como argumento, con `artifacts/<subcarpeta>/` como valor por defecto.
