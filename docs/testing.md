# Pruebas automatizadas

El repositorio tiene dos niveles de pruebas (plan, fase F0):

| Nivel | Herramienta | Carpeta | Comando |
| --- | --- | --- | --- |
| Unitarias | `node --test` (Node ≥ 22, sin dependencias) | `tests/unit/` | `npm test` |
| e2e | Playwright 1.56 + Chromium con WebGPU | `tests/e2e/` | `npm run test:e2e` |
| Ambas | — | — | `npm run test:all` |

Los ayudantes compartidos (generadores de PLY/`.splat` sintéticos) viven en `tests/helpers/`.

## Instalación

```bash
npm install                     # instala @playwright/test 1.56.1 (fijado)
```

`@playwright/test` va fijado a la misma versión que los navegadores instalados en la máquina.
Si Playwright no encuentra Chromium, apuntar `PLAYWRIGHT_BROWSERS_PATH` a la carpeta de
navegadores (por ejemplo `/opt/pw-browsers`) o ejecutar `npx playwright install chromium`
en una máquina con red y sin la variable `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD`.

## Pruebas unitarias

```bash
npm test                        # node --test "tests/unit/**/*.test.?(c|m)js"
node --test tests/unit/splat-io.test.mjs   # un solo fichero
```

Node 22 no acepta un directorio como argumento de `node --test`, por eso el script usa un
patrón *glob*: sólo se ejecutan los ficheros `tests/unit/**/*.test.mjs` (o `.js`/`.cjs`).
Si el patrón no encuentra ficheros, `node --test` termina con código 0 y cero pruebas.

Cubren `shared/splat-io.js` (PLY 3DGS/2DGS/nube de puntos, `.splat` de 32 y 44 bytes, SH0–3,
filas con escalas no finitas, entrada `SharedArrayBuffer`, submuestreo y `boundsFromGaussians`)
con ficheros construidos en memoria; no tocan disco ni red.
`tests/unit/graph.test.mjs` cubre `shared/graph.js`: kNN contra fuerza bruta, CSR simétrico sin duplicados,
pesos Mahalanobis + color, dos blobs → 2 componentes, puntos aislados, difusión de etiquetas y un
presupuesto de tiempo (250 k gaussianas, variable `GRAPH_BUDGET_MS_250K`). El criterio «1 M gaussianas
en < 3 s en un portátil» se mide con `node scripts/bench-graph.mjs 1000000` (`GRAPH_BUDGET_MS`).
`tests/unit/naming.test.mjs` cubre `shared/naming.js` (encuadre de instancias, búsqueda normalizada, fusión de nombres).
`tests/unit/lift.test.mjs` cubre `shared/lift.js` con matrices de contribución simuladas: argmax con sesgo de
fondo, histogramas y contención, asociación con ids permutados y vistas parciales, `liftViews` de extremo a
extremo (IoU > 0,9) y el esquema de `instancias.json`.

## Pruebas e2e con WebGPU

| Fichero | Cubre |
| --- | --- |
| `tests/e2e/smoke.spec.mjs` | Adaptador WebGPU, `splat-io` en el navegador, cómputo + render *offscreen* bajo SwiftShader, `parse-worker.js` con GaussForge vendorizado y el CDN bloqueado (criterio F0 «carga sin red») |
| `tests/e2e/f4-ml-browser.spec.mjs` | *Opcional* (`ML_E2E=1` + `scripts/download-ml-models.sh`): SAM 2.1 en el navegador (transformers.js, WASM) levanta las dos esferas en exactamente 2 instancias desde indicaciones de superpuntos, CLIP incrusta los recortes aislados y «an orange ball» / «a blue sphere» eligen la esfera correcta, la exportación lleva `embedding_clip` unitario de 512 d. ≈ 1 min bajo SwiftShader |
| `tests/e2e/f4-naming.spec.mjs` | Aceptación F4 con el sidecar simulado por `page.route` (sin clave): recorte aislado por instancia (sólo sus gaussianas, fondo blanco, cámara restaurada), `/name` → nombre_es/categoría/confianza en el panel, búsqueda con acentos y selección con Intro, exportación con nombres, tarjeta Imagine ligada a `id_instancia`, error claro sin sidecar |
| `tests/e2e/f3-lift.spec.mjs` | Aceptación F3: el pase K-buffer da contribuciones exactas (0 gaussianas mal asignadas desde una vista) y la profundidad mediana analítica; 6 vistas con ids permutados → 2 instancias con IoU 3D > 0,9 tras asociación por superpuntos y difusión; el visor levanta con la fuente «prueba» y exporta `instancias.json` |
| `tests/e2e/f2-groups.spec.mjs` | Aceptación F2: el grafo del worker separa las dos esferas en exactamente 2 grupos, la vista «Grupos» las colorea distinto (lectura *offscreen*), un clic con la vista activa convierte el grupo en instancia y `Difundir etiquetas` respeta las etiquetas limpias |
| `tests/e2e/f1-identity.spec.mjs` | Aceptación F1 sobre la escena sintética de dos esferas (`shared/synthetic.js`): `selftest.html`, ID/`pick()`, ocultar/aislar, profundidad < 1 %, normales, transformación por instancia, clic en el visor (`?scene=synthetic&offscreen=1`). El arnés vive en `tests/e2e/pages/` |


```bash
npm run test:e2e                # npx playwright test
ML_E2E=1 npx playwright test tests/e2e/f4-ml-browser.spec.mjs   # modelos en el navegador (pesos en vendor/ml/)
npx playwright test --headed    # ver el navegador (requiere pantalla)
npx playwright test -g "adapter"   # filtrar por título
```

`playwright.config.mjs` define un único proyecto, `chromium-webgpu`:

- Lanza el Chromium completo (`channel: "chromium"`, no el *headless shell*) con
  `--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-features=WebGPU,Vulkan
  --use-angle=vulkan --use-vulkan=swiftshader --enable-unsafe-swiftwebgpu`.
- Arranca `python3 -m http.server 8091 --bind 127.0.0.1` desde la raíz del repositorio
  (`webServer`, con `reuseExistingServer: true`) y usa `http://127.0.0.1:8091` como `baseURL`.
  Es imprescindible servir por `http://`: en `about:blank` o `file://` no existe
  `navigator.gpu`, y el worker del visor importa `../shared/splat-io.js`, así que la raíz
  de documentos debe ser la raíz del repositorio.
- Tiempo máximo por prueba 90 s, sin reintentos, reporter `list`.

### Advertencia: SwiftShader y el canvas

Sin GPU, el adaptador es **SwiftShader** (`vendor=google`, `architecture=swiftshader`).
Funcionan los pases de cómputo, los atómicos, los búferes de almacenamiento, los render
*offscreen* a `GPUTexture` y la lectura de vuelta (`copyTextureToBuffer`, `mapAsync`).
**No funciona el canvas**: cualquier `canvas.getContext("webgpu")` seguido de
`context.configure()` o `getCurrentTexture()` pierde el dispositivo con el mensaje
`A valid external Instance reference no longer exists`.

Por eso las pruebas e2e renderizan a texturas fuera de pantalla y leen los píxeles con
`copyTextureToBuffer`; nunca presentan en un canvas. La prueba `compute + offscreen render
survive under SwiftShader` registra `device.lost` y falla si el dispositivo se pierde, de
modo que una regresión en este punto se detecta de inmediato.

### Ejecutar contra una GPU real

La variable `WEBGPU_ARGS` (flags separados por espacios) sustituye por completo los flags
de SwiftShader:

```bash
# Linux con Vulkan (NVIDIA/AMD/Intel)
WEBGPU_ARGS="--enable-unsafe-webgpu --ignore-gpu-blocklist --use-angle=vulkan" npm run test:e2e

# Windows / macOS: normalmente basta con
WEBGPU_ARGS="--enable-unsafe-webgpu" npm run test:e2e
```

Con GPU real el canvas sí funciona, pero las pruebas siguen usando *offscreen* para que el
mismo conjunto pase en CI y en portátil. Más detalles sobre lanzar Chrome con Vulkan en
[`webgpu-chrome.md`](webgpu-chrome.md).

## Dónde quedan los resultados

| Salida | Ruta |
| --- | --- |
| Trazas, capturas y vídeos de Playwright (`outputDir`) | `artifacts/test-results/` |
| Trazas retenidas sólo en fallos (`trace: retain-on-failure`) | `artifacts/test-results/<prueba>/trace.zip` → `npx playwright show-trace <zip>` |
| Salida de `node --test` | consola (formato `spec`) |

`artifacts/` está ignorado en git salvo su `README.md`; `test-results/` y
`playwright-report/` en la raíz también están ignorados por si alguien ejecuta Playwright con
otra configuración.

## Resolución de problemas

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| `navigator.gpu` es `undefined` | La página no se sirve por `http://` o faltan flags | Comprobar `baseURL` y `WEBGPU_ARGS` |
| `requestAdapter()` devuelve `null` | Chromium *headless shell* en vez del completo, o GPU bloqueada | Mantener `channel: "chromium"`; añadir `--ignore-gpu-blocklist` |
| `A valid external Instance reference no longer exists` | Uso de canvas bajo SwiftShader | Renderizar a `GPUTexture` fuera de pantalla |
| `Executable doesn't exist` | Navegadores en otra ruta | `PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers npm run test:e2e` |
| El servidor no arranca en 30 s | Puerto 8091 ocupado por otro proceso | Liberarlo o dejar que `reuseExistingServer` lo reutilice si sirve la raíz del repo |
