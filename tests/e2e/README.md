# Pruebas e2e (Playwright + Chromium WebGPU)

Pruebas que arrancan Chromium headless con WebGPU (SwiftShader si no hay GPU),
sirven el repositorio por `http://127.0.0.1:8091` y comprueban:

| Fichero | Qué comprueba |
| --- | --- |
| `f1-identity.spec.mjs` + `pages/f1-harness.js`, `pages/f1-identity.html` | Aceptación F1: `selftest.html` reporta `SELFTEST_OK`; `pick()` y el búfer ID devuelven la etiqueta de cada esfera; ocultar/aislar; profundidad de una gaussiana aislada con error < 1 % (medida 0,0 %); normales; traslación por instancia; clic de ratón en el visor selecciona la instancia y el HUD lo muestra. |
| `f3-lift.spec.mjs` | Aceptación F3: K-buffer exacto + profundidad mediana, levantamiento de 6 vistas con IoU > 0,9, panel Segmentación y exportación. |
| `f2-groups.spec.mjs` | Aceptación F2 (grafo de superpuntos): 2 grupos en la escena sintética, colores distintos en la vista «Grupos», clic → grupo convertido en instancia, difusión de etiquetas. |
| `smoke.spec.mjs` | `navigator.gpu.requestAdapter()` no es nulo; `shared/splat-io.js` decodifica `demo.ply` (4292 gaussianas, SH0) dentro del navegador; un pase de cómputo con `atomicAdd` y un render *offscreen* a `rgba8unorm` sobreviven sin perder el dispositivo; `parse-worker.js` decodifica `demo.ply` con la copia vendorizada de GaussForge con el CDN bloqueado. |

```bash
npm run test:e2e            # npx playwright test
WEBGPU_ARGS="--enable-unsafe-webgpu --use-angle=vulkan" npm run test:e2e   # GPU real
```

Reglas para escribir pruebas nuevas:

- **Nunca** configurar un contexto WebGPU de canvas (`canvas.getContext("webgpu")`,
  `context.configure`, `getCurrentTexture`): bajo SwiftShader eso pierde el dispositivo.
  Renderizar a `GPUTexture` fuera de pantalla y leer con `copyTextureToBuffer` + `mapAsync`.
- La raíz de documentos es la raíz del repositorio (el worker importa `../shared/splat-io.js`).
- Las trazas y capturas quedan en `artifacts/test-results/` (ignorado en git).

Detalles, flags y resolución de problemas en [`docs/testing.md`](../../docs/testing.md).
