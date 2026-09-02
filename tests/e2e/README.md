# Pruebas e2e (Playwright + Chromium WebGPU)

Pruebas de humo que arrancan Chromium headless con WebGPU (SwiftShader si no hay GPU),
sirven el repositorio por `http://127.0.0.1:8091` y comprueban:

| Fichero | Qué comprueba |
| --- | --- |
| `smoke.spec.mjs` | `navigator.gpu.requestAdapter()` no es nulo; `shared/splat-io.js` decodifica `demo.ply` (4292 gaussianas, SH0) dentro del navegador; un pase de cómputo con `atomicAdd` y un render *offscreen* a `rgba8unorm` sobreviven sin perder el dispositivo. |

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
