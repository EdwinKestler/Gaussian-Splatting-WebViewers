# GaussForge (WASM) vendorizado

- **Paquete:** `@gaussforge/wasm` — https://www.npmjs.com/package/@gaussforge/wasm
- **Origen:** https://github.com/3dgscloud/GaussForge (directorio `wasm`)
- **Versión:** 0.6.0
- **Licencia:** Apache-2.0 (ver `package.json`; texto completo en `LICENSE`, junto a este archivo)
- **Tarball:** https://registry.npmjs.org/@gaussforge/wasm/-/wasm-0.6.0.tgz
- **LICENSE:** https://raw.githubusercontent.com/3dgscloud/GaussForge/v0.6.0/LICENSE (el tarball npm no lo incluye)

Archivos copiados tal cual del tarball (sin `.map`, `.d.ts` ni builds de Node),
más `LICENSE` descargado del repositorio de origen:

| Archivo | Bytes | SHA-256 |
| --- | --- | --- |
| `index.web.js` | 4435 | `89a2f70dcf839368de2da32ab72d552dbc85befec1691df70be9e9e68640225e` |
| `gauss_forge.web.js` | 1431875 | `4fcd31322e72b55770a273ca672a3581642aa26da6d38d64f2b03d13277a81b3` |
| `README.md` | 4207 | `e86cb118a785ba5ed1846e85861f1f71547084a6aa496fb51294988b5eac73e5` |
| `package.json` | 1622 | `231c63b91ae0edcc1ec2f09cb5959c6442f0e376903b37fd4f63e8157abe85bc` |
| `LICENSE` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

`gauss_forge.web.js` es un módulo Emscripten de un solo archivo con el binario wasm
embebido, por lo que el visor carga GaussForge sin acceso a la red.
`gaussian_splatting_webgpu/parse-worker.js` importa primero
`vendor/gaussforge/index.web.js`; sólo si falla recurre al CDN (jsDelivr) y, en
último término, al decodificador integrado de `shared/splat-io.js`.

Para actualizar a otra versión:

```bash
scripts/vendor-gaussforge.sh <versión>     # p. ej. scripts/vendor-gaussforge.sh 0.6.0
```

y ajusta la versión del URL de respaldo (`GAUSSFORGE_URL`) en `parse-worker.js`.
