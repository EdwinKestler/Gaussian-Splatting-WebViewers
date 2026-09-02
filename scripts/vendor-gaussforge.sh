#!/usr/bin/env bash
# Vendoriza @gaussforge/wasm (Apache-2.0) en vendor/gaussforge/ para que el visor
# WebGPU cargue GaussForge sin red (plan F0: "GaussForge vendorizado").
#
# Uso:
#   scripts/vendor-gaussforge.sh            # versión por defecto (0.6.0)
#   scripts/vendor-gaussforge.sh 0.7.0      # versión como argumento
#   VERSION=0.7.0 scripts/vendor-gaussforge.sh
#
# Descarga el tarball oficial del registro npm, extrae sólo la build web
# (index.web.js + gauss_forge.web.js con el wasm embebido), README.md y
# package.json, descarga el texto de la licencia (LICENSE) desde el repositorio
# de origen (el tarball no lo incluye; Apache-2.0 §4(a) exige redistribuirlo) y
# regenera NOTICE.md con versión, licencia y sumas SHA-256.
# No copia .map, .d.ts ni builds de Node. Imprime la versión vendorizada en stdout.
set -euo pipefail

VERSION="${1:-${VERSION:-0.6.0}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/vendor/gaussforge"
TARBALL_URL="https://registry.npmjs.org/@gaussforge/wasm/-/wasm-${VERSION}.tgz"
FILES=(dist/index.web.js dist/gauss_forge.web.js README.md package.json)
LICENSE_REPO="https://raw.githubusercontent.com/3dgscloud/GaussForge"
# Primero la etiqueta de la versión; si no existe, la rama principal.
LICENSE_URLS=("${LICENSE_REPO}/v${VERSION}/LICENSE" "${LICENSE_REPO}/main/LICENSE")

log() { printf '[vendor-gaussforge] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

if ! [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]]; then
  die "versión inválida '${VERSION}' (se esperaba semver, p. ej. 0.6.0)"
fi
for tool in curl tar python3 sha256sum; do
  command -v "${tool}" >/dev/null 2>&1 || die "falta la herramienta '${tool}'"
done

TMP="$(mktemp -d "${TMPDIR:-/tmp}/vendor-gaussforge.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

log "descargando ${TARBALL_URL}"
curl -fsSL --retry 3 --retry-delay 2 -o "${TMP}/wasm.tgz" "${TARBALL_URL}" \
  || die "no se pudo descargar la versión ${VERSION}"
tar -xzf "${TMP}/wasm.tgz" -C "${TMP}"
PKG="${TMP}/package"

for f in "${FILES[@]}"; do
  [[ -f "${PKG}/${f}" ]] || die "el paquete no contiene ${f}"
done

PKG_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "${PKG}/package.json")"
PKG_LICENSE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("license",""))' "${PKG}/package.json")"
[[ "${PKG_VERSION}" == "${VERSION}" ]] || die "package.json declara ${PKG_VERSION}, se pidió ${VERSION}"
[[ "${PKG_LICENSE}" == "Apache-2.0" ]] || log "AVISO: la licencia declarada es '${PKG_LICENSE}', no Apache-2.0; revisa NOTICE.md"

# La build web debe ser autocontenida (wasm embebido); si apareciera un .wasm
# suelto, la carga sin red necesitaría copiarlo también.
if compgen -G "${PKG}/dist/*.wasm" >/dev/null; then
  die "la build ${VERSION} trae .wasm separado; actualiza este script para copiarlo"
fi

# El tarball npm no trae LICENSE ("files": ["dist", "README.md"]); se toma del repositorio.
LICENSE_URL=""
for url in "${LICENSE_URLS[@]}"; do
  if curl -fsSL --retry 3 --retry-delay 2 -o "${TMP}/LICENSE" "${url}"; then
    LICENSE_URL="${url}"
    break
  fi
  log "AVISO: no se pudo descargar ${url}"
done
[[ -n "${LICENSE_URL}" ]] || die "no se pudo descargar el texto de la licencia (LICENSE)"
grep -q "Apache License" "${TMP}/LICENSE" || die "el LICENSE descargado no parece Apache-2.0"
log "licencia descargada desde ${LICENSE_URL}"

mkdir -p "${DEST}"
for f in "${FILES[@]}"; do
  cp -f "${PKG}/${f}" "${DEST}/$(basename "${f}")"
done
cp -f "${TMP}/LICENSE" "${DEST}/LICENSE"

sum_of() { sha256sum "${DEST}/$1" | cut -d' ' -f1; }
# wc -c es portable (stat -c es sólo GNU; en BSD/macOS es stat -f %z).
size_of() { wc -c < "${DEST}/$1" | tr -d ' '; }
# Filas calculadas antes del heredoc para que un fallo aborte bajo set -e.
row_of() { printf '| `%s` | %s | `%s` |' "$1" "$(size_of "$1")" "$(sum_of "$1")"; }
ROW_INDEX="$(row_of index.web.js)"
ROW_MODULE="$(row_of gauss_forge.web.js)"
ROW_README="$(row_of README.md)"
ROW_PACKAGE="$(row_of package.json)"
ROW_LICENSE="$(row_of LICENSE)"

cat > "${DEST}/NOTICE.md" <<NOTICE
# GaussForge (WASM) vendorizado

- **Paquete:** \`@gaussforge/wasm\` — https://www.npmjs.com/package/@gaussforge/wasm
- **Origen:** https://github.com/3dgscloud/GaussForge (directorio \`wasm\`)
- **Versión:** ${PKG_VERSION}
- **Licencia:** ${PKG_LICENSE} (ver \`package.json\`; texto completo en \`LICENSE\`, junto a este archivo)
- **Tarball:** ${TARBALL_URL}
- **LICENSE:** ${LICENSE_URL} (el tarball npm no lo incluye)

Archivos copiados tal cual del tarball (sin \`.map\`, \`.d.ts\` ni builds de Node),
más \`LICENSE\` descargado del repositorio de origen:

| Archivo | Bytes | SHA-256 |
| --- | --- | --- |
${ROW_INDEX}
${ROW_MODULE}
${ROW_README}
${ROW_PACKAGE}
${ROW_LICENSE}

\`gauss_forge.web.js\` es un módulo Emscripten de un solo archivo con el binario wasm
embebido, por lo que el visor carga GaussForge sin acceso a la red.
\`gaussian_splatting_webgpu/parse-worker.js\` importa primero
\`vendor/gaussforge/index.web.js\`; sólo si falla recurre al CDN (jsDelivr) y, en
último término, al decodificador integrado de \`shared/splat-io.js\`.

Para actualizar a otra versión:

\`\`\`bash
scripts/vendor-gaussforge.sh <versión>     # p. ej. scripts/vendor-gaussforge.sh ${PKG_VERSION}
\`\`\`

y ajusta la versión del URL de respaldo (\`GAUSSFORGE_URL\`) en \`parse-worker.js\`.
NOTICE

if ! grep -q "@gaussforge/wasm@${PKG_VERSION}/" "${REPO_ROOT}/gaussian_splatting_webgpu/parse-worker.js" 2>/dev/null; then
  log "AVISO: el URL de respaldo (CDN) en parse-worker.js no apunta a ${PKG_VERSION}; actualízalo a mano"
fi

log "vendorizado en ${DEST}:"
ls -l "${DEST}" >&2
echo "${PKG_VERSION}"
