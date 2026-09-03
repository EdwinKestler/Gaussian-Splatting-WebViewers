#!/usr/bin/env bash
# One-shot: fetch transformers.js (with its ONNX Runtime wasm files) and the
# in-browser model weights used by gaussian_splatting_webgpu/ml-browser.js
# (SAM 2.1 hiera-tiny for masks, CLIP ViT-B/32 for embeddings) into vendor/ml/.
# vendor/ml/ is gitignored: ~200 MB of weights never enter git. Serving the
# repository root then lets the viewer load everything without the Hub or CDN.
#
#   scripts/download-ml-models.sh            # default models
#   SAM_MODEL=onnx-community/sam2.1-hiera-small-ONNX scripts/download-ml-models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/vendor/ml"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.2.0}"
SAM_MODEL="${SAM_MODEL:-onnx-community/sam2.1-hiera-tiny-ONNX}"
SAM_DTYPE="${SAM_DTYPE:-fp16}"
CLIP_MODEL="${CLIP_MODEL:-Xenova/clip-vit-base-patch32}"
CLIP_DTYPE="${CLIP_DTYPE:-quantized}"   # file suffix on the Hub (quantized = q8)
HUB="${HF_ENDPOINT:-https://huggingface.co}"

log() { printf '[download-ml-models] %s\n' "$*" >&2; }

fetch() { # url dest
  mkdir -p "$(dirname "$2")"
  if [ -s "$2" ]; then log "ya existe: ${2#$ROOT/}"; return; fi
  log "descargando ${1}"
  curl -fsSL --retry 3 --retry-delay 2 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

# ---- transformers.js dist (npm tarball; includes ort-wasm-*.wasm/.mjs)
TMP="$(mktemp -d "${TMPDIR:-/tmp}/ml-models.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
log "transformers.js ${TRANSFORMERS_VERSION}"
curl -fsSL --retry 3 -o "$TMP/transformers.tgz" "https://registry.npmjs.org/@huggingface/transformers/-/transformers-${TRANSFORMERS_VERSION}.tgz"
mkdir -p "$OUT/transformers"
tar -xzf "$TMP/transformers.tgz" -C "$TMP" package/dist package/package.json package/LICENSE 2>/dev/null || tar -xzf "$TMP/transformers.tgz" -C "$TMP"
cp "$TMP"/package/dist/transformers.min.js "$OUT/transformers/"
cp "$TMP"/package/dist/*.wasm "$TMP"/package/dist/*.mjs "$OUT/transformers/" 2>/dev/null || true
cp "$TMP"/package/LICENSE "$OUT/transformers/LICENSE" 2>/dev/null || true
cp "$TMP"/package/package.json "$OUT/transformers/package.json"

# ---- ONNX Runtime Web wasm binaries (the exact version transformers.js depends on)
ORT_VERSION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['dependencies']['onnxruntime-web'].lstrip('^~'))" "$OUT/transformers/package.json")"
log "onnxruntime-web ${ORT_VERSION}"
curl -fsSL --retry 3 -o "$TMP/ort.tgz" "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-${ORT_VERSION}.tgz"
mkdir -p "$TMP/ort" && tar -xzf "$TMP/ort.tgz" -C "$TMP/ort" package/dist
cp "$TMP"/ort/package/dist/ort-wasm-simd-threaded*.wasm "$TMP"/ort/package/dist/ort-wasm-simd-threaded*.mjs "$OUT/transformers/"

# ---- model files (layout expected by transformers.js env.localModelPath)
hub_file() { # repo file
  fetch "$HUB/$1/resolve/main/$2" "$OUT/models/$1/$2"
}
hub_optional() { # repo file (skip on 404)
  if curl -fsSI --retry 2 "$HUB/$1/resolve/main/$2" >/dev/null 2>&1; then hub_file "$1" "$2"; else log "no existe en el Hub (omitido): $1/$2"; fi
}

log "SAM: $SAM_MODEL ($SAM_DTYPE)"
for f in config.json preprocessor_config.json processor_config.json; do hub_optional "$SAM_MODEL" "$f"; done
for part in vision_encoder prompt_encoder_mask_decoder; do
  suffix=""; [ "$SAM_DTYPE" != "fp32" ] && suffix="_$SAM_DTYPE"
  hub_file "$SAM_MODEL" "onnx/${part}${suffix}.onnx"
  hub_optional "$SAM_MODEL" "onnx/${part}${suffix}.onnx_data"
done

log "CLIP: $CLIP_MODEL ($CLIP_DTYPE)"
for f in config.json preprocessor_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do hub_optional "$CLIP_MODEL" "$f"; done
for part in vision_model text_model; do
  suffix=""; [ "$CLIP_DTYPE" != "fp32" ] && suffix="_$CLIP_DTYPE"
  hub_file "$CLIP_MODEL" "onnx/${part}${suffix}.onnx"
  hub_optional "$CLIP_MODEL" "onnx/${part}${suffix}.onnx_data"
done

# ---- manifest read by ml-browser.js
ORT_VERSION_OUT="$ORT_VERSION" python3 - "$OUT" "$TRANSFORMERS_VERSION" "$SAM_MODEL" "$SAM_DTYPE" "$CLIP_MODEL" "$CLIP_DTYPE" <<'PY'
import json, os, sys
out, tv, sam, samd, clip, clipd = sys.argv[1:7]
total = 0
for dirpath, _, files in os.walk(out):
    for f in files:
        total += os.path.getsize(os.path.join(dirpath, f))
manifest = {
    "version": 1,
    "transformers": {"version": tv, "entry": "transformers/transformers.min.js", "onnxruntime_web": os.environ.get("ORT_VERSION_OUT", "")},
    "sam": {"id": sam, "dtype": samd},
    "clip": {"id": clip, "dtype": "q8" if clipd == "quantized" else clipd},
    "bytes": total,
}
with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")
print(f"[download-ml-models] listo: {total/1e6:.1f} MB en {out}", file=sys.stderr)
PY
