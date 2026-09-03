#!/usr/bin/env bash
# Optional: download SAM 2.1 tiny + CLIP ViT-B/32 ONNX into models/.
# Not required for the current Grok/Imagine sidecar pipeline.
#
#   ./scripts/download-models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/models"
HUB="${HF_ENDPOINT:-https://huggingface.co}"
SAM_MODEL="${SAM_MODEL:-onnx-community/sam2.1-hiera-tiny-ONNX}"
SAM_DTYPE="${SAM_DTYPE:-fp16}"
CLIP_MODEL="${CLIP_MODEL:-Xenova/clip-vit-base-patch32}"
CLIP_DTYPE="${CLIP_DTYPE:-quantized}"

log() { printf '[download-models] %s\n' "$*" >&2; }

fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    log "exists: ${dest#"$ROOT"/}"
    return
  fi
  log "get $url"
  curl -fsSL --retry 3 --retry-delay 2 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

hub_file() {
  fetch "$HUB/$1/resolve/main/$2" "$OUT/$1/$2"
}

hub_optional() {
  if curl -fsSI --retry 2 "$HUB/$1/resolve/main/$2" >/dev/null 2>&1; then
    hub_file "$1" "$2"
  else
    log "skip missing $1/$2"
  fi
}

mkdir -p "$OUT"
log "SAM $SAM_MODEL ($SAM_DTYPE)"
for f in config.json preprocessor_config.json processor_config.json; do
  hub_optional "$SAM_MODEL" "$f"
done
for part in vision_encoder prompt_encoder_mask_decoder; do
  suffix=""
  [[ "$SAM_DTYPE" != fp32 ]] && suffix="_$SAM_DTYPE"
  hub_file "$SAM_MODEL" "onnx/${part}${suffix}.onnx"
  hub_optional "$SAM_MODEL" "onnx/${part}${suffix}.onnx_data"
done

log "CLIP $CLIP_MODEL ($CLIP_DTYPE)"
for f in config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.json merges.txt; do
  hub_optional "$CLIP_MODEL" "$f"
done
for part in vision_model text_model; do
  suffix=""
  [[ "$CLIP_DTYPE" != fp32 ]] && suffix="_$CLIP_DTYPE"
  hub_file "$CLIP_MODEL" "onnx/${part}${suffix}.onnx"
  hub_optional "$CLIP_MODEL" "onnx/${part}${suffix}.onnx_data"
done

python3 - "$OUT" "$SAM_MODEL" "$SAM_DTYPE" "$CLIP_MODEL" "$CLIP_DTYPE" <<'PY'
import json, os, sys
out, sam, samd, clip, clipd = sys.argv[1:6]
total = 0
for dirpath, _, files in os.walk(out):
    for name in files:
        if name in {"README.md", "manifest.json"}:
            continue
        total += os.path.getsize(os.path.join(dirpath, name))
manifest = {
    "version": 1,
    "required_for_current_pipeline": False,
    "sam": {"id": sam, "dtype": samd},
    "clip": {"id": clip, "dtype": "q8" if clipd == "quantized" else clipd},
    "bytes": total,
}
with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")
print(f"[download-models] {total/1e6:.1f} MB under {out}", file=sys.stderr)
PY
