#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SIDECAR_HOST="${SIDECAR_HOST:-127.0.0.1}"
export SIDECAR_PORT="${SIDECAR_PORT:-8766}"
cd "$ROOT"
exec python3 semantic_sidecar/server.py
