#!/usr/bin/env bash
# Launch a dedicated Chrome profile with WebGPU (Vulkan) and open this viewer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8090}"
PROFILE="${CHROME_PROFILE:-$HOME/.cache/chrome-webgpu-3dgs}"
URL="${1:-http://127.0.0.1:${PORT}/gaussian_splatting_webgpu/?url=../splats/model.splat}"

mkdir -p "$PROFILE"

if ! curl -sf -o /dev/null "http://127.0.0.1:${PORT}/gaussian_splatting_webgpu/" 2>/dev/null; then
  echo "Starting static server on :${PORT} from ${ROOT}"
  python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$ROOT" >/tmp/gs-webgpu-http.log 2>&1 &
  echo $! >/tmp/gs-webgpu-http.pid
  sleep 0.4
fi

# Do not open chrome://newtab — a scratch --user-data-dir often trips
# "Requested load of chrome://newtab/ for incorrect profile type".
exec google-chrome \
  --user-data-dir="$PROFILE" \
  --no-first-run \
  --no-default-browser-check \
  --disable-sync \
  --disable-background-networking \
  --ignore-gpu-blocklist \
  --enable-unsafe-webgpu \
  --enable-webgpu-developer-features \
  --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE \
  --use-angle=vulkan \
  --ozone-platform=x11 \
  --new-window \
  "$URL"
