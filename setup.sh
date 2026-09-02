#!/usr/bin/env bash
# Prepare this repo on a new machine. Does not start HTTP servers.
#
#   ./setup.sh              checks, dirs, .env template
#   ./setup.sh --sidecar    also install Pillow (Python)
#   ./setup.sh --tests      also npm install (Node ≥ 22)
#   ./setup.sh --e2e        also Playwright Chromium
#   ./setup.sh --all        sidecar + tests + e2e
#
# Then: python3 -m http.server 8090 --bind 127.0.0.1
#       ./semantic_sidecar/launch.sh
#       ./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

WITH_SIDECAR=0
WITH_TESTS=0
WITH_E2E=0

usage() {
  cat <<'EOF'
Prepare this repo on a new machine. Does not start HTTP servers.

  ./setup.sh              checks, dirs, .env template
  ./setup.sh --sidecar    also install Pillow (Python)
  ./setup.sh --tests      also npm install (Node ≥ 22)
  ./setup.sh --e2e        also Playwright Chromium
  ./setup.sh --all        sidecar + tests + e2e

Then:
  python3 -m http.server 8090 --bind 127.0.0.1
  ./semantic_sidecar/launch.sh
  ./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --sidecar) WITH_SIDECAR=1 ;;
    --tests) WITH_TESTS=1 ;;
    --e2e) WITH_TESTS=1; WITH_E2E=1 ;;
    --all) WITH_SIDECAR=1; WITH_TESTS=1; WITH_E2E=1 ;;
    *) echo "unknown flag: $1" >&2; usage 1 ;;
  esac
  shift
done

ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
fail() { printf '  [fail] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

ERRORS=0
OS="$(uname -s)"

echo "==> ${ROOT}"
echo "==> checks"

if have python3; then
  PY_VER="$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
  PY_MAJ="$(python3 -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')"
  if [[ "${PY_MAJ}" -ge 310 ]]; then
    ok "python3 ${PY_VER}"
  else
    fail "python3 ${PY_VER} (need ≥ 3.10 for the sidecar)"
    ERRORS=$((ERRORS + 1))
  fi
else
  fail "python3 not found"
  ERRORS=$((ERRORS + 1))
fi

CHROME=""
for c in google-chrome google-chrome-stable chromium chromium-browser microsoft-edge; do
  if have "$c"; then CHROME="$c"; break; fi
done
if [[ -z "${CHROME}" && "${OS}" == Darwin && -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fi
if [[ -n "${CHROME}" ]]; then
  ok "chrome: ${CHROME}"
else
  warn "Chrome/Chromium not in PATH (WebGPU viewer needs Chrome 113+ / Edge 113+)"
fi

if [[ "${OS}" == Linux ]]; then
  if have vulkaninfo; then
    ok "vulkaninfo present"
  else
    warn "vulkaninfo not found — Linux WebGPU uses Vulkan (install vulkan-tools)"
  fi
  if [[ -n "${CHROME}" && "${CHROME}" != google-chrome && "${CHROME}" != google-chrome-stable ]]; then
    warn "launch-webgpu-chrome.sh calls google-chrome; symlink or edit the script if you use ${CHROME}"
  fi
elif [[ "${OS}" == Darwin ]]; then
  warn "launch-webgpu-chrome.sh is Linux/Vulkan (google-chrome). On macOS serve :8090 and open Chrome yourself."
fi

if [[ -f vendor/gaussforge/index.web.js && -f vendor/gaussforge/gauss_forge.web.js ]]; then
  ok "vendored GaussForge (vendor/gaussforge/)"
else
  warn "vendor/gaussforge/ missing — run scripts/vendor-gaussforge.sh 0.6.0 (needs network)"
fi

echo "==> directories"
mkdir -p splats img_output artifacts
chmod +x \
  setup.sh \
  semantic_sidecar/launch.sh \
  gaussian_splatting_webgpu/launch-webgpu-chrome.sh \
  scripts/vendor-gaussforge.sh \
  2>/dev/null || true
ok "splats/  img_output/  artifacts/"

if [[ -f gaussian_splatting_webgpu/demo.ply ]]; then
  ok "demo scene: gaussian_splatting_webgpu/demo.ply"
fi
if [[ -f splats/model.splat ]]; then
  ok "default scene: splats/model.splat"
else
  warn "splats/ is gitignored and empty here. Copy a .splat or point_cloud.ply:"
  echo "         mkdir -p splats && cp /path/to/model.splat splats/model.splat"
  echo "         (viewer still loads demo.ply via the Demo PLY button)"
fi

echo "==> env"
if [[ ! -f .env.example ]]; then
  fail ".env.example missing"
  ERRORS=$((ERRORS + 1))
elif [[ ! -f .env ]]; then
  cp .env.example .env
  ok "wrote .env from .env.example (add XAI_API_KEY for the sidecar)"
else
  ok ".env already exists (not overwritten)"
fi

echo "==> sidecar (Pillow)"
if python3 -c 'from PIL import Image' >/dev/null 2>&1; then
  ok "Pillow importable"
elif [[ "${WITH_SIDECAR}" -eq 1 ]]; then
  echo "  installing Pillow…"
  if have apt-get && have sudo && [[ "${OS}" == Linux ]]; then
    sudo apt-get install -y python3-pil
  else
    python3 -m pip install --user Pillow || python3 -m pip install --user --break-system-packages Pillow
  fi
  if python3 -c 'from PIL import Image' >/dev/null 2>&1; then
    ok "Pillow installed"
  else
    fail "Pillow still missing — apt install python3-pil  or  pip install Pillow"
    ERRORS=$((ERRORS + 1))
  fi
else
  warn "Pillow not installed (needed for ./semantic_sidecar/launch.sh). Re-run: ./setup.sh --sidecar"
fi

if [[ "${WITH_TESTS}" -eq 1 ]]; then
  echo "==> tests (npm)"
  if ! have node || ! have npm; then
    fail "Node/npm not found (need Node ≥ 22 for npm test)"
    ERRORS=$((ERRORS + 1))
  else
    NODE_MAJ="$(node -p 'process.versions.node.split(".")[0]')"
    if [[ "${NODE_MAJ}" -lt 22 ]]; then
      warn "Node $(node -v) — package.json engines.node is ≥ 22"
    else
      ok "node $(node -v)"
    fi
    npm install
    ok "npm install"
    if [[ "${WITH_E2E}" -eq 1 ]]; then
      npx playwright install chromium
      ok "Playwright Chromium"
    fi
  fi
fi

echo
if [[ "${ERRORS}" -gt 0 ]]; then
  echo "setup finished with ${ERRORS} error(s)."
  exit 1
fi

echo "setup ok. Next:"
echo "  python3 -m http.server 8090 --bind 127.0.0.1"
echo "  ./semantic_sidecar/launch.sh                 # optional, needs XAI_API_KEY in .env"
echo "  ./gaussian_splatting_webgpu/launch-webgpu-chrome.sh"
echo
echo "  Viewer:  http://127.0.0.1:8090/gaussian_splatting_webgpu/"
echo "  Demo:    http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=./demo.ply"
echo "  Notes:   docs/pipeline.md   docs/webgpu-chrome.md"
if [[ "${WITH_TESTS}" -eq 1 ]]; then
  echo "  Tests:   npm test"
  [[ "${WITH_E2E}" -eq 1 ]] && echo "           npm run test:e2e"
fi
