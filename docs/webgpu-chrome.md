# Launch WebGPU Chrome

The WebGPU 3DGS viewer needs a **real Chrome window** with WebGPU (Vulkan on Linux) and an **HTTP server**. `file://` will not work.

## 1. Serve the repo from the root

Do **not** start the server inside `gaussian_splatting_webgpu/`. The parse worker imports `../shared/splat-io.js`, which only exists when the document root is the repository.

```bash
cd /path/to/Gaussian-Splatting-WebViewers
python3 -m http.server 8090 --bind 127.0.0.1
```

Viewer URL:

```text
http://127.0.0.1:8090/gaussian_splatting_webgpu/
```

Load a local file with a query:

```text
http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat
```

The default scene is `splats/alarm_clock_generated.splat` (compact 32-byte SH0, shipped in git). `demo.ply` is a tiny SH0 sphere (Demo PLY button). For the full Kerbl / INRIA radiance field, drop a trained `point_cloud/iteration_*/point_cloud.ply` (it must contain `f_rest_*` spherical-harmonic bands). Compact `.splat` files cannot reconstruct SH1–3. Leave **Point cloud debug** unchecked.

## 2. Launch Chrome with WebGPU

Use a **fresh profile** and pass a **real `http://` URL**. Do not open `chrome://newtab` on a scratch `--user-data-dir`.

```bash
rm -rf ~/.cache/chrome-webgpu-3dgs

google-chrome \
  --user-data-dir=$HOME/.cache/chrome-webgpu-3dgs \
  --no-first-run \
  --no-default-browser-check \
  --disable-sync \
  --disable-background-networking \
  --ignore-gpu-blocklist \
  --enable-unsafe-webgpu \
  --enable-webgpu-developer-features \
  --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE \
  --use-angle=vulkan \
  --new-window \
  'http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat'
```

One-liner from this repo (starts the server if needed):

```bash
./gaussian_splatting_webgpu/launch-webgpu-chrome.sh
```

Optional:

```bash
PORT=8090 CHROME_PROFILE=$HOME/.cache/chrome-webgpu-3dgs \
  ./gaussian_splatting_webgpu/launch-webgpu-chrome.sh \
  'http://127.0.0.1:8090/gaussian_splatting_webgpu/?url=../splats/alarm_clock_generated.splat'
```

## 3. Confirm WebGPU is on

In that Chrome window:

1. Open `chrome://gpu` and check that **WebGPU** is enabled (not software-only / disabled).
2. In DevTools console:

```js
console.log(!!navigator.gpu);
const a = await navigator.gpu.requestAdapter();
console.log(a);
```

`navigator.gpu` must be `true` and `requestAdapter()` must return an adapter.

## What those log lines mean

These are **not** WebGPU failures:

| Log | Meaning |
| --- | --- |
| `Created TensorFlow Lite XNNPACK delegate for CPU` | Chrome ML sidecar. Ignore. |
| `Registration response error message: DEPRECATED_ENDPOINT` | Push/GCM against a temp profile. Ignore. |
| `Requested load of chrome://newtab/ for incorrect profile type` | Scratch `--user-data-dir` cannot open the new-tab page. **Pass an `http://` URL** and use a **new** profile directory. Do not reuse `/tmp/chrome-webgpu-profile`. |

This line is a **good** sign on NVIDIA/Linux:

```text
maxDynamicUniformBuffersPerPipelineLayout artificially reduced ...
```

It means the Vulkan/WebGPU path started.

## Linux notes

- WebGPU goes through **Vulkan**. `vulkaninfo --summary` should list your GPU.
- Do **not** add `--use-angle=swiftshader` if you want the discrete GPU.
- **Headless Chrome does not expose `navigator.gpu`.** Use a normal window.
- If a Chrome window is already open **without** these flags, a new command may attach to it and ignore the flags. Always pass a dedicated `--user-data-dir`.
