// Playwright configuration for the WebGPU e2e smoke tests.
//
// The single project launches the full Chromium build (channel "chromium",
// not the headless shell) with flags that expose WebGPU in headless mode.
// Without a GPU the adapter is SwiftShader (Vulkan software rasteriser);
// see docs/testing.md for the caveats (never render to a canvas in tests).
//
// Override the launch flags on a machine with a real GPU:
//   WEBGPU_ARGS="--enable-unsafe-webgpu --use-angle=vulkan" npx playwright test

import { defineConfig } from "@playwright/test";

const PORT = 8091;
const BASE_URL = `http://127.0.0.1:${PORT}`;

/** Flags that make headless Chromium expose navigator.gpu via SwiftShader. */
const SWIFTSHADER_ARGS = [
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--enable-features=WebGPU,Vulkan",
  "--use-angle=vulkan",
  "--use-vulkan=swiftshader",
  "--enable-unsafe-swiftwebgpu",
];

/** Space-separated WEBGPU_ARGS replaces the SwiftShader flags entirely. */
function resolveWebgpuArgs(env) {
  const raw = (env.WEBGPU_ARGS ?? "").trim();
  if (!raw) return SWIFTSHADER_ARGS;
  const args = raw.split(/\s+/).filter(Boolean);
  console.log(`[playwright] WEBGPU_ARGS override: ${args.join(" ")}`);
  return args;
}

export default defineConfig({
  testDir: "tests/e2e",
  outputDir: "artifacts/test-results",
  timeout: 90_000,
  retries: 0,
  reporter: "list",
  fullyParallel: false,
  use: {
    baseURL: BASE_URL,
    headless: true,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        channel: "chromium",
        headless: true,
        launchOptions: {
          args: resolveWebgpuArgs(process.env),
        },
      },
    },
  ],
  webServer: {
    // Document root must be the repo root: the parse worker imports ../shared/splat-io.js.
    command: `python3 -m http.server ${PORT} --bind 127.0.0.1`,
    url: `${BASE_URL}/index.html`,
    reuseExistingServer: true,
    timeout: 30_000,
    stdout: "ignore",
    stderr: "ignore",
  },
});
