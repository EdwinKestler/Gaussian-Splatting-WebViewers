import { WebGPUSplatRenderer } from "./gpu-renderer.js";

const DEFAULT_PLY = "./demo.ply";
const SAMPLE_SPLAT =
  "https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat";
const LOCAL_SPLAT = "../splat_converter/test.splat";

const $ = (id) => document.getElementById(id);

function mul4(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const out = new Float32Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) / (near - far);
  out[11] = -1;
  out[14] = (2 * far * near) / (near - far);
  return out;
}

function lookAt(eye, target, up) {
  const zx = eye[0] - target[0];
  const zy = eye[1] - target[1];
  const zz = eye[2] - target[2];
  let zlen = Math.hypot(zx, zy, zz) || 1;
  const z0 = zx / zlen;
  const z1 = zy / zlen;
  const z2 = zz / zlen;
  let x0 = up[1] * z2 - up[2] * z1;
  let x1 = up[2] * z0 - up[0] * z2;
  let x2 = up[0] * z1 - up[1] * z0;
  let xlen = Math.hypot(x0, x1, x2) || 1;
  x0 /= xlen;
  x1 /= xlen;
  x2 /= xlen;
  const y0 = z1 * x2 - z2 * x1;
  const y1 = z2 * x0 - z0 * x2;
  const y2 = z0 * x1 - z1 * x0;
  const out = new Float32Array(16);
  out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
  out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
  out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
  out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
  out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
  out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
  out[15] = 1;
  return out;
}

class OrbitCamera {
  constructor() {
    this.target = [0, 0, 0];
    this.radius = 4;
    this.yaw = 0.6;
    this.pitch = 0.35;
    this.upSign = 1;
    this.fov = (50 * Math.PI) / 180;
    this.near = 0.05;
    this.far = 200;
    this.dampYaw = 0;
    this.dampPitch = 0;
    this.dampPanX = 0;
    this.dampPanY = 0;
    this.dampZoom = 0;
  }

  eye() {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.radius * cp * Math.sin(this.yaw),
      this.target[1] + this.radius * Math.sin(this.pitch) * this.upSign,
      this.target[2] + this.radius * cp * Math.cos(this.yaw),
    ];
  }

  up() {
    return [0, this.upSign, 0];
  }

  fit(bounds) {
    this.target = bounds.center.slice();
    this.radius = Math.max(bounds.radius * 2.4, 0.8);
    this.yaw = 0.55;
    this.pitch = 0.28;
  }

  step() {
    this.yaw += this.dampYaw;
    this.pitch += this.dampPitch;
    this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch));
    const eye = this.eye();
    const view = lookAt(eye, this.target, this.up());
    const right = [view[0], view[4], view[8]];
    const upv = [view[1], view[5], view[9]];
    this.target[0] += right[0] * this.dampPanX + upv[0] * this.dampPanY;
    this.target[1] += right[1] * this.dampPanX + upv[1] * this.dampPanY;
    this.target[2] += right[2] * this.dampPanX + upv[2] * this.dampPanY;
    this.radius = Math.max(0.15, this.radius * Math.exp(this.dampZoom));
    this.dampYaw *= 0.85;
    this.dampPitch *= 0.85;
    this.dampPanX *= 0.85;
    this.dampPanY *= 0.85;
    this.dampZoom *= 0.85;
  }
}

function bindOrbit(canvas, camera) {
  let dragging = false;
  let button = 0;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    button = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    const pan = button === 2 || button === 1 || e.shiftKey;
    if (pan) {
      const scale = camera.radius * 0.0025;
      camera.dampPanX -= dx * scale;
      camera.dampPanY += dy * scale;
    } else {
      camera.dampYaw -= dx * 0.005;
      camera.dampPitch += dy * 0.005 * camera.upSign;
    }
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      camera.dampZoom += Math.sign(e.deltaY) * 0.08;
    },
    { passive: false }
  );
}

async function fetchBuffer(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} loading ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !onProgress) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(total ? received / total : 0, received);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out.buffer;
}

const ui = {
  setStatus(text, kind = "") {
    const el = $("status");
    el.textContent = text;
    el.dataset.kind = kind;
  },
  setProgress(p) {
    $("progress-bar").style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
  },
  setMeta(text) {
    $("meta").textContent = text;
  },
  setFps(fps) {
    $("fps").textContent = `${fps.toFixed(0)} fps`;
  },
  showOverlay(show) {
    $("overlay").classList.toggle("hidden", !show);
  },
};

async function main() {
  const canvas = $("gpu-canvas");
  const overlayUnsupported = $("unsupported");
  const renderer = new WebGPUSplatRenderer(canvas);
  try {
    await renderer.init();
  } catch (err) {
    overlayUnsupported.classList.remove("hidden");
    $("unsupported-msg").textContent = err.message || String(err);
    return;
  }

  const camera = new OrbitCamera();
  bindOrbit(canvas, camera);
  const worker = new Worker(new URL("./parse-worker.js", import.meta.url), {
    type: "module",
  });

  let jobId = 0;
  let pending = null;
  let convertPending = null;
  let last = performance.now();
  let fps = 0;

  const paramsFromUi = () => ({
    pointMode: $("point-mode").checked ? 1 : 0,
    splatScale: Number($("splat-scale").value),
    alpha: Number($("opacity").value),
    brightness: Number($("brightness").value),
    contrast: Number($("contrast").value),
    gamma: Number($("gamma").value),
    intensity: Number($("intensity").value),
    saturation: Number($("saturation").value),
    colorMode: Number($("color-mode").value),
    pixelDiscard: Number($("pixel-discard").value),
    antialias: Number($("antialias").value),
  });

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      renderer.context.configure({
        device: renderer.device,
        format: renderer.format,
        alphaMode: "premultiplied",
      });
    }
  }

  function parseBuffer(buffer, name, compression) {
    const id = ++jobId;
    ui.setStatus(`Parsing ${name}…`);
    ui.showOverlay(true);
    return new Promise((resolve, reject) => {
      pending = { id, resolve, reject };
      worker.postMessage({ id, buffer, name, compression }, [buffer]);
    });
  }

  worker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === "convert") {
      if (!convertPending || msg.id !== convertPending.id) return;
      const { resolve, reject } = convertPending;
      convertPending = null;
      if (!msg.ok) reject(new Error(msg.error));
      else resolve(msg);
      return;
    }
    if (!pending || msg.id !== pending.id) return;
    const { resolve, reject } = pending;
    pending = null;
    if (!msg.ok) {
      reject(new Error(msg.error));
      return;
    }
    resolve(msg);
  };

  async function loadFromBuffer(buffer, name) {
    const compression = Number($("compression").value);
    const result = await parseBuffer(buffer, name, compression);
    renderer.setSplats(result.packed, {
      shDegree: result.shDegree || 0,
      sh1: result.sh1,
    });
    camera.fit(result.bounds);
    const decoder = result.decoder === "gaussforge" ? "GaussForge" : "built-in";
    const sh = result.shDegree ? ` · SH${result.shDegree}` : "";
    ui.setMeta(
      `${name} · ${result.format} · ${decoder}${sh} · ${result.count.toLocaleString()} gaussians`
    );
    const note = result.warning ? `Ready (${result.warning})` : "Ready";
    ui.setStatus(note, "ok");
    ui.setProgress(1);
    ui.showOverlay(false);
  }

  async function exportFormat(outFormat) {
    const id = ++jobId;
    ui.setStatus(`Exporting ${outFormat} via GaussForge…`);
    try {
      const msg = await new Promise((resolve, reject) => {
        convertPending = { id, resolve, reject };
        worker.postMessage({ id, type: "convert", outFormat });
      });
      const blob = new Blob([msg.bytes], { type: "application/octet-stream" });
      const a = document.createElement("a");
      const ext = outFormat === "compressed.ply" ? "compressed.ply" : outFormat;
      a.href = URL.createObjectURL(blob);
      a.download = `export.${ext}`;
      a.click();
      URL.revokeObjectURL(a.href);
      ui.setStatus(`Exported ${outFormat}`, "ok");
    } catch (err) {
      ui.setStatus(err.message, "err");
    }
  }

  async function loadUrl(url, name = url.split("/").pop() || "scene") {
    ui.showOverlay(true);
    ui.setStatus(`Downloading ${name}…`);
    ui.setProgress(0);
    const buffer = await fetchBuffer(url, (p) => {
      ui.setProgress(p * 0.85);
      ui.setStatus(`Downloading ${name}… ${(p * 100).toFixed(0)}%`);
    });
    await loadFromBuffer(buffer, name);
  }

  $("file-input").addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      ui.showOverlay(true);
      ui.setStatus(`Reading ${file.name}…`);
      const buffer = await file.arrayBuffer();
      await loadFromBuffer(buffer, file.name);
    } catch (err) {
      ui.setStatus(err.message, "err");
      ui.showOverlay(false);
    }
  });

  $("load-url").addEventListener("click", async () => {
    const url = $("url-input").value.trim();
    if (!url) return;
    try {
      await loadUrl(url);
    } catch (err) {
      ui.setStatus(err.message, "err");
      ui.showOverlay(false);
    }
  });

  $("load-demo").addEventListener("click", () => loadUrl(DEFAULT_PLY, "demo.ply"));
  $("load-train").addEventListener("click", () =>
    loadUrl(SAMPLE_SPLAT, "train.splat")
  );
  $("load-local").addEventListener("click", () =>
    loadUrl(LOCAL_SPLAT, "test.splat")
  );
  $("y-flip").addEventListener("click", () => {
    camera.upSign *= -1;
    camera.pitch = -camera.pitch;
  });
  $("export-format").addEventListener("change", (e) => {
    const fmt = e.target.value;
    e.target.value = "";
    if (fmt) exportFormat(fmt);
  });

  const drop = document.body;
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    document.body.classList.add("drag");
  });
  drop.addEventListener("dragleave", () => document.body.classList.remove("drag"));
  drop.addEventListener("drop", async (e) => {
    e.preventDefault();
    document.body.classList.remove("drag");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    try {
      ui.showOverlay(true);
      ui.setStatus(`Reading ${file.name}…`);
      await loadFromBuffer(await file.arrayBuffer(), file.name);
    } catch (err) {
      ui.setStatus(err.message, "err");
      ui.showOverlay(false);
    }
  });

  function frame(now) {
    resize();
    camera.step();
    const aspect = canvas.width / canvas.height;
    const proj = perspective(camera.fov, aspect, camera.near, camera.far);
    const view = lookAt(camera.eye(), camera.target, camera.up());
    const fy = canvas.height / (2 * Math.tan(camera.fov / 2));
    renderer.setCamera(proj, view, [fy, fy], [canvas.width, canvas.height], camera.eye());
    renderer.setParams(paramsFromUi());
    renderer.render();
    const dt = now - last;
    last = now;
    fps = fps * 0.9 + (1000 / Math.max(dt, 0.1)) * 0.1;
    ui.setFps(fps);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  const params = new URLSearchParams(location.search);
  const start =
    params.get("url") ||
    params.get("file") ||
    DEFAULT_PLY;
  const startName = start.split("/").pop() || "scene";
  try {
    await loadUrl(start, startName);
  } catch (err) {
    ui.setStatus(`${err.message} — drop a .ply or .splat file`, "err");
    ui.showOverlay(false);
  }
}

main();
