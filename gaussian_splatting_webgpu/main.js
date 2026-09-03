import { WebGPUSplatRenderer } from "./gpu-renderer.js";
import { labelColor } from "../shared/instances.js";
import { makeTwoSpheres } from "../shared/synthetic.js";

const DEMO_PLY = "./demo.ply";
const DEFAULT_SCENE = "../splats/alarm_clock_generated.splat";
const MODEL_SPLAT = "../splats/model.splat";
const SIDECAR_URL = "http://127.0.0.1:8766";
const SAMPLE_SPLAT =
  "https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat";
const LOCAL_SPLAT = "../splat_converter/test.splat";

/** A pointerup this close (CSS px) and this soon after pointerdown is a click, not a drag. */
const CLICK_MAX_PX = 4;
const CLICK_MAX_MS = 300;
/** Tint strength applied by "Teñir" (0 = off). */
const TINT_STRENGTH = 0.6;

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

/** Column-major 4x4 times a [x, y, z, w] vector. */
function transformVec4(m, v) {
  const out = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    out[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
  }
  return out;
}

/** Canvas device pixels per CSS pixel (same clamp as resize()). */
function devicePixelScale() {
  return Math.min(window.devicePixelRatio || 1, 2);
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
    this.far = 500;
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
    this.radius = Math.max(bounds.radius * 3.2, 0.8);
    this.yaw = 0.55;
    this.pitch = 0.28;
    this.far = Math.max(200, bounds.radius * 20);
    this.near = Math.max(0.02, bounds.radius * 0.0004);
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

/** True when the pointerup `e` completes a left-button click started at `down`. */
function isClick(down, e) {
  if (!down || down.button !== 0 || e.button !== 0) return false;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  return moved <= CLICK_MAX_PX && performance.now() - down.t <= CLICK_MAX_MS;
}

/**
 * Orbit / pan / zoom on `canvas`. `onClick(cssX, cssY)` fires for a left-button
 * pointerup within CLICK_MAX_PX / CLICK_MAX_MS of its pointerdown (canvas-relative CSS px).
 */
function bindOrbit(canvas, camera, onClick) {
  let dragging = false;
  let button = 0;
  let lastX = 0;
  let lastY = 0;
  let down = null;
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    button = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = false;
    if (onClick && isClick(down, e)) {
      const rect = canvas.getBoundingClientRect();
      onClick(e.clientX - rect.left, e.clientY - rect.top);
    }
    down = null;
  });
  canvas.addEventListener("pointercancel", () => {
    dragging = false;
    down = null;
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
  setNote(text, kind = "") {
    const el = $("format-note");
    if (!el) return;
    el.textContent = text || "";
    el.hidden = !text;
    el.dataset.kind = kind;
  },
  setFps(fps) {
    $("fps").textContent = `${fps.toFixed(0)} fps`;
  },
  showOverlay(show) {
    $("overlay").classList.toggle("hidden", !show);
  },
};

const formatCount = (n) => n.toLocaleString("es-ES");
const cssColor = (rgb) => `rgb(${rgb.map((v) => Math.round(v * 255)).join(", ")})`;

/**
 * HUD "Instancias": registry of named instances (label → name/count), the
 * current selection, and the per-instance hide / isolate / tint controls.
 * The renderer's instance table is the single source of truth for flags.
 */
class InstancePanel {
  constructor(renderer, elements) {
    this.renderer = renderer;
    this.listEl = elements.list;
    this.statusEl = elements.status;
    /** @type {Map<number, {name: string, count: number}>} */
    this.entries = new Map();
    /** @type {{label: number, name: string, index: number} | null} */
    this.selection = null;
    this.isolateLabel = 0;
    this.listEl.addEventListener("click", (e) => this._onListClick(e));
  }

  /** Forget every instance and clear GPU-side flags (new cloud loaded). */
  reset() {
    this.entries.clear();
    this.selection = null;
    this.isolateLabel = 0;
    this.renderer.resetInstances();
    this.renderer.setParams({ isolateLabel: 0 });
    this.setStatus("Seleccionada: ninguna");
    this.renderList();
  }

  register(label, name, count) {
    if (!Number.isInteger(label) || label <= 0) throw new Error(`etiqueta inválida ${label}`);
    this.entries.set(label, { name: name || `instancia ${label}`, count: count | 0 });
  }

  /** Register every non-zero label present in `labels` with counts (names: label → nombre). */
  fromLabels(labels, names = {}) {
    const counts = new Map();
    for (let i = 0; i < labels.length; i++) {
      const l = labels[i];
      if (l) counts.set(l, (counts.get(l) || 0) + 1);
    }
    for (const label of [...counts.keys()].sort((a, b) => a - b)) {
      this.register(label, names[label], counts.get(label));
    }
    this.renderList();
  }

  nameOf(label) {
    const e = this.entries.get(label);
    return e ? e.name : `instancia ${label}`;
  }

  setStatus(text) {
    this.statusEl.textContent = text;
  }

  /** Apply a renderer.pick() result: select the instance under the pixel. */
  selectHit(hit) {
    if (!hit || hit.index < 0) {
      this.clear();
      return;
    }
    if (hit.label === 0) {
      this.clear();
      this.setStatus(`Seleccionada: ninguna · fondo · gaussiana ${hit.index}`);
      return;
    }
    if (!this.entries.has(hit.label)) {
      // Label from a file without a registry entry: count it on demand.
      const labels = this.renderer.getLabels();
      let n = 0;
      for (let i = 0; i < labels.length; i++) if (labels[i] === hit.label) n++;
      this.register(hit.label, null, n);
    }
    this.select(hit.label, hit.index);
  }

  /** Select instance `label` (0/null clears); `index` is the picked gaussian, -1 if unknown. */
  select(label, index = -1) {
    if (!label) {
      this.clear();
      return;
    }
    if (this.selection && this.selection.label !== label) {
      this.renderer.setInstance(this.selection.label, { selected: false });
    }
    this.renderer.setInstance(label, { selected: true });
    this.selection = { label, name: this.nameOf(label), index };
    const gauss = index >= 0 ? String(index) : "—";
    this.setStatus(`Seleccionada: instancia ${label} (${this.selection.name}) · gaussiana ${gauss}`);
    console.log(`[instancias] seleccionada instancia ${label} (${this.selection.name}) gaussiana ${gauss}`);
    this.renderList();
  }

  clear() {
    if (this.selection) this.renderer.setInstance(this.selection.label, { selected: false });
    this.selection = null;
    this.setStatus("Seleccionada: ninguna");
    this.renderList();
  }

  isolate(label) {
    this.isolateLabel = this.isolateLabel === label ? 0 : label;
    this.renderer.setParams({ isolateLabel: this.isolateLabel });
    this.renderList();
  }

  toggleHidden(label) {
    const visible = this.renderer.getInstance(label).visible;
    this.renderer.setInstance(label, { visible: !visible });
    this.renderList();
  }

  toggleTint(label) {
    const strength = this.renderer.getInstance(label).tint[3];
    const rgb = labelColor(label);
    this.renderer.setInstance(label, { tint: [...rgb, strength > 0 ? 0 : TINT_STRENGTH] });
    this.renderList();
  }

  showAll() {
    this.isolateLabel = 0;
    this.renderer.setParams({ isolateLabel: 0 });
    this.renderer.setInstance(0, { visible: true });
    for (const label of this.entries.keys()) this.renderer.setInstance(label, { visible: true });
    this.renderList();
  }

  /** Plain snapshot of the registry (for tests and the list). */
  rows() {
    return [...this.entries.entries()].map(([label, e]) => {
      const inst = this.renderer.getInstance(label);
      return {
        label,
        name: e.name,
        count: e.count,
        visible: inst.visible,
        selected: inst.selected,
        tinted: inst.tint[3] > 0,
        isolated: this.isolateLabel === label,
        color: labelColor(label),
      };
    });
  }

  renderList() {
    this.listEl.innerHTML = "";
    const rows = this.rows();
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "inst-empty";
      empty.textContent = "Sin instancias. Carga la escena sintética o un fichero con etiquetas.";
      this.listEl.appendChild(empty);
      return;
    }
    for (const row of rows) this.listEl.appendChild(this._rowElement(row));
  }

  _rowElement(row) {
    const el = document.createElement("div");
    el.className = "inst-row";
    el.dataset.label = String(row.label);
    if (row.selected) el.classList.add("selected");
    if (!row.visible) el.classList.add("hidden");
    const button = (act, text, active) =>
      `<button type="button" data-act="${act}"${active ? ' class="active"' : ""}>${text}</button>`;
    el.innerHTML =
      `<span class="inst-swatch" style="background:${cssColor(row.color)}"></span>` +
      `<span class="inst-id">#${row.label}</span>` +
      `<span class="inst-name"></span>` +
      `<span class="inst-count">${formatCount(row.count)} gaussianas</span>` +
      `<div class="inst-btns">` +
      button("isolate", "Aislar", row.isolated) +
      button("hide", row.visible ? "Ocultar" : "Mostrar", !row.visible) +
      button("tint", "Teñir", row.tinted) +
      `</div>`;
    el.querySelector(".inst-name").textContent = row.name;
    return el;
  }

  _onListClick(e) {
    const rowEl = e.target.closest(".inst-row");
    if (!rowEl) return;
    const label = Number(rowEl.dataset.label);
    const act = e.target.closest("button[data-act]");
    if (!act) {
      this.select(label);
      return;
    }
    if (act.dataset.act === "isolate") this.isolate(label);
    else if (act.dataset.act === "hide") this.toggleHidden(label);
    else if (act.dataset.act === "tint") this.toggleTint(label);
  }
}

async function main() {
  const canvas = $("gpu-canvas");
  const overlayUnsupported = $("unsupported");
  const query = new URLSearchParams(location.search);
  // ?offscreen=1: never configure the canvas context (headless SwiftShader tests);
  // pick()/renderOffscreen() still work, render() is a no-op.
  const offscreenOnly = query.get("offscreen") === "1";
  const renderer = new WebGPUSplatRenderer(canvas);
  try {
    await renderer.init({ offscreenOnly });
  } catch (err) {
    overlayUnsupported.classList.remove("hidden");
    $("unsupported-msg").textContent = err.message || String(err);
    return;
  }
  if (offscreenOnly) console.info("[viewer] modo offscreen=1: el canvas no se configura; sólo pick/renderOffscreen");

  const camera = new OrbitCamera();
  const panel = new InstancePanel(renderer, { list: $("inst-list"), status: $("inst-status") });
  bindOrbit(canvas, camera, (x, y) => pickAt(x, y));
  const worker = new Worker(new URL("./parse-worker.js", import.meta.url), {
    type: "module",
  });

  let jobId = 0;
  let pending = null;
  let convertPending = null;
  let last = performance.now();
  let fps = 0;
  let freezeFrame = false;
  let lastSemantic = null;
  /** Matrices used by the last presented frame (for projecting points to pixels). */
  let lastFrame = null;

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
    isolateLabel: panel.isolateLabel,
  });

  function resize() {
    const dpr = devicePixelScale();
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      if (renderer.context) {
        renderer.context.configure({
          device: renderer.device,
          format: renderer.format,
          alphaMode: "premultiplied",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
      }
    }
  }

  function updateCamera() {
    const aspect = canvas.width / canvas.height;
    const proj = perspective(camera.fov, aspect, camera.near, camera.far);
    const view = lookAt(camera.eye(), camera.target, camera.up());
    const fy = canvas.height / (2 * Math.tan(camera.fov / 2));
    renderer.setCamera(proj, view, [fy, fy], [canvas.width, canvas.height], camera.eye());
    lastFrame = { proj, view, width: canvas.width, height: canvas.height };
  }

  /** World point → canvas CSS pixel [x, y] with the last frame's camera (null if behind/unknown). */
  function projectToCss(p) {
    if (!lastFrame) return null;
    const c = transformVec4(lastFrame.proj, transformVec4(lastFrame.view, [p[0], p[1], p[2], 1]));
    if (c[3] <= 0) return null;
    const dpr = devicePixelScale();
    const px = (c[0] / c[3] * 0.5 + 0.5) * lastFrame.width;
    const py = (1 - (c[1] / c[3] * 0.5 + 0.5)) * lastFrame.height;
    return [px / dpr, py / dpr];
  }

  /** Click on the canvas (CSS px): pick the gaussian and select its instance. */
  async function pickAt(cssX, cssY) {
    if (!renderer.count) return null;
    const dpr = devicePixelScale();
    try {
      const hit = await renderer.pick(cssX * dpr, cssY * dpr);
      panel.selectHit(hit);
      return hit;
    } catch (err) {
      ui.setStatus(`Selección fallida: ${err.message}`, "err");
      console.error("[instancias] pick falló", err);
      return null;
    }
  }

  function parseBuffer(buffer, name, compression) {
    const id = ++jobId;
    ui.setStatus(`Parsing ${name}…`);
    ui.setNote("");
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
    renderer.setCloud(result.gaussians, result.sh, result.shDegree || 0);
    panel.reset();
    camera.fit(result.bounds);
    const decoder = result.decoder === "gaussforge" ? "GaussForge" : "built-in";
    const degree = result.shDegree || 0;
    const sh = ` · SH${degree}`;
    ui.setMeta(
      `${name} · ${result.format} · ${decoder}${sh} · ${result.count.toLocaleString()} gaussians`
    );
    const compact =
      result.format === "splat" ||
      result.format === "splat32" ||
      result.format === "splat44";
    if (compact) {
      ui.setNote(
        "Compact .splat is SH0 with 8-bit rotations — a preview, not the trained radiance field. Drop an INRIA point_cloud/iteration_*/point_cloud.ply (with f_rest_* / SH1–3) for full 3DGS.",
        "warn"
      );
    } else if (degree < 3) {
      ui.setNote(
        `This file is spherical-harmonics degree ${degree} (view-independent DC). A trained graphdeco-inria PLY includes f_rest_0…f_rest_44 (SH3) for view-dependent color.`,
        "warn"
      );
    } else {
      ui.setNote(
        "Full 3DGS radiance field: float covariance + SH degree 3 (Kerbl et al.). Leave Point cloud debug unchecked.",
        "ok"
      );
    }
    const note = result.warning ? `Ready (${result.warning})` : "Ready";
    ui.setStatus(note, "ok");
    ui.setProgress(1);
    ui.showOverlay(false);
    window.__gsViewer = {
      name,
      format: result.format,
      decoder: result.decoder,
      shDegree: degree,
      count: result.count,
      compact,
    };
  }

  /** Build the two-sphere scene on the CPU and register its instances (labels 1 and 2). */
  function loadSynthetic(options = {}) {
    ui.setStatus("Generando escena sintética…");
    const scene = makeTwoSpheres(options);
    renderer.setCloud(scene.gaussians, scene.sh, scene.shDegree);
    panel.reset();
    renderer.setLabels(scene.labels);
    panel.fromLabels(scene.labels, scene.names);
    camera.fit(scene.bounds);
    ui.setMeta(`Escena sintética · 2 esferas · SH0 · ${formatCount(scene.count)} gaussianas`);
    ui.setNote(
      "Escena sintética de dos esferas (etiquetas 1 y 2) para probar la selección de instancias: clic en una esfera la selecciona.",
      "ok"
    );
    ui.setStatus("Lista (escena sintética)", "ok");
    ui.setProgress(1);
    ui.showOverlay(false);
    console.log(`[instancias] escena sintética: ${scene.count} gaussianas, instancias ${JSON.stringify(scene.names)}`);
    window.__gsViewer = {
      name: "synthetic-two-spheres",
      format: "synthetic",
      decoder: "synthetic",
      shDegree: 0,
      count: scene.count,
      compact: false,
    };
    return scene;
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

  $("load-demo").addEventListener("click", () => loadUrl(DEMO_PLY, "demo.ply"));
  $("load-alarm").addEventListener("click", () =>
    loadUrl(DEFAULT_SCENE, "alarm_clock_generated.splat")
  );
  $("load-model").addEventListener("click", () =>
    loadUrl(MODEL_SPLAT, "model.splat")
  );
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

  $("syn-two-spheres").addEventListener("click", () => {
    try {
      loadSynthetic();
    } catch (err) {
      ui.setStatus(`Escena sintética fallida: ${err.message}`, "err");
    }
  });
  $("inst-show-all").addEventListener("click", () => panel.showAll());
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") panel.clear();
  });

  window.__gsInstances = {
    select: (label, index = -1) => panel.select(label, index),
    clear: () => panel.clear(),
    labels: () => renderer.getLabels(),
    get current() {
      return panel.selection ? { ...panel.selection } : null;
    },
    get isolateLabel() {
      return panel.isolateLabel;
    },
    list: () => panel.rows(),
    isolate: (label) => panel.isolate(label),
    toggleHidden: (label) => panel.toggleHidden(label),
    toggleTint: (label) => panel.toggleTint(label),
    showAll: () => panel.showAll(),
    loadSynthetic: (options) => loadSynthetic(options),
    pickAt: (cssX, cssY) => pickAt(cssX, cssY),
    project: (p) => projectToCss(p),
  };

  async function blobToB64(blob) {
    const buf = await blob.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(bin);
  }

  function drawBoxes(objects) {
    const overlay = $("sem-overlay");
    if (!overlay) return;
    overlay.innerHTML = "";
    if (!objects || !objects.length) return;
    for (const obj of objects) {
      const box = obj.best_box || (obj.views && obj.views[0] && obj.views[0].box);
      if (!box) continue;
      const el = document.createElement("div");
      el.className = "sem-box";
      el.style.left = `${box[0] * 100}%`;
      el.style.top = `${box[1] * 100}%`;
      el.style.width = `${box[2] * 100}%`;
      el.style.height = `${box[3] * 100}%`;
      el.innerHTML = `<span>${obj.name}</span>`;
      overlay.appendChild(el);
    }
  }

  function renderSemanticPanel(result) {
    const list = $("sem-list");
    const cards = $("sem-cards");
    if (!list) return;
    lastSemantic = result;
    list.innerHTML = "";
    cards.innerHTML = "";
    const objects = result.objects || [];
    $("sem-meta").textContent = `${objects.length} objects · ${result.view_count} views · ${result.vision_model}${
      result.imagine_model ? " · " + result.imagine_model : ""
    }`;
    for (const obj of objects) {
      const li = document.createElement("li");
      li.textContent = `${obj.name} (${(obj.confidence * 100).toFixed(0)}%)`;
      list.appendChild(li);
    }
    for (const card of result.cards || []) {
      const fig = document.createElement("figure");
      if (card.error) {
        fig.innerHTML = `<figcaption>${card.name}: ${card.error}</figcaption>`;
      } else {
        const src = card.b64
          ? `data:${card.mime || "image/jpeg"};base64,${card.b64}`
          : card.url;
        const saved = card.path ? `<br><code>${card.path}</code>` : "";
        fig.innerHTML = `<img alt="${card.name}" src="${src}" /><figcaption>${card.name}${saved}</figcaption>`;
      }
      cards.appendChild(fig);
    }
    drawBoxes(objects);
    window.__gsSemantic = result;
  }

  async function captureViews(n) {
    freezeFrame = true;
    const saved = { yaw: camera.yaw, pitch: camera.pitch, radius: camera.radius };
    const views = [];
    const count = Math.max(1, Math.min(8, n | 0));
    try {
      for (let i = 0; i < count; i++) {
        if (count > 1) camera.yaw = saved.yaw + (i * 2 * Math.PI) / count;
        camera.dampYaw = 0;
        camera.dampPitch = 0;
        camera.dampPanX = 0;
        camera.dampPanY = 0;
        camera.dampZoom = 0;
        resize();
        updateCamera();
        renderer.setParams(paramsFromUi());
        renderer.render();
        const blob = await renderer.snapshotPng(1024);
        views.push({
          png_b64: await blobToB64(blob),
          yaw: camera.yaw,
          pitch: camera.pitch,
          eye: camera.eye(),
        });
        ui.setStatus(`Captured view ${i + 1}/${count}`);
      }
    } finally {
      camera.yaw = saved.yaw;
      camera.pitch = saved.pitch;
      camera.radius = saved.radius;
      freezeFrame = false;
    }
    return views;
  }

  async function analyzeScene() {
    if (!renderer.count) {
      ui.setStatus("Load a scene first", "err");
      return;
    }
    const n = Number($("sem-views").value) || 1;
    const makeCards = $("sem-cards-toggle").checked;
    ui.setStatus("Capturing 3DGS views…");
    ui.showOverlay(true);
    try {
      const health = await fetch(`${SIDECAR_URL}/health`);
      if (!health.ok) throw new Error("sidecar HTTP " + health.status);
      const h = await health.json();
      if (!h.xai) throw new Error("sidecar has no XAI_API_KEY");
      const views = await captureViews(n);
      ui.setStatus(`Tagging ${views.length} view(s) with Grok…`);
      const res = await fetch(`${SIDECAR_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          views,
          make_cards: makeCards,
          max_cards: Number($("sem-max-cards").value) || 3,
        }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || `analyze ${res.status}`);
      renderSemanticPanel(data);
      ui.setStatus(
        `Tagged ${data.objects.length} object(s)` +
          (makeCards ? ` · ${data.cards.length} Imagine card(s)` : ""),
        "ok"
      );
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      const hint = /fetch|Failed|ECONNREFUSED|NetworkError/i.test(msg)
        ? " — start ./semantic_sidecar/launch.sh (port 8766)"
        : "";
      ui.setStatus(msg + hint, "err");
    } finally {
      ui.showOverlay(false);
    }
  }

  $("sem-analyze").addEventListener("click", () => analyzeScene());
  window.__gsAnalyze = analyzeScene;
  window.__gsCamera = camera;
  window.__gsRenderer = renderer;

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
    if (!freezeFrame) {
      resize();
      camera.step();
      updateCamera();
      renderer.setParams(paramsFromUi());
      renderer.render();
    }
    const dt = now - last;
    last = now;
    fps = fps * 0.9 + (1000 / Math.max(dt, 0.1)) * 0.1;
    ui.setFps(fps);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // ?scene=synthetic loads the two-sphere scene instead of a file.
  if (query.get("scene") === "synthetic") {
    try {
      loadSynthetic();
    } catch (err) {
      ui.setStatus(`Escena sintética fallida: ${err.message}`, "err");
    }
    return;
  }
  const start =
    query.get("url") ||
    query.get("file") ||
    DEFAULT_SCENE;
  const startName = start.split("/").pop() || "scene";
  try {
    await loadUrl(start, startName);
  } catch (err) {
    ui.setStatus(`${err.message} — drop a .ply or .splat file`, "err");
    ui.showOverlay(false);
  }
}

main();
