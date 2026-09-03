import { WebGPUSplatRenderer, MAX_INSTANCES, OUTPUT_MODE } from "./gpu-renderer.js";
import { DEFAULT_LIFT_OPTIONS, buildInstancesJson, labelsToBytes, liftViews } from "../shared/lift.js";
import { applyNames, frameBounds, instanceBounds, searchInstances } from "../shared/naming.js";
import { diffuseLabels, groupColor, indicesOfGroup, shDcToRgb } from "../shared/graph.js";
import { labelColor } from "../shared/instances.js";
import { makeTwoSpheres } from "../shared/synthetic.js";
import { EditLog, bakeSession, composeTransform, mat4Multiply, mat4Translation, opsFromJsonl, rangesFromIndices, replay, sessionFingerprint, transformPoint } from "../shared/edit-ops.js";
import { encodePly, encodeSplat32, exportFileName } from "../shared/export-io.js";
import { orbitCameras } from "../shared/tsdf.js";
import { encodeGlb } from "../shared/glb.js";

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
function bindOrbit(canvas, camera, onClick, tools = null) {
  let dragging = false;
  let button = 0;
  let lastX = 0;
  let lastY = 0;
  let down = null;
  /** F5 drag tools (rectángulo / pincel) take the left button; orbit keeps the rest. */
  let toolDrag = null;
  const cssPoint = (e) => {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    const tool = tools ? tools.active() : "clic";
    if (e.button === 0 && !e.shiftKey && (tool === "rect" || tool === "pincel")) {
      const [x, y] = cssPoint(e);
      toolDrag = { tool, x0: x, y0: y, cx: e.clientX, cy: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      if (tool === "pincel") tools.onBrush(x, y);
      else tools.showRect(e.clientX, e.clientY, e.clientX, e.clientY);
      return;
    }
    dragging = true;
    button = e.button;
    lastX = e.clientX;
    lastY = e.clientY;
    down = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    if (toolDrag) {
      const [x, y] = cssPoint(e);
      if (toolDrag.tool === "rect") {
        tools.hideRect();
        tools.onRect(toolDrag.x0, toolDrag.y0, x, y);
      } else tools.onBrushEnd();
      toolDrag = null;
      return;
    }
    dragging = false;
    if (onClick && isClick(down, e)) {
      const [x, y] = cssPoint(e);
      onClick(x, y);
    }
    down = null;
  });
  canvas.addEventListener("pointercancel", () => {
    if (toolDrag && toolDrag.tool === "rect") tools.hideRect();
    toolDrag = null;
    dragging = false;
    down = null;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (toolDrag) {
      if (toolDrag.tool === "rect") tools.showRect(toolDrag.cx, toolDrag.cy, e.clientX, e.clientY);
      else {
        const [x, y] = cssPoint(e);
        tools.onBrush(x, y);
      }
      return;
    }
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
    /** Row actions handled outside the panel (F4: "name", "card"). */
    this.onAction = null;
    /** Labels highlighted by the text search (null = no filter). */
    this.matches = null;
    this.listEl.addEventListener("click", (e) => this._onListClick(e));
  }

  /** Forget every instance and clear GPU-side flags (new cloud loaded). */
  reset() {
    this.entries.clear();
    this.selection = null;
    this.isolateLabel = 0;
    this.matches = null;
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

  /** Recount gaussians per registered label (e.g. after a label diffusion). */
  refreshCounts(labels) {
    const counts = new Map();
    for (let i = 0; i < labels.length; i++) {
      const l = labels[i];
      if (l) counts.set(l, (counts.get(l) || 0) + 1);
    }
    for (const [label, e] of this.entries) e.count = counts.get(label) || 0;
    this.renderList();
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
        nombre: e.nombre || "",
        nombre_es: e.nombre_es || "",
        categoria: e.categoria || "",
        confianza: Number.isFinite(e.confianza) ? e.confianza : null,
        descripcion_es: e.descripcion_es || "",
        error: e.error || "",
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
    if (this.matches) el.classList.add(this.matches.has(row.label) ? "match" : "dim");
    const button = (act, text, active) =>
      `<button type="button" data-act="${act}"${active ? ' class="active"' : ""}>${text}</button>`;
    el.innerHTML =
      `<span class="inst-swatch" style="background:${cssColor(row.color)}"></span>` +
      `<span class="inst-id">#${row.label}</span>` +
      `<span class="inst-name"></span>` +
      `<span class="inst-count">${formatCount(row.count)} gaussianas</span>` +
      (row.categoria || row.error ? `<span class="inst-meta"></span>` : "") +
      `<div class="inst-btns">` +
      button("isolate", "Aislar", row.isolated) +
      button("hide", row.visible ? "Ocultar" : "Mostrar", !row.visible) +
      button("tint", "Teñir", row.tinted) +
      button("name", "Nombrar", false) +
      button("card", "Tarjeta", false) +
      button("mesh", "Malla", false) +
      `</div>`;
    el.querySelector(".inst-name").textContent = row.name;
    const meta = el.querySelector(".inst-meta");
    if (meta) {
      meta.textContent = row.error
        ? `sin nombre: ${row.error}`
        : `${row.categoria}${row.confianza != null ? ` · ${Math.round(row.confianza * 100)} %` : ""}${row.nombre && row.nombre !== row.name ? ` · ${row.nombre}` : ""}`;
    }
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
    else if (this.onAction) this.onAction(act.dataset.act, label);
  }

  /** Highlight the rows matching `query` (empty = clear); returns the matches best first. */
  search(query) {
    const q = String(query || "").trim();
    if (!q) {
      this.matches = null;
      this.renderList();
      return [];
    }
    const records = [...this.entries.entries()].map(([label, e]) => ({ label, ...e }));
    const found = searchInstances(records, q);
    this.matches = new Set(found.map((r) => r.label));
    this.renderList();
    return found;
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
  bindOrbit(canvas, camera, (x, y) => pickAt(x, y), {
    active: () => editEl.tool.value,
    showRect: (x0, y0, x1, y1) => showSelectionRect(x0, y0, x1, y1),
    hideRect: () => hideSelectionRect(),
    onRect: (x0, y0, x1, y1) => selectRect(x0, y0, x1, y1).catch((err) => setEditStatus(`Selección fallida: ${err.message}`, "err")),
    onBrush: (x, y) => brushAt(x, y),
    onBrushEnd: () => brushEnd(),
  });
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
  /** CPU copy of the loaded cloud (input for the F2 graph worker). */
  let cloud = null; // { gaussians, sh, shDegree, count }
  /** F2 superpoint graph state (see shared/graph.js / graph-worker.js). */
  const groups = {
    result: null,
    worker: null,
    pending: null,
    jobId: 0,
    computing: false,
    prevColorMode: "0",
    /** superpoint id (1-based) → F1 instance label created by a click */
    instanceOfGroup: new Map(),
  };

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
      // F5 tools that select from one click: esfera 3D / superpunto.
      const tool = editEl.tool.value;
      if (tool === "esfera" || tool === "grupo") {
        await selectFromHit(tool, hit);
        return hit;
      }
      // Vista Grupos: a click turns the superpoint under the cursor into an instance.
      if (isGroupView() && groups.result && hit.index >= 0 && hit.group > 0) {
        promoteGroup(hit.group, hit.index);
        return hit;
      }
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

  // ------------------------------------------------------------ Grupos (F2)

  const grpEl = {
    status: $("grp-status"),
    compute: $("grp-compute"),
    view: $("grp-view"),
    diffuse: $("grp-diffuse"),
    k: $("grp-k"),
    threshold: $("grp-threshold"),
    sigma: $("grp-sigma"),
  };

  function setGroupStatus(text, kind = "") {
    grpEl.status.textContent = text;
    grpEl.status.dataset.kind = kind;
  }

  function isGroupView() {
    return Number($("color-mode").value) === 4;
  }

  /** Switch the Color selector to/from «Grupos» (mode 4), remembering the previous mode. */
  function setGroupView(on) {
    const sel = $("color-mode");
    if (on) {
      if (sel.value !== "4") groups.prevColorMode = sel.value;
      sel.value = "4";
    } else if (sel.value === "4") {
      sel.value = groups.prevColorMode || "0";
    }
    grpEl.view.classList.toggle("active", isGroupView());
  }

  /** Forget the graph of the previous cloud. */
  function resetGroups() {
    groups.result = null;
    groups.instanceOfGroup.clear();
    if (renderer.count) renderer.setGroups(null);
    grpEl.diffuse.disabled = true;
    setGroupStatus("Sin grupos. Calcula el grafo para colorear superpuntos.");
    if (isGroupView()) setGroupView(false);
  }

  function graphOptionsFromUi() {
    return {
      k: Number(grpEl.k.value) || 10,
      threshold: Number(grpEl.threshold.value),
      sigmaColor: Number(grpEl.sigma.value),
    };
  }

  function getGraphWorker() {
    if (groups.worker) return groups.worker;
    const w = new Worker(new URL("../shared/graph-worker.js", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      const msg = e.data;
      if (!groups.pending || msg.id !== groups.pending.id) return;
      const { resolve, reject } = groups.pending;
      groups.pending = null;
      if (msg.ok) resolve(msg.result);
      else reject(new Error(msg.error));
    };
    w.onerror = (e) => {
      if (!groups.pending) return;
      groups.pending.reject(new Error(e.message || "error en graph-worker"));
      groups.pending = null;
    };
    groups.worker = w;
    return w;
  }

  function summarizeGroups() {
    const r = groups.result;
    if (!r) return null;
    return {
      count: r.count,
      superpointCount: r.superpointCount,
      sizes: Array.from(r.sizes.subarray(0, 32)),
      k: r.k,
      cellSize: r.cellSize,
      threshold: r.threshold,
      sigmaColor: r.sigmaColor,
      stats: { ...r.stats },
    };
  }

  function applyGroups(result, ms) {
    groups.result = result;
    groups.instanceOfGroup.clear();
    const ids = new Uint32Array(result.count);
    for (let i = 0; i < result.count; i++) ids[i] = result.superpoint[i] + 1;
    renderer.setGroups(ids);
    grpEl.diffuse.disabled = false;
    const sizes = result.sizes;
    const median = sizes.length ? sizes[sizes.length >> 1] : 0;
    setGroupStatus(
      `${formatCount(result.superpointCount)} grupos · mayor ${formatCount(sizes[0] || 0)} · mediana ${formatCount(median)} gaussianas · ` +
        `grado medio ${result.stats.avgDegree.toFixed(1)} · ${ms.toFixed(0)} ms`,
      "ok"
    );
    console.info(`[grupos] ${result.superpointCount} superpuntos de ${result.count} gaussianas en ${ms.toFixed(0)} ms`, result.stats);
    setGroupView(true);
  }

  /** Build the superpoint graph of the current cloud in the worker and colour the view. */
  async function computeGroups(overrides = {}) {
    if (!cloud || !cloud.count) {
      ui.setStatus("Carga una escena antes de calcular grupos", "err");
      return null;
    }
    if (groups.computing) throw new Error("ya se está calculando el grafo");
    const options = { ...graphOptionsFromUi(), ...overrides };
    groups.computing = true;
    grpEl.compute.disabled = true;
    setGroupStatus(`Calculando grafo kNN (k=${options.k}) de ${formatCount(cloud.count)} gaussianas…`);
    const t0 = performance.now();
    const generation = cloud;
    try {
      const worker = getGraphWorker();
      const id = ++groups.jobId;
      const colors = shDcToRgb(cloud.sh, cloud.count);
      const gaussians = cloud.gaussians.slice(); // the worker takes ownership of this copy
      const result = await new Promise((resolve, reject) => {
        groups.pending = { id, resolve, reject };
        worker.postMessage({ id, type: "build", gaussians, colors, options }, [gaussians.buffer, colors.buffer]);
      });
      if (cloud !== generation) throw new Error("la escena cambió durante el cálculo");
      applyGroups(result, performance.now() - t0);
      return summarizeGroups();
    } catch (err) {
      setGroupStatus(`Grafo fallido: ${err.message}`, "err");
      console.error("[grupos]", err);
      throw err;
    } finally {
      groups.computing = false;
      grpEl.compute.disabled = false;
    }
  }

  /** Turn superpoint `group` (1-based id) into an F1 instance and select it; returns the label (0 = failed). */
  function promoteGroup(group, index = -1) {
    if (!groups.result || !Number.isInteger(group) || group <= 0 || group > groups.result.superpointCount) return 0;
    let label = groups.instanceOfGroup.get(group);
    if (!label) {
      const labels = renderer.getLabels();
      let max = 0;
      for (let i = 0; i < labels.length; i++) if (labels[i] > max) max = labels[i];
      label = max + 1;
      if (label >= MAX_INSTANCES) {
        ui.setStatus(`Sin espacio para más instancias (máximo ${MAX_INSTANCES - 1})`, "err");
        return 0;
      }
      const indices = indicesOfGroup(groups.result.superpoint, group - 1);
      resetEdit();
      renderer.setLabel(indices, label);
      renderer.setInstance(label, { tint: [...groupColor(group), 0] });
      panel.register(label, `grupo ${group}`, indices.length);
      groups.instanceOfGroup.set(group, label);
      console.info(`[grupos] grupo ${group} → instancia ${label} (${indices.length} gaussianas)`);
    }
    panel.select(label, index);
    return label;
  }

  /** Weighted-majority diffusion of the current instance labels over the graph. */
  function diffuseInstanceLabels(iterations = 5) {
    const r = groups.result;
    if (!r) {
      ui.setStatus("Calcula los grupos antes de difundir etiquetas", "err");
      return null;
    }
    resetEdit();
    const before = renderer.getLabels();
    const after = diffuseLabels(before, r.csr, r.csr.weights, { iterations });
    let changed = 0;
    for (let i = 0; i < after.length; i++) if (after[i] !== before[i]) changed++;
    renderer.setLabels(after);
    panel.refreshCounts(after);
    ui.setStatus(`Difusión de etiquetas: ${formatCount(changed)} gaussianas cambiadas (${iterations} iteraciones)`, "ok");
    return changed;
  }

  grpEl.compute.addEventListener("click", () => computeGroups().catch(() => {}));
  grpEl.view.addEventListener("click", () => setGroupView(!isGroupView()));
  grpEl.diffuse.addEventListener("click", () => diffuseInstanceLabels());
  $("color-mode").addEventListener("change", () => grpEl.view.classList.toggle("active", isGroupView()));

  window.__gsGroups = {
    compute: (options) => computeGroups(options),
    get result() {
      return summarizeGroups();
    },
    get view() {
      return isGroupView();
    },
    setView: (on) => setGroupView(on),
    groupOf: (index) => renderer.groupOf(index),
    superpointOf: (index) => (groups.result ? groups.result.superpoint[index] : null),
    promote: (group, index = -1) => promoteGroup(group, index),
    diffuse: (iterations) => diffuseInstanceLabels(iterations),
    groupColor: (group) => groupColor(group),
  };

  // ------------------------------------------------------ Segmentación (F3)

  const segEl = {
    views: $("seg-views"),
    bias: $("seg-bias"),
    source: $("seg-source"),
    lift: $("seg-lift"),
    export: $("seg-export"),
    status: $("seg-status"),
  };
  /** Last lift: labels applied, names, per-view metadata (for instancias.json). */
  const seg = { last: null, running: false };
  /** In-browser models (ml-browser.js), loaded on demand. */
  const ml = { module: null, sam: null, clip: null, samPrompts: 12 };
  const SEG_K = 24;
  const SEG_MAX_EDGE = 512;
  const SEG_PITCH = 0.45;

  function setSegStatus(text, kind = "") {
    segEl.status.textContent = text;
    segEl.status.dataset.kind = kind;
  }

  /** Mask resolution: the canvas aspect at ≤ SEG_MAX_EDGE px on the long edge. */
  function maskSize() {
    const w = canvas.width || 512;
    const h = canvas.height || 384;
    const scale = Math.min(1, SEG_MAX_EDGE / Math.max(w, h));
    return [Math.max(64, Math.round(w * scale)), Math.max(64, Math.round(h * scale))];
  }

  /** Push the orbit camera to the renderer (same maths as the frame loop). */
  function pushCamera() {
    resize();
    const aspect = canvas.width / canvas.height;
    const proj = perspective(camera.fov, aspect, camera.near, camera.far);
    const view = lookAt(camera.eye(), camera.target, camera.up());
    const fy = canvas.height / (2 * Math.tan(camera.fov / 2));
    renderer.setCamera(proj, view, [fy, fy], [canvas.width, canvas.height], camera.eye());
    lastFrame = { proj, view, width: canvas.width, height: canvas.height };
  }

  /** View v of n: yaw around the target, pitch alternating ±SEG_PITCH so poles get covered. */
  function applyOrbitView(v, n, saved) {
    camera.yaw = saved.yaw + (v * 2 * Math.PI) / n;
    camera.pitch = n > 1 ? (v % 2 ? -SEG_PITCH : SEG_PITCH) : saved.pitch;
    camera.dampYaw = camera.dampPitch = camera.dampPanX = camera.dampPanY = camera.dampZoom = 0;
    pushCamera();
  }

  /** Test source: the current labels seen through the ID pass, with a per-view cyclic id shift. */
  async function maskFromCurrentLabels(W, H, shift) {
    const labels = renderer.getLabels();
    let maxLabel = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] > maxLabel) maxLabel = labels[i];
    if (!maxLabel) throw new Error("la fuente «prueba» necesita instancias etiquetadas (escena sintética o grupos promovidos)");
    const perm = (l) => (l ? ((l - 1 + shift) % maxLabel) + 1 : 0);
    const id = await renderer.renderOffscreen({ mode: OUTPUT_MODE.ID, width: W, height: H });
    const mask = new Uint32Array(W * H);
    for (let p = 0; p < mask.length; p++) {
      const g = id.data[p];
      if (g) mask[p] = perm(labels[g - 1]);
    }
    const names = [];
    for (let l = 1; l <= maxLabel; l++) names[perm(l)] = panel.nameOf(l);
    return { mask, labelCount: maxLabel + 1, names };
  }

  async function rgbaToPngB64(rgba, W, H) {
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(W, H);
    img.data.set(rgba);
    ctx.putImageData(img, 0, 0);
    const blob = await new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encode failed"))), "image/png"));
    return blobToB64(blob);
  }

  /** 8-bit label PNG from the sidecar → Uint32Array mask. */
  async function decodeMaskPng(b64, W, H) {
    const blob = await (await fetch(`data:image/png;base64,${b64}`)).blob();
    const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    if (bmp.width !== W || bmp.height !== H) throw new Error(`máscara ${bmp.width}x${bmp.height} no coincide con la vista ${W}x${H}`);
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    const d = ctx.getImageData(0, 0, W, H).data;
    const mask = new Uint32Array(W * H);
    let max = 0;
    for (let p = 0; p < mask.length; p++) {
      const v = d[p * 4];
      mask[p] = v;
      if (v > max) max = v;
    }
    return { mask, labelCount: max + 1 };
  }

  /** Capture n orbit views as PNG and ask the sidecar for label masks. */
  async function masksFromSidecar(n, W, H, saved, backend) {
    const captures = [];
    for (let v = 0; v < n; v++) {
      applyOrbitView(v, n, saved);
      setSegStatus(`Capturando vista ${v + 1}/${n} para el sidecar…`);
      const col = await renderer.renderOffscreen({ mode: OUTPUT_MODE.COLOR, width: W, height: H, clearColor: [0, 0, 0, 1] });
      captures.push({ png_b64: await rgbaToPngB64(col.data, W, H), width: W, height: H });
    }
    const health = await fetch(`${SIDECAR_URL}/health`);
    if (!health.ok) throw new Error(`sidecar HTTP ${health.status}`);
    setSegStatus(`Pidiendo máscaras (${backend}) al sidecar para ${n} vistas…`);
    const res = await fetch(`${SIDECAR_URL}/segment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ views: captures, backend }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `segment ${res.status}`);
    const out = [];
    for (let v = 0; v < n; v++) {
      const view = data.views[v];
      const m = await decodeMaskPng(view.mask_png_b64, W, H);
      m.names = [];
      for (const o of view.objects || []) m.names[o.id] = o.name;
      out.push(m);
    }
    return out;
  }

  async function loadMl() {
    if (!ml.module) ml.module = await import("./ml-browser.js");
    return ml.module;
  }

  async function loadBrowserSam() {
    if (ml.sam) return ml.sam;
    const mod = await loadMl();
    setSegStatus("Cargando SAM 2 (transformers.js)…");
    ml.sam = await mod.BrowserSam.load({
      progress: (p) => {
        if (p.status === "progress" && p.file) setSegStatus(`Cargando SAM 2: ${p.file} ${Math.round(p.progress || 0)} %`);
      },
    });
    console.info(`[ml] SAM listo: ${ml.sam.meta.id} · ${ml.sam.meta.device} · ${ml.sam.meta.dtype}`);
    return ml.sam;
  }

  async function loadBrowserClip() {
    if (ml.clip) return ml.clip;
    const mod = await loadMl();
    setNameStatus("Cargando CLIP (transformers.js)…");
    ml.clip = await mod.BrowserClip.load({
      progress: (p) => {
        if (p.status === "progress" && p.file) setNameStatus(`Cargando CLIP: ${p.file} ${Math.round(p.progress || 0)} %`);
      },
    });
    console.info(`[ml] CLIP listo: ${ml.clip.meta.id} · ${ml.clip.meta.device} · ${ml.clip.meta.dtype}`);
    return ml.clip;
  }

  /** Project a world point with the renderer's current camera to pixel coords of a W×H target. */
  function projectToTarget(p, W, H) {
    const cam = renderer.camera;
    const v = transformVec4(cam.view, [p[0], p[1], p[2], 1]);
    const c = transformVec4(cam.projection, v);
    if (c[3] <= 0) return null;
    const x = (c[0] / c[3] * 0.5 + 0.5) * W;
    const y = (1 - (c[1] / c[3] * 0.5 + 0.5)) * H;
    if (x < 0 || y < 0 || x >= W || y >= H) return null;
    return [x, y];
  }

  /**
   * Point prompts for SAM from the F2 superpoints: the largest superpoints
   * whose centroid projects inside the view and is actually visible there
   * (the ID pass hits a gaussian of that superpoint at the pixel).
   */
  async function samPromptsForView(W, H, maxPrompts) {
    const g = groups.result;
    if (!g) throw new Error("SAM en el navegador necesita los superpuntos (F2)");
    const id = await renderer.renderOffscreen({ mode: OUTPUT_MODE.ID, width: W, height: H });
    const prompts = [];
    const order = [...g.sizes.keys()].sort((a, b) => g.sizes[b] - g.sizes[a]);
    for (const sp of order) {
      if (prompts.length >= maxPrompts) break;
      if (g.sizes[sp] < 20) break;
      const c = [g.centroids[sp * 3], g.centroids[sp * 3 + 1], g.centroids[sp * 3 + 2]];
      const px = projectToTarget(c, W, H);
      if (!px) continue;
      const hit = id.data[Math.floor(px[1]) * W + Math.floor(px[0])];
      if (!hit || g.superpoint[hit - 1] !== sp) continue;
      prompts.push({ points: [[px[0], px[1]]], superpoint: sp });
    }
    return prompts;
  }

  /** Capture n orbit views and segment each one with SAM 2 in the browser. */
  async function masksFromBrowserSam(n, W, H, saved, maxPrompts) {
    const sam = await loadBrowserSam();
    const out = [];
    for (let v = 0; v < n; v++) {
      applyOrbitView(v, n, saved);
      const prompts = await samPromptsForView(W, H, maxPrompts);
      setSegStatus(`SAM 2 vista ${v + 1}/${n}: codificando ${W}×${H} (${prompts.length} indicaciones)…`);
      const col = await renderer.renderOffscreen({ mode: OUTPUT_MODE.COLOR, width: W, height: H, clearColor: [0, 0, 0, 1] });
      const msEncode = await sam.setImage(col.data, W, H);
      const lm = await sam.labelMask(prompts);
      console.info(`[ml] SAM vista ${v + 1}: ${lm.objects.length} máscaras de ${prompts.length} indicaciones (${lm.duplicates} duplicadas) · codificación ${msEncode.toFixed(0)} ms`);
      out.push({ mask: lm.labels, labelCount: lm.labelCount, names: [], sam: { prompts: prompts.length, duplicates: lm.duplicates, objects: lm.objects, msEncode } });
    }
    return out;
  }

  /**
   * Lift 2D masks from n orbit views to per-gaussian instance labels:
   * K-buffer contributions → FlashSplat → superpoint association → diffusion.
   */
  async function liftMasks(options = {}) {
    if (!cloud || !cloud.count) {
      ui.setStatus("Carga una escena antes de levantar máscaras", "err");
      return null;
    }
    if (seg.running) throw new Error("ya hay un levantamiento en curso");
    resetEdit();
    const source = options.source || segEl.source.value;
    const n = Math.max(1, Math.min(12, Number(options.views ?? segEl.views.value) || 1));
    const bias = options.backgroundBias ?? Number(segEl.bias.value);
    const iterations = options.diffusion ?? 5;
    const iouThreshold = options.iouThreshold ?? DEFAULT_LIFT_OPTIONS.iouThreshold;
    const gaussianThreshold = options.gaussianThreshold ?? DEFAULT_LIFT_OPTIONS.gaussianThreshold;
    seg.running = true;
    segEl.lift.disabled = true;
    freezeFrame = true;
    const saved = { yaw: camera.yaw, pitch: camera.pitch, radius: camera.radius };
    const t0 = performance.now();
    try {
      if (!groups.result) {
        setSegStatus("Calculando superpuntos (F2) para asociar las vistas…");
        await computeGroups();
        setGroupView(false);
      }
      const [W, H] = maskSize();
      let sidecarMasks = null;
      if (source === "sam2") {
        sidecarMasks = await masksFromBrowserSam(n, W, H, saved, options.samPrompts || ml.samPrompts);
      } else if (source !== "prueba") {
        sidecarMasks = await masksFromSidecar(n, W, H, saved, source === "sidecar-sam" ? "sam" : "grok-boxes");
      }
      const views = [];
      const viewMeta = [];
      for (let v = 0; v < n; v++) {
        applyOrbitView(v, n, saved);
        const m = sidecarMasks ? sidecarMasks[v] : await maskFromCurrentLabels(W, H, v);
        setSegStatus(`Levantando vista ${v + 1}/${n} (${m.labelCount - 1} máscaras, ${W}×${H})…`);
        const c = await renderer.renderContributions({ mask: m.mask, width: W, height: H, labelCount: m.labelCount, k: SEG_K });
        views.push({ contrib: c.contrib, labelCount: m.labelCount, names: m.names || [] });
        viewMeta.push({ indice: v, yaw: camera.yaw, pitch: camera.pitch, eye: camera.eye(), mascaras: m.labelCount - 1, chunks: c.chunks, splits: c.splits, instancias: [], sam: m.sam || null });
      }
      setSegStatus("Asignando etiquetas (FlashSplat) y asociando vistas…");
      const lift = liftViews(views, {
        count: cloud.count,
        superpoint: groups.result.superpoint,
        backgroundBias: bias,
        iouThreshold,
        gaussianThreshold,
      });
      lift.association.members.forEach((list, k) => {
        for (const [vi] of list) if (!viewMeta[vi].instancias.includes(k + 1)) viewMeta[vi].instancias.push(k + 1);
      });
      let labels = lift.labels;
      let changed = 0;
      if (iterations > 0) {
        const d = diffuseLabels(labels, groups.result.csr, groups.result.csr.weights, { iterations });
        for (let i = 0; i < d.length; i++) if (d[i] !== labels[i]) changed++;
        labels = d;
      }
      panel.reset();
      renderer.setLabels(labels);
      const names = {};
      lift.names.forEach((nm, g) => {
        if (g > 0) names[g] = nm;
      });
      panel.fromLabels(labels, names);
      const ms = performance.now() - t0;
      seg.last = {
        source,
        views: viewMeta,
        bias,
        iterations,
        changed,
        k: SEG_K,
        width: W,
        height: H,
        ms,
        globalCount: lift.globalCount,
        names: lift.names,
        merges: lift.association.pairs.length,
        association: lift.association.strategy,
        iouThreshold,
        gaussianThreshold,
      };
      segEl.export.disabled = false;
      setSegStatus(
        `${formatCount(lift.globalCount)} instancias · ${n} vistas · ${lift.association.pairs.length} fusiones · ` +
          `${formatCount(changed)} etiquetas corregidas por difusión · ${ms.toFixed(0)} ms`,
        "ok"
      );
      console.info(`[segmentación] ${lift.globalCount} instancias desde ${n} vistas en ${ms.toFixed(0)} ms`, seg.last);
      return summarizeSeg();
    } catch (err) {
      setSegStatus(`Levantamiento fallido: ${err.message}`, "err");
      console.error("[segmentación]", err);
      throw err;
    } finally {
      camera.yaw = saved.yaw;
      camera.pitch = saved.pitch;
      camera.radius = saved.radius;
      pushCamera();
      freezeFrame = false;
      seg.running = false;
      segEl.lift.disabled = false;
    }
  }

  function summarizeSeg() {
    const l = seg.last;
    if (!l) return null;
    return { ...l, names: l.names.slice(), views: l.views.map((v) => ({ ...v })) };
  }

  /** instancias.json + etiquetas.u32 for the current labels (plan §3.3). */
  function buildSegmentationExport() {
    if (!panel.entries.size) throw new Error("no hay instancias que exportar");
    // Without a lift (e.g. promoted groups or the synthetic scene) the method is "manual".
    const l = seg.last || { source: "manual", bias: null, iterations: 0, k: null, views: [] };
    const labels = renderer.getLabels();
    const names = [];
    const colors = [];
    for (const [label, e] of panel.entries) {
      names[label] = { nombre: e.nombre || e.name, nombre_es: e.nombre_es || e.name, categoria: e.categoria || "", confianza: Number.isFinite(e.confianza) ? e.confianza : null, malla: e.malla || null };
      colors[label] = labelColor(label).map((v) => Math.round(v * 255));
    }
    const info = window.__gsViewer || {};
    const json = buildInstancesJson({
      escena: info.name || "escena",
      fecha: new Date().toISOString(),
      fuente: { formato: info.format || "", sh_grado: cloud ? cloud.shDegree : 0 },
      metodo: {
        mascaras: l.source,
        sesgo_fondo: l.bias,
        asociacion: l.association || "manual",
        umbral_iou: l.iouThreshold ?? null,
        umbral_gaussiana: l.gaussianThreshold ?? null,
        difusion_iter: l.iterations,
        k_buffer: l.k,
      },
      labels,
      gaussians: cloud ? cloud.gaussians : null,
      names,
      colors,
      views: l.views.map((v) => ({ indice: v.indice, instancias: v.instancias })),
      embeddings: embeddings.vectors.size ? { vectors: Object.fromEntries(embeddings.vectors), modelo: embeddings.modelo, dimension: embeddings.dimension } : null,
    });
    return { json, bytes: labelsToBytes(labels) };
  }

  function downloadBlob(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  /** Download instancias.json + etiquetas.u32 and, when the sidecar answers, save them under artifacts/. */
  /** Validate instancias.json against shared/schemas/instancias.schema.json (plan §3.3); problems are logged, never fatal. */
  async function validateInstancesJson(json) {
    try {
      const { loadSchema, validateAgainst } = await import("../shared/schemas.js");
      const errors = validateAgainst(await loadSchema("instancias"), json);
      if (errors.length) console.warn(`[segmentación] instancias.json no cumple el esquema:\n  ${errors.join("\n  ")}`);
      return errors;
    } catch (err) {
      console.warn("[segmentación] no se pudo validar el esquema:", err.message);
      return null;
    }
  }

  async function exportSegmentation({ download = true, save = true } = {}) {
    const { json, bytes } = buildSegmentationExport();
    const schemaErrors = await validateInstancesJson(json);
    if (download) {
      downloadBlob(new Blob([JSON.stringify(json, null, 2)], { type: "application/json" }), "instancias.json");
      downloadBlob(new Blob([bytes], { type: "application/octet-stream" }), "etiquetas.u32");
    }
    let saved = null;
    if (save) {
      try {
        const res = await fetch(`${SIDECAR_URL}/segmentaciones`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ escena: json.escena, instancias: json, etiquetas_b64: await blobToB64(new Blob([bytes])) }),
        });
        const data = await res.json();
        if (res.ok && data.ok) saved = data;
      } catch (err) {
        console.info("[segmentación] sidecar no disponible para guardar en artifacts/:", err.message);
      }
    }
    setSegStatus(
      `Exportadas ${json.n_instancias} instancias (${formatCount(json.fuente.n_gaussianas)} etiquetas)` +
        (saved ? ` · guardado en ${saved.carpeta}` : " · descarga local (sidecar no disponible)") +
        (schemaErrors && schemaErrors.length ? ` · ${schemaErrors.length} avisos de esquema (consola)` : " · esquema válido"),
      schemaErrors && schemaErrors.length ? "err" : "ok"
    );
    return { json, saved, schemaErrors };
  }

  segEl.lift.addEventListener("click", () => liftMasks().catch(() => {}));
  segEl.export.addEventListener("click", () => exportSegmentation().catch((err) => setSegStatus(`Exportación fallida: ${err.message}`, "err")));

  window.__gsSegment = {
    lift: (options) => liftMasks(options),
    export: (options) => exportSegmentation(options),
    build: () => buildSegmentationExport(),
    get last() {
      return summarizeSeg();
    },
  };

  // ---------------------------------------------------------- Edición (F5)

  /** Scratch label that paints the live selection (never exported). */
  const SELECT_LABEL = MAX_INSTANCES - 1;
  const editEl = {
    tool: $("edit-tool"), mode: $("edit-mode"), brush: $("edit-brush"), sphere: $("edit-sphere"),
    selNew: $("edit-sel-new"), selAdd: $("edit-sel-add"), selBg: $("edit-sel-bg"), selClear: $("edit-sel-clear"), selStatus: $("edit-sel-status"),
    tx: $("edit-tx"), ty: $("edit-ty"), tz: $("edit-tz"), axis: $("edit-axis"), deg: $("edit-deg"), scale: $("edit-scale"),
    apply: $("edit-apply"), reset: $("edit-reset"), duplicate: $("edit-duplicate"), del: $("edit-delete"),
    merge: $("edit-merge"), mergeTarget: $("edit-merge-target"), rename: $("edit-rename"), name: $("edit-name"),
    undo: $("edit-undo"), redo: $("edit-redo"), saveOps: $("edit-save-ops"),
    scope: $("edit-scope"), format: $("edit-format"), export: $("edit-export"), status: $("edit-status"), rect: $("sel-rect"),
  };
  /**
   * log: EditLog over the labels as they were when editing started (its base);
   * selection: Uint8Array per gaussian while a selection is being built,
   * labelsBefore: the renderer labels the selection paints over.
   */
  const edit = { log: null, selection: null, selCount: 0, labelsBefore: null, brushQueue: Promise.resolve(), sceneRadius: 0, lastDeleted: new Set() };

  function setEditStatus(text, kind = "") {
    editEl.status.textContent = text;
    editEl.status.dataset.kind = kind;
    if (kind === "err") console.warn(`[edición] ${text}`);
    else console.info(`[edición] ${text}`);
  }

  /** The edit log, created on demand from the current cloud and labels. */
  function editLog() {
    if (!edit.log) {
      if (!cloud || !cloud.count) throw new Error("carga una escena antes de editar");
      const names = {};
      for (const [l, e] of panel.entries) names[l] = e.name;
      edit.log = new EditLog({ gaussians: cloud.gaussians, sh: cloud.sh, shDegree: cloud.shDegree, labels: edit.labelsBefore || renderer.getLabels(), names });
    }
    return edit.log;
  }

  /** Labels changed outside the log (load, lift, groups): start over from the new state. */
  function resetEdit() {
    clearSelection(false);
    edit.log = null;
    edit.sceneRadius = 0;
    edit.lastDeleted = new Set();
    updateEditButtons();
  }

  function updateEditButtons() {
    const has = edit.selCount > 0;
    editEl.selNew.disabled = !has;
    editEl.selAdd.disabled = !has || !panel.selection;
    editEl.selBg.disabled = !has;
    editEl.selClear.disabled = !has;
    editEl.undo.disabled = !edit.log || !edit.log.ops.length;
    editEl.redo.disabled = !edit.log || !edit.log.undone.length;
    editEl.saveOps.disabled = !edit.log || !edit.log.ops.length;
    editEl.selStatus.textContent = has ? `${formatCount(edit.selCount)} gaussianas seleccionadas` : "Sin selección.";
    const l = panel.selection ? panel.selection.label : 0;
    const deleted = !!(edit.log && l && edit.log.session.deleted.has(l));
    editEl.del.textContent = deleted ? "Restaurar" : "Borrar";
  }

  function sceneRadius() {
    if (!edit.sceneRadius && cloud && cloud.count) {
      const b = instanceBounds(new Uint32Array(cloud.count).fill(1), cloud.gaussians, 1);
      edit.sceneRadius = b ? b.radius : 1;
    }
    return edit.sceneRadius || 1;
  }

  // ---- selection (painted with SELECT_LABEL on top of the current labels)

  function ensureSelection() {
    if (edit.selection) return;
    edit.selection = new Uint8Array(renderer.count);
    edit.selCount = 0;
    edit.labelsBefore = renderer.getLabels();
    renderer.setInstance(SELECT_LABEL, { tint: [1, 0.85, 0.3, 0.9], selected: true, visible: true });
  }

  function paintSelection() {
    const labels = edit.labelsBefore.slice();
    for (let i = 0; i < labels.length; i++) if (edit.selection[i]) labels[i] = SELECT_LABEL;
    renderer.setLabels(labels);
    updateEditButtons();
  }

  /** Add or remove gaussian indices from the live selection. */
  function applySelection(indices, mode = editEl.mode.value) {
    if (!renderer.count) return 0;
    ensureSelection();
    let changed = 0;
    for (const i of indices) {
      if (i < 0 || i >= edit.selection.length) continue;
      if (mode === "remove") {
        if (edit.selection[i]) { edit.selection[i] = 0; edit.selCount--; changed++; }
      } else if (!edit.selection[i]) { edit.selection[i] = 1; edit.selCount++; changed++; }
    }
    paintSelection();
    return changed;
  }

  function clearSelection(restore = true) {
    if (!edit.selection) return;
    if (restore && edit.labelsBefore && edit.labelsBefore.length === renderer.count) renderer.setLabels(edit.labelsBefore);
    renderer.setInstance(SELECT_LABEL, { tint: [0, 0, 0, 0], selected: false });
    edit.selection = null;
    edit.selCount = 0;
    edit.labelsBefore = null;
    updateEditButtons();
  }

  function selectedIndices() {
    const out = [];
    if (edit.selection) for (let i = 0; i < edit.selection.length; i++) if (edit.selection[i]) out.push(i);
    return out;
  }

  function showSelectionRect(x0, y0, x1, y1) {
    const r = editEl.rect.style;
    r.display = "block";
    r.left = `${Math.min(x0, x1)}px`;
    r.top = `${Math.min(y0, y1)}px`;
    r.width = `${Math.abs(x1 - x0)}px`;
    r.height = `${Math.abs(y1 - y0)}px`;
  }

  function hideSelectionRect() {
    editEl.rect.style.display = "none";
  }

  async function selectRect(x0, y0, x1, y1, mode = editEl.mode.value) {
    const dpr = devicePixelScale();
    const idx = await renderer.pickRect(x0 * dpr, y0 * dpr, x1 * dpr, y1 * dpr);
    const n = applySelection(idx, mode);
    setEditStatus(`Rectángulo: ${formatCount(idx.length)} gaussianas visibles, ${formatCount(n)} ${mode === "remove" ? "quitadas" : "añadidas"}`);
    return idx.length;
  }

  function brushAt(cssX, cssY, mode = editEl.mode.value) {
    const dpr = devicePixelScale();
    const radius = Number(editEl.brush.value) * dpr;
    edit.brushQueue = edit.brushQueue.then(async () => {
      const idx = await renderer.pickDisc(cssX * dpr, cssY * dpr, radius);
      applySelection(idx, mode);
    }).catch((err) => setEditStatus(`Pincel: ${err.message}`, "err"));
    return edit.brushQueue;
  }

  function brushEnd() {
    return edit.brushQueue.then(() => setEditStatus(`Pincel: ${formatCount(edit.selCount)} gaussianas seleccionadas`));
  }

  /** Every gaussian within `radius` of the picked gaussian's centre (world units). */
  function selectSphere(index, radius, mode = editEl.mode.value) {
    if (!cloud || index < 0 || index >= cloud.count) return 0;
    const g = cloud.gaussians;
    const cx = g[index * 12], cy = g[index * 12 + 1], cz = g[index * 12 + 2];
    const r2 = radius * radius;
    const idx = [];
    for (let i = 0; i < cloud.count; i++) {
      const dx = g[i * 12] - cx, dy = g[i * 12 + 1] - cy, dz = g[i * 12 + 2] - cz;
      if (dx * dx + dy * dy + dz * dz <= r2) idx.push(i);
    }
    const n = applySelection(idx, mode);
    setEditStatus(`Esfera r=${radius.toFixed(3)}: ${formatCount(idx.length)} gaussianas, ${formatCount(n)} ${mode === "remove" ? "quitadas" : "añadidas"}`);
    return idx.length;
  }

  function selectGroup(index, mode = editEl.mode.value) {
    if (!groups.result) throw new Error("calcula los grupos (F2) para seleccionar por superpunto");
    if (index < 0 || index >= groups.result.superpoint.length) return 0;
    const sp = groups.result.superpoint[index];
    const idx = indicesOfGroup(groups.result.superpoint, sp);
    const n = applySelection(idx, mode);
    setEditStatus(`Superpunto ${sp + 1}: ${formatCount(idx.length)} gaussianas, ${formatCount(n)} ${mode === "remove" ? "quitadas" : "añadidas"}`);
    return idx.length;
  }

  async function selectFromHit(tool, hit) {
    if (hit.index < 0) return 0;
    try {
      if (tool === "esfera") return selectSphere(hit.index, (Number(editEl.sphere.value) / 100) * sceneRadius());
      return selectGroup(hit.index);
    } catch (err) {
      setEditStatus(err.message, "err");
      return 0;
    }
  }

  /** Turn the selection into labels: "new" instance, "add" to the selected one, or "bg" (fondo). */
  function commitSelection(mode = "new", target = null) {
    const idx = selectedIndices();
    if (!idx.length) throw new Error("no hay selección");
    const log = editLog();
    let label;
    if (mode === "new") label = log.session.nextLabel();
    else if (mode === "add") {
      label = target ?? (panel.selection ? panel.selection.label : 0);
      if (!label) throw new Error("selecciona una instancia de destino");
    } else label = 0;
    if (label >= SELECT_LABEL) throw new Error(`sin espacio para más instancias (máximo ${SELECT_LABEL - 1})`);
    clearSelection(true);
    const r = pushOp({ op: "asignar", id_instancia: label, rangos: rangesFromIndices(idx) });
    setEditStatus(`${formatCount(r.count)} gaussianas → ${label ? `instancia ${label}` : "fondo"}`, "ok");
    if (label) panel.select(label);
    return label;
  }

  // ---- ops → renderer / panel

  function pushOp(op) {
    const log = editLog();
    const r = log.push(op);
    syncFromSession();
    return r;
  }

  /** Mirror the session (labels, cloud size, transforms, deleted, names) into the renderer and the panel. */
  function syncFromSession() {
    const s = editLog().session;
    clearSelection(false);
    if (s.count !== renderer.count) {
      renderer.setCloud(s.gaussians, s.sh, s.shDegree);
      cloud = { gaussians: s.gaussians, sh: s.sh, shDegree: s.shDegree, count: s.count };
      resetGroups();
      naming.crops.clear();
      edit.sceneRadius = 0;
    }
    renderer.setLabels(s.labels);
    const present = new Set(s.labelSet());
    for (const l of s.xforms.keys()) present.add(l);
    for (const l of present) {
      if (!panel.entries.has(l)) panel.register(l, s.names.get(l) || `instancia ${l}`, 0);
      else if (s.names.has(l)) panel.entries.get(l).name = s.names.get(l);
    }
    for (const l of [...panel.entries.keys()]) if (!present.has(l) && !s.deleted.has(l)) panel.entries.delete(l);
    for (const l of panel.entries.keys()) {
      const wasVisible = renderer.getInstance(l).visible;
      const undeleted = edit.lastDeleted.has(l) && !s.deleted.has(l); // restaurar / undo of borrar
      renderer.setInstance(l, { xform: s.xformOf(l), visible: s.deleted.has(l) ? false : wasVisible || undeleted });
    }
    edit.lastDeleted = new Set(s.deleted);
    panel.refreshCounts(s.labels);
    if (panel.selection && !panel.entries.has(panel.selection.label)) panel.clear();
    updateEditButtons();
  }

  function currentLabel() {
    const l = panel.selection ? panel.selection.label : 0;
    if (!l) throw new Error("selecciona una instancia");
    return l;
  }

  /** Absolute xform (spec.xform) or a relative move/rotate/scale about the instance centre. */
  function transformInstance(label, spec = {}) {
    const s = editLog().session;
    let m;
    if (spec.xform) m = Float32Array.from(spec.xform);
    else {
      const c = s.centreOf(label) || [0, 0, 0];
      const pivot = transformPoint(s.xformOf(label), c);
      m = mat4Multiply(composeTransform({ ...spec, pivot }), s.xformOf(label));
    }
    pushOp({ op: "transformar", id_instancia: label, xform: Array.from(m) });
    setEditStatus(`Instancia ${label} transformada`, "ok");
    return Array.from(m);
  }

  function duplicateInstance(label, offset = null) {
    const s = editLog().session;
    const nueva = s.nextLabel();
    if (nueva >= SELECT_LABEL) throw new Error("sin espacio para más instancias");
    let shift = offset;
    if (!shift) {
      const b = instanceBounds(s.labels, s.gaussians, label);
      shift = [b ? b.radius * 2 : 1, 0, 0];
    }
    const xform = mat4Multiply(mat4Translation(shift), s.xformOf(label));
    const r = pushOp({ op: "duplicar", id_instancia: label, nueva, xform: Array.from(xform) });
    setEditStatus(`Instancia ${label} duplicada → ${nueva} (${formatCount(r.count)} gaussianas)`, "ok");
    panel.select(nueva);
    return nueva;
  }

  function deleteInstance(label) {
    pushOp({ op: "borrar", id_instancia: label });
    setEditStatus(`Instancia ${label} borrada (oculta y excluida de la exportación)`, "ok");
  }

  function restoreInstance(label) {
    pushOp({ op: "restaurar", id_instancia: label });
    renderer.setInstance(label, { visible: true });
    panel.renderList();
    setEditStatus(`Instancia ${label} restaurada`, "ok");
  }

  function mergeInstances(origen, destino) {
    if (origen === destino) throw new Error("origen y destino son la misma instancia");
    if (!panel.entries.has(destino)) throw new Error(`no existe la instancia ${destino}`);
    const r = pushOp({ op: "fusionar", origen, destino });
    setEditStatus(`Instancia ${origen} fusionada en ${destino} (${formatCount(r.count)} gaussianas)`, "ok");
    panel.select(destino);
    return destino;
  }

  function renameInstance(label, nombre) {
    const name = String(nombre || "").trim();
    if (!name) throw new Error("escribe un nombre");
    pushOp({ op: "renombrar", id_instancia: label, nombre_es: name });
    const e = panel.entries.get(label);
    if (e) { e.nombre_es = name; e.name = name; }
    panel.renderList();
    setEditStatus(`Instancia ${label} → «${name}»`, "ok");
  }

  function undoEdit() {
    const log = editLog();
    const op = log.undo();
    if (!op) return null;
    syncFromSession();
    setEditStatus(`Deshecho: ${op.op}`, "ok");
    return op;
  }

  function redoEdit() {
    const log = editLog();
    const op = log.redo();
    if (!op) return null;
    syncFromSession();
    setEditStatus(`Rehecho: ${op.op}`, "ok");
    return op;
  }

  /** Replace the log with ops from a JSONL text (replayed over the current base). */
  function replayOps(jsonl) {
    const ops = opsFromJsonl(jsonl);
    const base = editLog().base;
    replay(base, ops); // validate first: throws before touching the viewer
    edit.log = new EditLog(base, ops);
    syncFromSession();
    setEditStatus(`${ops.length} operaciones reproducidas`, "ok");
    return ops.length;
  }

  /** Convert bytes with the vendored GaussForge in the parse worker (PLY → SPZ, compressed PLY, …). */
  function convertBytes(buffer, inFormat, outFormat) {
    const id = ++jobId;
    return new Promise((resolve, reject) => {
      convertPending = { id, resolve, reject };
      worker.postMessage({ id, type: "convert", buffer, inFormat, outFormat }, [buffer]);
    }).then((msg) => msg.bytes);
  }

  /**
   * Export one instance or the visible scene with the transforms baked in.
   * PLY carries instance_id / class_id / confidence; SPZ and compressed PLY go
   * through GaussForge from that PLY; .splat is SH0 only.
   */
  async function exportObject({ scope = editEl.scope.value, label = null, format = editEl.format.value, download = true, save = true } = {}) {
    const log = editLog();
    const s = log.session;
    if (scope === "instancia") label = label ?? currentLabel();
    const hidden = new Set();
    for (const l of panel.entries.keys()) if (!renderer.getInstance(l).visible) hidden.add(l);
    const baked = scope === "instancia" ? bakeSession(s, { label }) : bakeSession(s, { hidden });
    if (!baked.count) throw new Error("nada que exportar (¿instancia vacía, borrada u oculta?)");
    const clases = [...new Set([...panel.entries.values()].map((e) => e.categoria).filter(Boolean))].sort();
    const classIds = new Uint32Array(baked.count);
    const confidences = new Float32Array(baked.count);
    for (let i = 0; i < baked.count; i++) {
      const e = panel.entries.get(baked.labels[i]);
      if (!e) continue;
      classIds[i] = e.categoria ? clases.indexOf(e.categoria) + 1 : 0;
      confidences[i] = Number.isFinite(e.confianza) ? e.confianza : 0;
    }
    const escena = (window.__gsViewer && window.__gsViewer.name) || "escena";
    const t0 = performance.now();
    setEditStatus(`Exportando ${scope === "instancia" ? `instancia ${label}` : "escena"} como ${format} (${formatCount(baked.count)} gaussianas)…`);
    let bytes;
    const comment = `F5 ${escena} ${scope}${label != null ? ` instancia ${label}` : ""}`;
    if (format === "splat") bytes = encodeSplat32(baked);
    else if (format === "ply") bytes = new Uint8Array(encodePly(baked, { labels: baked.labels, classIds, confidences, comment }));
    else bytes = await convertBytes(encodePly(baked, { comment }), "ply", format); // SPZ & co. cannot carry the extras
    const name = exportFileName(escena, scope === "instancia" ? label : null, format);
    const labelsOut = scope === "instancia" ? [label] : [...new Set(baked.labels)].filter((l) => l > 0).sort((a, b) => a - b);
    const metadatos = {
      version: 1,
      escena,
      fecha: new Date().toISOString(),
      ambito: scope,
      id_instancia: scope === "instancia" ? label : null,
      formato: format,
      archivo: name,
      n_gaussianas: baked.count,
      sh_grado: baked.shDegree,
      clases,
      transformaciones_aplicadas: labelsOut.filter((l) => s.xforms.has(l)),
      n_operaciones: log.ops.length,
      instancias: labelsOut.map((l) => {
        const e = panel.entries.get(l) || {};
        const b = instanceBounds(baked.labels, baked.gaussians, l);
        return { id_instancia: l, nombre: e.nombre || e.name || `instancia ${l}`, nombre_es: e.nombre_es || e.name || `instancia ${l}`, categoria: e.categoria || "", confianza: Number.isFinite(e.confianza) ? e.confianza : null, n_gaussianas: b ? b.count : 0, bbox: b ? { min: b.min, max: b.max } : null, xform: Array.from(s.xformOf(l)) };
      }),
    };
    if (download) downloadBlob(new Blob([bytes], { type: "application/octet-stream" }), name);
    let saved = null;
    if (save) {
      try {
        const res = await fetch(`${SIDECAR_URL}/exportaciones`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ escena, id_instancia: metadatos.id_instancia, formato: format, bytes_b64: await blobToB64(new Blob([bytes])), metadatos, ops_jsonl: log.toJsonl() }),
        });
        const data = await res.json();
        if (res.ok && data.ok) saved = data;
        else console.info("[edición] el sidecar no guardó la exportación:", data.error || res.status);
      } catch (err) {
        console.info("[edición] sidecar no disponible para guardar en artifacts/:", err.message);
      }
    }
    const ms = performance.now() - t0;
    setEditStatus(`${name}: ${formatCount(baked.count)} gaussianas · ${(bytes.length / 1e6).toFixed(2)} MB · ${ms.toFixed(0)} ms` + (saved ? ` · guardado en ${saved.archivo}` : " · descarga local"), "ok");
    return { name, format, count: baked.count, bytes: bytes.length, saved, metadatos, data: download ? null : bytes };
  }

  /** Save ops.jsonl (plus the base labels it applies to) next to instancias.json / etiquetas.u32. */
  async function saveOps() {
    const log = editLog();
    const { json, bytes } = buildSegmentationExport();
    const res = await fetch(`${SIDECAR_URL}/segmentaciones`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        escena: json.escena,
        instancias: json,
        etiquetas_b64: await blobToB64(new Blob([bytes])),
        etiquetas_base_b64: await blobToB64(new Blob([labelsToBytes(log.base.labels)])),
        ops_jsonl: log.toJsonl(),
      }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    setEditStatus(`${log.ops.length} operaciones guardadas en ${data.ops}`, "ok");
    return data;
  }

  // ---- HUD wiring
  const guard = (fn) => () => {
    try {
      const r = fn();
      if (r && typeof r.catch === "function") r.catch((err) => setEditStatus(err.message, "err"));
    } catch (err) {
      setEditStatus(err.message, "err");
    }
  };
  const axisVec = () => ({ x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] })[editEl.axis.value] || [0, 1, 0];
  editEl.tool.addEventListener("change", () => {
    document.body.classList.toggle("tool-rect", editEl.tool.value === "rect");
    document.body.classList.toggle("tool-pincel", editEl.tool.value === "pincel");
  });
  editEl.selNew.addEventListener("click", guard(() => commitSelection("new")));
  editEl.selAdd.addEventListener("click", guard(() => commitSelection("add")));
  editEl.selBg.addEventListener("click", guard(() => commitSelection("bg")));
  editEl.selClear.addEventListener("click", () => { clearSelection(true); setEditStatus("Selección limpiada"); });
  editEl.apply.addEventListener("click", guard(() => {
    const spec = { translate: [Number(editEl.tx.value) || 0, Number(editEl.ty.value) || 0, Number(editEl.tz.value) || 0], rotateAxis: axisVec(), rotateDeg: Number(editEl.deg.value) || 0, scale: Number(editEl.scale.value) || 1 };
    transformInstance(currentLabel(), spec);
    editEl.tx.value = editEl.ty.value = editEl.tz.value = "0";
    editEl.deg.value = "0";
    editEl.scale.value = "1";
  }));
  editEl.reset.addEventListener("click", guard(() => transformInstance(currentLabel(), { xform: Array.from(mat4Translation([0, 0, 0])) })));
  editEl.duplicate.addEventListener("click", guard(() => duplicateInstance(currentLabel())));
  editEl.del.addEventListener("click", guard(() => {
    const l = currentLabel();
    if (editLog().session.deleted.has(l)) restoreInstance(l);
    else deleteInstance(l);
  }));
  editEl.merge.addEventListener("click", guard(() => mergeInstances(currentLabel(), Number(editEl.mergeTarget.value) | 0)));
  editEl.rename.addEventListener("click", guard(() => renameInstance(currentLabel(), editEl.name.value)));
  editEl.undo.addEventListener("click", guard(() => undoEdit()));
  editEl.redo.addEventListener("click", guard(() => redoEdit()));
  editEl.saveOps.addEventListener("click", guard(() => saveOps()));
  editEl.export.addEventListener("click", guard(() => exportObject()));
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.target.matches("input, textarea, select")) {
      e.preventDefault();
      guard(() => (e.shiftKey ? redoEdit() : undoEdit()))();
    }
  });

  window.__gsEdit = {
    select: async ({ tool = "rect", mode = "add", x0 = 0, y0 = 0, x1 = 0, y1 = 0, x = 0, y = 0, index = -1, radius = null } = {}) => {
      if (tool === "rect") return selectRect(x0, y0, x1, y1, mode);
      if (tool === "pincel") { await brushAt(x, y, mode); return edit.selCount; }
      if (tool === "esfera") return selectSphere(index, radius ?? (Number(editEl.sphere.value) / 100) * sceneRadius(), mode);
      if (tool === "grupo") return selectGroup(index, mode);
      throw new Error(`herramienta desconocida: ${tool}`);
    },
    selection: () => ({ count: edit.selCount, indices: selectedIndices() }),
    clearSelection: () => clearSelection(true),
    commit: (mode, target) => commitSelection(mode, target),
    transform: (label, spec) => transformInstance(label, spec),
    duplicate: (label, offset) => duplicateInstance(label, offset),
    remove: (label) => deleteInstance(label),
    restore: (label) => restoreInstance(label),
    merge: (origen, destino) => mergeInstances(origen, destino),
    rename: (label, name) => renameInstance(label, name),
    undo: () => undoEdit(),
    redo: () => redoEdit(),
    replay: (jsonl) => replayOps(jsonl),
    ops: () => (edit.log ? edit.log.ops.map((o) => ({ ...o })) : []),
    jsonl: () => (edit.log ? edit.log.toJsonl() : ""),
    fingerprint: () => sessionFingerprint(editLog().session),
    session: () => {
      const s = editLog().session;
      return { count: s.count, labels: s.labelSet(), xforms: Object.fromEntries([...s.xforms].map(([l, m]) => [l, Array.from(m)])), deleted: [...s.deleted], names: Object.fromEntries(s.names) };
    },
    export: (options) => exportObject(options),
    saveOps: () => saveOps(),
    reset: () => resetEdit(),
  };
  window.__gsLoad = { buffer: (buffer, name) => loadFromBuffer(buffer, name), url: (url, name) => loadUrl(url, name), synthetic: (options) => loadSynthetic(options) };

  // ---------------------------------------------------------- Malla (F6)

  const meshEl = { views: $("mesh-views"), resolution: $("mesh-resolution"), edge: $("mesh-edge"), depth: $("mesh-depth"), carve: $("mesh-carve"), build: $("mesh-build"), status: $("mesh-status") };
  const meshState = { running: false, worker: null, jobId: 0, pending: null, last: null };

  function setMeshStatus(text, kind = "") {
    meshEl.status.textContent = text;
    meshEl.status.dataset.kind = kind;
    if (kind === "err") console.warn(`[malla] ${text}`);
    else console.info(`[malla] ${text}`);
  }

  function meshWorker() {
    if (!meshState.worker) {
      meshState.worker = new Worker(new URL("../shared/tsdf-worker.js", import.meta.url), { type: "module" });
      meshState.worker.onmessage = (e) => {
        const msg = e.data;
        if (!meshState.pending || msg.id !== meshState.pending.id) return;
        const { resolve, reject } = meshState.pending;
        meshState.pending = null;
        if (msg.ok) resolve(msg);
        else reject(new Error(msg.error));
      };
      meshState.worker.onerror = (e) => {
        if (meshState.pending) {
          meshState.pending.reject(new Error(e.message || "worker de malla"));
          meshState.pending = null;
        }
      };
    }
    return meshState.worker;
  }

  /** Depth (+ alpha) of the current camera: alpha-weighted mean (fast) or the 2DGS median from the K-buffer. */
  async function depthMap(kind, W, H) {
    if (kind === "mediana") {
      const c = await renderer.renderContributions({ mask: new Uint32Array(W * H), width: W, height: H, labelCount: 1, k: SEG_K });
      return { depth: c.medianDepth, alpha: c.alpha };
    }
    const d = await renderer.renderOffscreen({ mode: OUTPUT_MODE.DEPTH, width: W, height: H });
    return { depth: d.data, alpha: d.alpha };
  }

  /**
   * Mesh one instance: isolate it, orbit `views` cameras around its bounds,
   * fuse depth + colour into a TSDF in the worker and write a GLB.
   */
  async function meshInstance(label, options = {}) {
    if (!cloud || !cloud.count) throw new Error("carga una escena antes de crear mallas");
    if (meshState.running) throw new Error("ya hay una malla en curso");
    const labels = renderer.getLabels();
    const bounds = instanceBounds(labels, cloud.gaussians, label);
    if (!bounds) throw new Error(`la instancia ${label} no tiene gaussianas`);
    const views = Math.max(4, Math.min(64, Number(options.views ?? meshEl.views.value) || 24));
    const resolution = Math.max(16, Math.min(256, Number(options.resolution ?? meshEl.resolution.value) || 96));
    const edge = Math.max(64, Math.min(1024, Number(options.edge ?? meshEl.edge.value) || 256));
    const depthKind = options.depth || meshEl.depth.value;
    const carve = options.carve ?? meshEl.carve.checked;
    const download = options.download ?? true;
    const save = options.save ?? true;
    meshState.running = true;
    meshEl.build.disabled = true;
    freezeFrame = true;
    const saved = { target: camera.target.slice(), radius: camera.radius, yaw: camera.yaw, pitch: camera.pitch, isolate: renderer.params.isolateLabel };
    const t0 = performance.now();
    try {
      clearSelection(true);
      // Instance centre/radius in world space after its F5 transform.
      const xf = renderer.getInstance(label).xform;
      const center = transformPoint(xf, bounds.center);
      const scaleMax = Math.max(Math.hypot(xf[0], xf[1], xf[2]), Math.hypot(xf[4], xf[5], xf[6]), Math.hypot(xf[8], xf[9], xf[10]));
      const radius = bounds.radius * scaleMax;
      const frame = frameBounds({ center, radius }, camera.fov, 1.5);
      const W = edge;
      const H = Math.max(64, Math.round((edge * canvas.height) / Math.max(canvas.width, 1)));
      const cams = orbitCameras({ center, distance: frame.radius, count: views, fov: camera.fov, width: W, height: H, aspect: canvas.width / canvas.height });
      renderer.setParams({ isolateLabel: label });
      const captured = [];
      let msRender = 0;
      for (let v = 0; v < cams.length; v++) {
        const cam = cams[v];
        camera.target = center.slice();
        camera.radius = frame.radius;
        camera.yaw = cam.yaw;
        camera.pitch = cam.pitch;
        camera.dampYaw = camera.dampPitch = camera.dampPanX = camera.dampPanY = camera.dampZoom = 0;
        pushCamera();
        setMeshStatus(`Malla de ${label}: vista ${v + 1}/${cams.length} (${W}×${H}, profundidad ${depthKind})…`);
        const tv = performance.now();
        const { depth, alpha } = await depthMap(depthKind, W, H);
        const col = await renderer.renderOffscreen({ mode: OUTPUT_MODE.COLOR, width: W, height: H, clearColor: [0, 0, 0, 0] });
        msRender += performance.now() - tv;
        // The renderer's view matrix is the one actually used; take it from the last frame.
        captured.push({ depth, alpha, color: col.data, width: W, height: H, view: Float32Array.from(lastFrame.view), fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy });
      }
      setMeshStatus(`Malla de ${label}: fusionando ${cams.length} vistas en ${resolution}³ vóxeles…`);
      const id = ++meshState.jobId;
      const fused = await new Promise((resolve, reject) => {
        meshState.pending = { id, resolve, reject };
        meshWorker().postMessage(
          { id, views: captured, center, radius, resolution, carve, alphaMin: 0.05 },
          captured.flatMap((c) => [c.depth.buffer, c.alpha.buffer, c.color.buffer])
        );
      });
      const mesh = fused.mesh;
      if (!mesh.vertexCount) throw new Error("la fusión no produjo superficie (¿instancia demasiado pequeña o vistas vacías?)");
      const escena = (window.__gsViewer && window.__gsViewer.name) || "escena";
      const e = panel.entries.get(label) || {};
      const metadatos = {
        version: 1,
        escena,
        fecha: new Date().toISOString(),
        id_instancia: label,
        nombre_es: e.nombre_es || e.name || `instancia ${label}`,
        metodo: { profundidad: depthKind, vistas: cams.length, arista: [W, H], voxeles: resolution, voxel: fused.stats.voxelSize, truncamiento: fused.stats.truncation, tallado: carve, extraccion: "surface-nets", componentes: fused.stats.components, triangulos_descartados: fused.stats.removedTriangles },
        malla: { vertices: fused.stats.vertices, triangulos: fused.stats.triangles, bbox: fused.stats.bbox, euler: fused.stats.euler },
        aviso: window.__gsViewer && window.__gsViewer.variant === "2dgs" ? null : "3DGS vainilla: superficie ruidosa; usa PLY de 2DGS/GOF para malla de producción",
        tiempos_ms: { render: Math.round(msRender), fusion: Math.round(fused.msFuse), extraccion: Math.round(fused.ms - fused.msFuse) },
      };
      const glb = encodeGlb(mesh, { name: `${escena} instancia ${label}`, extras: { id_instancia: label, nombre_es: metadatos.nombre_es, escena } });
      const name = `${exportFileName(escena, label, "glb")}`;
      if (download) downloadBlob(new Blob([glb], { type: "model/gltf-binary" }), name);
      let savedTo = null;
      if (save) {
        try {
          const res = await fetch(`${SIDECAR_URL}/mallas`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ escena, id_instancia: label, glb_b64: await blobToB64(new Blob([glb])), metadatos }),
          });
          const data = await res.json();
          if (res.ok && data.ok) savedTo = data;
          else console.info("[malla] el sidecar no guardó la malla:", data.error || res.status);
        } catch (err) {
          console.info("[malla] sidecar no disponible para guardar en artifacts/:", err.message);
        }
      }
      if (savedTo && e) e.malla = savedTo.malla;
      const ms = performance.now() - t0;
      meshState.last = { label, name, stats: fused.stats, metadatos, saved: savedTo, ms, bytes: glb.byteLength };
      setMeshStatus(
        `${name}: ${formatCount(fused.stats.vertices)} vértices · ${formatCount(fused.stats.triangles)} triángulos · ${(glb.byteLength / 1e6).toFixed(2)} MB · render ${Math.round(msRender)} ms · fusión ${Math.round(fused.ms)} ms` +
          (savedTo ? ` · guardado en ${savedTo.malla}` : " · descarga local") +
          (metadatos.aviso ? ` · aviso: ${metadatos.aviso}` : ""),
        metadatos.aviso ? "warn" : "ok"
      );
      return { name, label, stats: fused.stats, metadatos, saved: savedTo, ms, bytes: glb.byteLength, glb: download ? null : glb, mesh: options.returnMesh ? mesh : null };
    } catch (err) {
      setMeshStatus(`Malla fallida: ${err.message}`, "err");
      throw err;
    } finally {
      renderer.setParams({ isolateLabel: saved.isolate });
      camera.target = saved.target;
      camera.radius = saved.radius;
      camera.yaw = saved.yaw;
      camera.pitch = saved.pitch;
      camera.dampYaw = camera.dampPitch = camera.dampPanX = camera.dampPanY = camera.dampZoom = 0;
      pushCamera();
      freezeFrame = false;
      meshState.running = false;
      meshEl.build.disabled = false;
    }
  }

  meshEl.build.addEventListener("click", () => {
    const l = panel.selection ? panel.selection.label : 0;
    if (!l) { setMeshStatus("selecciona una instancia", "err"); return; }
    meshInstance(l).catch(() => {});
  });

  window.__gsMesh = {
    build: (label, options) => meshInstance(label, options),
    get last() {
      return meshState.last;
    },
  };

  // ---------------------------------------------------------- Nombrar (F4)

  const nameEl = { button: $("inst-name"), status: $("inst-name-status"), search: $("inst-search") };
  const naming = { running: false, crops: new Map() };
  const CROP_EDGE = 512;

  function setNameStatus(text, kind = "") {
    nameEl.status.textContent = text;
    nameEl.status.dataset.kind = kind;
  }

  /**
   * Isolated render of one instance: camera framed on its bounds, only its
   * gaussians drawn, opaque white background. Returns a PNG (base64).
   */
  async function renderInstanceCrop(label, edge = CROP_EDGE) {
    if (!cloud) throw new Error("no hay escena cargada");
    const bounds = instanceBounds(renderer.getLabels(), cloud.gaussians, label);
    if (!bounds) throw new Error(`la instancia ${label} no tiene gaussianas`);
    const frame = frameBounds(bounds, camera.fov, 1.4);
    const saved = { target: camera.target.slice(), radius: camera.radius, isolate: renderer.params.isolateLabel };
    camera.target = frame.target;
    camera.radius = frame.radius;
    camera.dampYaw = camera.dampPitch = camera.dampPanX = camera.dampPanY = camera.dampZoom = 0;
    pushCamera();
    renderer.setParams({ isolateLabel: label });
    const W = edge;
    const H = Math.max(64, Math.round((edge * canvas.height) / Math.max(canvas.width, 1)));
    try {
      const col = await renderer.renderOffscreen({ mode: OUTPUT_MODE.COLOR, width: W, height: H, clearColor: [1, 1, 1, 1] });
      const png = await rgbaToPngB64(col.data, W, H);
      naming.crops.set(label, { png, rgba: col.data, width: W, height: H, bounds });
      return png;
    } finally {
      renderer.setParams({ isolateLabel: saved.isolate });
      camera.target = saved.target;
      camera.radius = saved.radius;
      pushCamera();
    }
  }

  /** Ask the sidecar to name instances (all registered ones by default). */
  async function nameInstances({ labels: only = null, backend = null } = {}) {
    const targets = only || [...panel.entries.keys()];
    if (!targets.length) throw new Error("no hay instancias que nombrar");
    if (naming.running) throw new Error("ya hay un nombrado en curso");
    naming.running = true;
    nameEl.button.disabled = true;
    freezeFrame = true;
    const t0 = performance.now();
    try {
      const health = await fetch(`${SIDECAR_URL}/health`);
      if (!health.ok) throw new Error(`sidecar HTTP ${health.status}`);
      const h = await health.json();
      const chosen = backend || (h.name_backend === "mock" ? "mock" : "grok");
      if (chosen === "grok" && !h.xai) throw new Error("el sidecar no tiene XAI_API_KEY");
      const instances = [];
      for (const label of targets) {
        const entry = panel.entries.get(label);
        if (!entry) continue;
        setNameStatus(`Renderizando instancia ${label} aislada (${instances.length + 1}/${targets.length})…`);
        instances.push({ id: label, hint: entry.nombre || (entry.name.startsWith("grupo") ? "" : entry.name), png_b64: await renderInstanceCrop(label) });
      }
      setNameStatus(`Nombrando ${instances.length} instancias con el sidecar (${chosen})…`);
      const res = await fetch(`${SIDECAR_URL}/name`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instances, backend: chosen }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || `name ${res.status}`);
      const { applied, failed } = applyNames(panel.entries, data.instances);
      panel.renderList();
      if (panel.selection) panel.select(panel.selection.label, panel.selection.index);
      const ms = performance.now() - t0;
      setNameStatus(`${applied} instancias nombradas${failed ? `, ${failed} sin nombre` : ""} · ${data.backend}${data.vision_model ? " · " + data.vision_model : ""} · ${ms.toFixed(0)} ms`, failed ? "err" : "ok");
      console.info(`[nombres] ${applied} nombradas, ${failed} fallidas en ${ms.toFixed(0)} ms`, data.instances);
      return { applied, failed, backend: data.backend, instances: data.instances };
    } catch (err) {
      const msg = String(err && err.message ? err.message : err);
      const hint = /fetch|Failed|ECONNREFUSED|NetworkError/i.test(msg) ? " — arranca ./semantic_sidecar/launch.sh (puerto 8766)" : "";
      setNameStatus(`Nombrado fallido: ${msg}${hint}`, "err");
      throw err;
    } finally {
      freezeFrame = false;
      naming.running = false;
      nameEl.button.disabled = false;
    }
  }

  /** Imagine card of one instance from its isolated crop, shown in the right panel with its id. */
  async function cardForInstance(label) {
    const entry = panel.entries.get(label);
    if (!entry) throw new Error(`instancia ${label} desconocida`);
    freezeFrame = true;
    try {
      setNameStatus(`Generando tarjeta Imagine de la instancia ${label}…`);
      const png = await renderInstanceCrop(label);
      const res = await fetch(`${SIDECAR_URL}/card`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ png_b64: png, name: entry.nombre || entry.name }),
      });
      const card = await res.json();
      if (!res.ok || !card.ok) throw new Error(card.error || `card ${res.status}`);
      const cards = $("sem-cards");
      const fig = document.createElement("figure");
      fig.dataset.label = String(label);
      const src = card.b64 ? `data:${card.mime || "image/jpeg"};base64,${card.b64}` : card.url;
      fig.innerHTML = `<img alt="" src="${src}" /><figcaption></figcaption>`;
      fig.querySelector("img").alt = entry.name;
      fig.querySelector("figcaption").textContent = `#${label} ${entry.name}${card.path ? ` · ${card.path}` : ""}`;
      cards.prepend(fig);
      setNameStatus(`Tarjeta de la instancia ${label} lista`, "ok");
      return { label, path: card.path || null };
    } catch (err) {
      setNameStatus(`Tarjeta fallida: ${err.message}`, "err");
      throw err;
    } finally {
      freezeFrame = false;
    }
  }

  // ---- CLIP embeddings per instance (F4 optional) and semantic search
  const embeddings = { vectors: new Map(), modelo: "", dimension: 0 };

  async function embedInstances({ labels: only = null } = {}) {
    const targets = only || [...panel.entries.keys()];
    if (!targets.length) throw new Error("no hay instancias que incrustar");
    const clip = await loadBrowserClip();
    freezeFrame = true;
    const t0 = performance.now();
    try {
      let done = 0;
      for (const label of targets) {
        if (!panel.entries.has(label)) continue;
        setNameStatus(`CLIP: instancia ${label} (${done + 1}/${targets.length})…`);
        let crop = naming.crops.get(label);
        if (!crop || !crop.rgba) {
          await renderInstanceCrop(label);
          crop = naming.crops.get(label);
        }
        embeddings.vectors.set(label, await clip.embedImage(crop.rgba, crop.width, crop.height));
        done++;
      }
      embeddings.modelo = clip.meta.id;
      embeddings.dimension = embeddings.vectors.values().next().value?.length || 0;
      const ms = performance.now() - t0;
      setNameStatus(`${done} embeddings CLIP (${embeddings.dimension} d) · ${clip.meta.device} · ${ms.toFixed(0)} ms`, "ok");
      return { count: done, dimension: embeddings.dimension, modelo: embeddings.modelo, ms };
    } catch (err) {
      setNameStatus(`Embeddings fallidos: ${err.message}`, "err");
      throw err;
    } finally {
      freezeFrame = false;
    }
  }

  /** Rank instances by CLIP similarity to a text query; highlights and optionally selects the best. */
  async function searchSemantic(query, select = true) {
    const q = String(query || "").trim();
    if (!q) return [];
    if (!embeddings.vectors.size) await embedInstances();
    const clip = await loadBrowserClip();
    const t = await clip.embedText(q);
    const Clip = (await loadMl()).BrowserClip;
    const ranked = [...embeddings.vectors.entries()]
      .map(([label, v]) => ({ label, score: Clip.cosine(v, t) }))
      .sort((a, b) => b.score - a.score);
    panel.matches = new Set(ranked.slice(0, Math.max(1, Math.ceil(ranked.length / 3))).map((r) => r.label));
    panel.renderList();
    if (select && ranked.length) panel.select(ranked[0].label);
    setNameStatus(`Semántica «${q}»: ${ranked.slice(0, 3).map((r) => `#${r.label} ${r.score.toFixed(3)}`).join(" · ")}`, "ok");
    return ranked;
  }

  $("inst-embed").addEventListener("click", () => embedInstances().catch(() => {}));

  panel.onAction = (act, label) => {
    if (act === "name") nameInstances({ labels: [label] }).catch(() => {});
    else if (act === "card") cardForInstance(label).catch(() => {});
    else if (act === "mesh") meshInstance(label).catch(() => {});
  };
  nameEl.button.addEventListener("click", () => nameInstances().catch(() => {}));
  nameEl.search.addEventListener("input", () => panel.search(nameEl.search.value));
  nameEl.search.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    if ($("inst-semantic").checked) {
      searchSemantic(nameEl.search.value).catch(() => {});
      return;
    }
    const found = panel.search(nameEl.search.value);
    if (found.length) panel.select(found[0].label);
    else if (nameEl.search.value.trim()) panel.setStatus(`Sin resultados para «${nameEl.search.value.trim()}»`);
  });

  window.__gsNames = {
    name: (options) => nameInstances(options),
    card: (label) => cardForInstance(label),
    crop: (label) => renderInstanceCrop(label),
    search: (query, select = true) => {
      const found = panel.search(query);
      if (select && found.length) panel.select(found[0].label);
      return found;
    },
    entries: () => panel.rows(),
    embed: (options) => embedInstances(options),
    searchSemantic: (query, select = true) => searchSemantic(query, select),
    embeddings: () => ({ modelo: embeddings.modelo, dimension: embeddings.dimension, labels: [...embeddings.vectors.keys()] }),
  };

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
    cloud = { gaussians: result.gaussians, sh: result.sh, shDegree: result.shDegree || 0, count: result.count };
    panel.reset();
    resetGroups();
    resetEdit();
    let restored = 0;
    if (result.labels && result.labels.length === result.count) {
      // F5: a PLY exported with instance_id brings its instances back.
      renderer.setLabels(result.labels);
      panel.fromLabels(result.labels);
      restored = panel.entries.size;
      console.info(`[edición] ${restored} instancias restauradas desde instance_id`);
    }
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
    const note = (result.warning ? `Ready (${result.warning})` : "Ready") + (restored ? ` · ${restored} instancias (instance_id)` : "");
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
      variant: result.variant || null,
      labelSource: result.labelSource || null,
      restoredInstances: restored,
    };
  }

  /** Build the two-sphere scene on the CPU and register its instances (labels 1 and 2). */
  function loadSynthetic(options = {}) {
    // applyLabels=false leaves every gaussian as fondo (used to test group → instance promotion)
    const { applyLabels = true, ...sceneOptions } = options;
    ui.setStatus("Generando escena sintética…");
    const scene = makeTwoSpheres(sceneOptions);
    renderer.setCloud(scene.gaussians, scene.sh, scene.shDegree);
    cloud = { gaussians: scene.gaussians, sh: scene.sh, shDegree: scene.shDegree, count: scene.count };
    panel.reset();
    resetGroups();
    resetEdit();
    if (applyLabels) {
      renderer.setLabels(scene.labels);
      panel.fromLabels(scene.labels, scene.names);
    }
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
      loadSynthetic({ applyLabels: query.get("labels") !== "0" });
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
