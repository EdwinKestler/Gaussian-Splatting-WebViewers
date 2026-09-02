/**
 * Browser-side harness for the F1 identity e2e tests (tests/e2e/f1-identity.spec.mjs).
 *
 * Served by the test web server from the repo root as
 * /tests/e2e/pages/f1-harness.js (the imports below resolve against that path).
 *
 * The renderer is always initialised with { offscreenOnly: true }: nothing here
 * ever touches a canvas WebGPU context, so it runs under headless SwiftShader.
 * Camera conventions match gaussian_splatting_webgpu/main.js and selftest.html
 * (column-major matrices, -z forward, pixel origin top-left).
 */

import {
  makeTwoSpheres,
  makeSingleGaussian,
  SPHERE_LABELS,
  GAUSSIAN_STRIDE,
} from "../../../shared/synthetic.js";
import {
  WebGPUSplatRenderer,
  OUTPUT_MODE,
  MAX_INSTANCES,
} from "../../../gaussian_splatting_webgpu/gpu-renderer.js";

export { makeTwoSpheres, makeSingleGaussian, SPHERE_LABELS, GAUSSIAN_STRIDE, OUTPUT_MODE, MAX_INSTANCES };

// ------------------------------------------------------------------ maths

function assertVec3(name, v) {
  if (!v || v.length !== 3 || ![0, 1, 2].every((k) => Number.isFinite(v[k]))) {
    throw new Error(`${name} must be three finite numbers`);
  }
}

/** Column-major OpenGL-style perspective matrix (same as main.js). */
export function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect;
  o[5] = f;
  o[10] = (far + near) / (near - far);
  o[11] = -1;
  o[14] = (2 * far * near) / (near - far);
  return o;
}

export function normalize3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

export function cross3(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Column-major view matrix looking from `eye` at `target`. */
export function lookAt(eye, target, up) {
  const z = normalize3([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize3(cross3(up, z));
  const y = cross3(z, x);
  const o = new Float32Array(16);
  o[0] = x[0]; o[1] = y[0]; o[2] = z[0]; o[3] = 0;
  o[4] = x[1]; o[5] = y[1]; o[6] = z[1]; o[7] = 0;
  o[8] = x[2]; o[9] = y[2]; o[10] = z[2]; o[11] = 0;
  o[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
  o[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
  o[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
  o[15] = 1;
  return o;
}

/** m (column-major 4x4) * v (vec4). */
export function mulVec4(m, v) {
  return [0, 1, 2, 3].map((r) => m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3]);
}

/** World point -> integer pixel [x, y] (origin top-left) or null when behind the camera. */
export function projectToPixel(proj, view, p, width, height) {
  const c = mulVec4(proj, mulVec4(view, [p[0], p[1], p[2], 1]));
  if (c[3] <= 0) return null;
  const nx = c[0] / c[3];
  const ny = c[1] / c[3];
  return [Math.round((nx * 0.5 + 0.5) * width), Math.round((1 - (ny * 0.5 + 0.5)) * height)];
}

/** Column-major translation matrix as a plain array of 16 numbers. */
export function translationMatrix(t) {
  assertVec3("translation", t);
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, t[0], t[1], t[2], 1];
}

/** Unit quaternion (w, x, y, z) for a rotation of `deg` degrees about `axis`. */
export function quatAxisAngle(axis, deg) {
  assertVec3("axis", axis);
  const a = normalize3(axis);
  const half = (deg * Math.PI) / 360;
  const s = Math.sin(half);
  return [Math.cos(half), a[0] * s, a[1] * s, a[2] * s];
}

// ---------------------------------------------------------------- harness

/**
 * One offscreen renderer plus a pinhole camera of fixed size.
 * Every method returns plain JSON-serialisable values so a Playwright
 * `page.evaluate` can hand them back to the test.
 */
export class F1Harness {
  constructor({ width = 512, height = 384, fovDeg = 50, near = 0.05, far = 100 } = {}) {
    for (const [name, v] of Object.entries({ width, height, fovDeg, near, far })) {
      if (!Number.isFinite(v) || v <= 0) throw new Error(`${name} must be > 0, got ${v}`);
    }
    if (!Number.isInteger(width) || !Number.isInteger(height)) {
      throw new Error("width/height must be integers");
    }
    this.width = width;
    this.height = height;
    this.fov = (fovDeg * Math.PI) / 180;
    this.near = near;
    this.far = far;
    this.renderer = null;
    this.adapter = null;
    this.lost = null;
    this.uncaptured = [];
    this.proj = null;
    this.view = null;
    this.eye = null;
    this.focal = 0;
    this.scene = null;
  }

  /** Create the renderer (offscreenOnly) and start device-loss bookkeeping. */
  async init() {
    if (!navigator.gpu) throw new Error("navigator.gpu no disponible (¿se sirve la página por http://?)");
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const info = adapter.info || {};
      this.adapter = { vendor: info.vendor || "", architecture: info.architecture || "" };
    }
    const renderer = new WebGPUSplatRenderer(document.createElement("canvas"));
    await renderer.init({ offscreenOnly: true });
    renderer.device.lost.then((info) => {
      this.lost = { reason: info.reason, message: info.message };
    });
    renderer.device.addEventListener("uncapturederror", (ev) => {
      this.uncaptured.push(String(ev.error && ev.error.message ? ev.error.message : ev.error));
    });
    this.renderer = renderer;
    console.log(`[f1-harness] renderer offscreenOnly listo · formato flotante ${renderer.floatFormat}`);
    return this;
  }

  /** Point the camera and push it to the renderer; returns the pixel focal length. */
  lookFrom(eye, target = [0, 0, 0], up = [0, 1, 0]) {
    assertVec3("eye", eye);
    assertVec3("target", target);
    const { width, height } = this;
    this.proj = perspective(this.fov, width / height, this.near, this.far);
    this.view = lookAt(eye, target, up);
    this.eye = eye.slice();
    this.focal = height / (2 * Math.tan(this.fov / 2));
    this.renderer.setCamera(this.proj, this.view, [this.focal, this.focal], [width, height], eye);
    return this.focal;
  }

  /** World point -> pixel with the current camera. */
  project(p) {
    if (!this.proj) throw new Error("lookFrom() must be called first");
    return projectToPixel(this.proj, this.view, p, this.width, this.height);
  }

  /** View-space distance (-cam.z) of a world point. */
  viewDepth(p) {
    return -mulVec4(this.view, [p[0], p[1], p[2], 1])[2];
  }

  /** View depth of the point of a sphere's surface closest to the camera. */
  frontDepth(center, radius) {
    const dir = normalize3([center[0] - this.eye[0], center[1] - this.eye[1], center[2] - this.eye[2]]);
    return this.viewDepth([center[0] - radius * dir[0], center[1] - radius * dir[1], center[2] - radius * dir[2]]);
  }

  /** Upload a scene ({gaussians, sh, shDegree, labels?}). */
  setScene(scene) {
    if (!scene || !scene.gaussians) throw new Error("scene must have gaussians");
    this.renderer.setCloud(scene.gaussians, scene.sh, scene.shDegree || 0);
    if (scene.labels) this.renderer.setLabels(scene.labels);
    this.scene = scene;
    return scene.count;
  }

  /** renderOffscreen() at the harness size. */
  render(mode) {
    return this.renderer.renderOffscreen({ mode, width: this.width, height: this.height });
  }

  /** Labels currently in the renderer (CPU mirror). */
  labels() {
    return this.renderer.getLabels();
  }

  /**
   * Render the ID pass and summarise it per label: pixel counts, centroids and
   * x/y ranges, plus the number of ids outside [1, count] (must be 0).
   */
  async idStats() {
    const id = await this.render(OUTPUT_MODE.ID);
    const labels = this.labels();
    const count = labels.length;
    const { width, height } = this;
    const perLabel = new Map();
    let invalid = 0;
    let nonZero = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v = id.data[y * width + x];
        if (!v) continue;
        nonZero++;
        if (v > count) {
          invalid++;
          continue;
        }
        const label = labels[v - 1];
        let s = perLabel.get(label);
        if (!s) {
          s = { pixels: 0, sumX: 0, sumY: 0, minX: width, maxX: -1, minY: height, maxY: -1 };
          perLabel.set(label, s);
        }
        s.pixels++;
        s.sumX += x;
        s.sumY += y;
        if (x < s.minX) s.minX = x;
        if (x > s.maxX) s.maxX = x;
        if (y < s.minY) s.minY = y;
        if (y > s.maxY) s.maxY = y;
      }
    }
    const byLabel = {};
    for (const [label, s] of perLabel) {
      byLabel[label] = {
        pixels: s.pixels,
        centroid: [s.sumX / s.pixels, s.sumY / s.pixels],
        xRange: [s.minX, s.maxX],
        yRange: [s.minY, s.maxY],
      };
    }
    this._lastId = id;
    return { nonZero, invalid, byLabel };
  }

  /** Gaussian index/label under a pixel of the last idStats() frame (index -1 = nothing). */
  idAt(x, y) {
    if (!this._lastId) throw new Error("idStats() must be called first");
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return { index: -1, label: 0 };
    const v = this._lastId.data[y * this.width + x];
    if (!v) return { index: -1, label: 0 };
    const index = v - 1;
    const labels = this.labels();
    return { index, label: index < labels.length ? labels[index] : -1 };
  }

  /** renderer.pick() (device pixels == harness pixels). */
  pick(x, y, opts = {}) {
    return this.renderer.pick(x, y, opts);
  }

  /** Expected depth and alpha at one pixel of a DEPTH readback. */
  async depthAt(x, y) {
    const d = await this.render(OUTPUT_MODE.DEPTH);
    const p = y * this.width + x;
    return { depth: d.alpha[p] > 0 ? d.data[p] : null, alpha: d.alpha[p], format: d.format };
  }

  /** Unit view-space normal and alpha at one pixel of a NORMAL readback. */
  async normalAt(x, y) {
    const n = await this.render(OUTPUT_MODE.NORMAL);
    const p = y * this.width + x;
    return {
      normal: n.alpha[p] > 0 ? [n.data[p * 3], n.data[p * 3 + 1], n.data[p * 3 + 2]] : null,
      alpha: n.alpha[p],
      format: n.format,
    };
  }

  /** Colour RGBA (0..255) and alpha coverage in a (2*half+1)^2 box around a pixel. */
  async colourAt(x, y, half = 0) {
    const c = await this.render(OUTPUT_MODE.COLOR);
    const { width, height } = this;
    let covered = 0;
    let total = 0;
    for (let yy = y - half; yy <= y + half; yy++) {
      for (let xx = x - half; xx <= x + half; xx++) {
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
        total++;
        if (c.data[(yy * width + xx) * 4 + 3] > 0) covered++;
      }
    }
    const o = (y * width + x) * 4;
    return { rgba: Array.from(c.data.subarray(o, o + 4)), covered, total };
  }

  /** Wait for the GPU and report device health. */
  async health() {
    await this.renderer.device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 100));
    return { lost: this.lost, uncaptured: this.uncaptured.slice(), adapter: this.adapter, floatFormat: this.renderer.floatFormat };
  }
}
