/**
 * Synthetic gaussian scenes for tests and the viewer (plan F1: two spheres).
 *
 * Pure JS, no DOM/WebGPU: importable from Node (`node --test`) and the browser.
 *
 * Output layout matches shared/splat-io.js and gpu-renderer.js:
 *   gaussians: Float32Array N*12 = [x,y,z, opacity, sx,sy,sz, pad, qw,qx,qy,qz]
 *   sh:        Float32Array N*48 (16 RGB coefficients, DC first; only DC is set)
 *   labels:    Uint32Array N (instance id per gaussian, 0 = fondo)
 */

export const GAUSSIAN_STRIDE = 12;
export const SH_STRIDE = 48;
/** Zeroth-order SH basis constant: colour = 0.5 + SH_C0 * dc. */
export const SH_C0 = 0.28209479177387814;

/** Default base colours: sphere A teal, sphere B orange (linear RGB in 0..1). */
export const SPHERE_COLORS = Object.freeze({
  A: Object.freeze([0.16, 0.72, 0.66]),
  B: Object.freeze([0.95, 0.55, 0.16]),
});

/** Labels assigned by makeTwoSpheres (1 = esfera A, 2 = esfera B). */
export const SPHERE_LABELS = Object.freeze({ A: 1, B: 2 });

/**
 * Deterministic 32-bit PRNG (mulberry32). Returns a function yielding floats in [0, 1).
 * @param {number} seed
 * @returns {() => number}
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** RGB (0..1) → SH DC coefficient (inverse of colour = 0.5 + SH_C0 * dc). */
export function rgbToShDc(rgb) {
  assertVec3("color", rgb);
  return [(rgb[0] - 0.5) / SH_C0, (rgb[1] - 0.5) / SH_C0, (rgb[2] - 0.5) / SH_C0];
}

function assertVec3(name, v) {
  if (!v || v.length !== 3 || ![0, 1, 2].every((k) => Number.isFinite(v[k]))) {
    throw new Error(`${name} must be three finite numbers`);
  }
}

function assertPositive(name, v) {
  if (!Number.isFinite(v) || v <= 0) throw new Error(`${name} must be > 0, got ${v}`);
}

function assertUnit(name, v) {
  if (!Number.isFinite(v) || v < 0 || v > 1) throw new Error(`${name} must be in [0, 1], got ${v}`);
}

/**
 * Write one gaussian (identity rotation, isotropic scale) into `gaussians`/`sh` at slot i.
 */
function writeGaussian(gaussians, sh, i, position, scale, opacity, color) {
  const o = i * GAUSSIAN_STRIDE;
  gaussians[o] = position[0];
  gaussians[o + 1] = position[1];
  gaussians[o + 2] = position[2];
  gaussians[o + 3] = opacity;
  gaussians[o + 4] = scale[0];
  gaussians[o + 5] = scale[1];
  gaussians[o + 6] = scale[2];
  gaussians[o + 7] = 0;
  gaussians[o + 8] = 1; // qw
  gaussians[o + 9] = 0;
  gaussians[o + 10] = 0;
  gaussians[o + 11] = 0;
  const dc = rgbToShDc(color);
  const s = i * SH_STRIDE;
  sh[s] = dc[0];
  sh[s + 1] = dc[1];
  sh[s + 2] = dc[2];
}

/** Uniform random unit vector (Marsaglia) from a PRNG. */
function randomUnitVector(rand) {
  const z = 2 * rand() - 1;
  const phi = 2 * Math.PI * rand();
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(phi), r * Math.sin(phi), z];
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

/**
 * k-th of n points of a Fibonacci lattice on the unit sphere (even spacing),
 * displaced by a PRNG jitter of `jitter` × the mean point spacing.
 */
function fibonacciUnitVector(k, n, rand, jitter) {
  const spacing = Math.sqrt((4 * Math.PI) / n); // mean angular distance between points
  let z = 1 - (2 * (k + 0.5)) / n;
  if (jitter > 0) z += (2 * rand() - 1) * jitter * (2 / n);
  z = Math.max(-1, Math.min(1, z));
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  let phi = k * GOLDEN_ANGLE;
  if (jitter > 0 && r > 1e-6) phi += ((2 * rand() - 1) * jitter * spacing) / r;
  return [r * Math.cos(phi), r * Math.sin(phi), z];
}

/** Unit direction for sample k of n according to `distribution`. */
function sampleDirection(distribution, k, n, rand, jitter) {
  if (distribution === "random") return randomUnitVector(rand);
  return fibonacciUnitVector(k, n, rand, jitter);
}

/** Bounds in the same shape as splat-io's boundsFromGaussians ({min, max, center, radius}). */
function sphereBounds(centers, radius) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const c of centers) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], c[a] - radius);
      max[a] = Math.max(max[a], c[a] + radius);
    }
  }
  const center = [0, 1, 2].map((a) => 0.5 * (min[a] + max[a]));
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  return { min, max, center, radius: extent * 0.5 || 1 };
}

/**
 * Two hollow spheres of gaussians side by side on the x axis.
 *
 * Sphere A (label 1, teal) is centred at x = -separation/2, sphere B (label 2,
 * orange) at x = +separation/2. Gaussians lie on each sphere surface with a
 * small isotropic scale (default radius * 0.06), identity rotation and opacity
 * 0.9. Deterministic for a given seed (mulberry32).
 *
 * distribution "fibonacci" (default): evenly spaced Fibonacci lattice plus a
 * PRNG jitter of `jitter` × mean spacing, so the surface has no gaps (the ID
 * pass only keeps alpha ≥ 0.5, i.e. ~1σ around each centre). "random":
 * uniform random directions (Poisson gaps are expected).
 *
 * @param {{n?: number, radius?: number, separation?: number, seed?: number,
 *   scale?: number, opacity?: number, colorA?: number[], colorB?: number[],
 *   distribution?: "fibonacci"|"random", jitter?: number}} [options]
 *   n: total gaussian count (split evenly; must be >= 2)
 * @returns {{gaussians: Float32Array, sh: Float32Array, shDegree: 0, count: number,
 *   labels: Uint32Array, centers: number[][], radius: number,
 *   bounds: {min:number[], max:number[], center:number[], radius:number},
 *   names: {1: string, 2: string}}}
 */
export function makeTwoSpheres({
  n = 4000,
  radius = 0.5,
  separation = 2.0,
  seed = 1,
  scale = radius * 0.06,
  opacity = 0.9,
  colorA = SPHERE_COLORS.A,
  colorB = SPHERE_COLORS.B,
  distribution = "fibonacci",
  jitter = 0.25,
} = {}) {
  if (!Number.isInteger(n) || n < 2) throw new Error(`n must be an integer >= 2, got ${n}`);
  assertPositive("radius", radius);
  assertPositive("separation", separation);
  assertPositive("scale", scale);
  assertUnit("opacity", opacity);
  if (!Number.isInteger(seed)) throw new Error(`seed must be an integer, got ${seed}`);
  assertVec3("colorA", colorA);
  assertVec3("colorB", colorB);
  if (distribution !== "fibonacci" && distribution !== "random") {
    throw new Error(`distribution must be "fibonacci" or "random", got ${distribution}`);
  }
  if (!Number.isFinite(jitter) || jitter < 0) throw new Error(`jitter must be >= 0, got ${jitter}`);

  const centers = [
    [-separation / 2, 0, 0],
    [separation / 2, 0, 0],
  ];
  const colors = [colorA, colorB];
  const perSphere = [Math.ceil(n / 2), Math.floor(n / 2)];
  const gaussians = new Float32Array(n * GAUSSIAN_STRIDE);
  const sh = new Float32Array(n * SH_STRIDE);
  const labels = new Uint32Array(n);
  const rand = mulberry32(seed);
  const iso = [scale, scale, scale];

  let i = 0;
  for (let s = 0; s < 2; s++) {
    const c = centers[s];
    const label = s === 0 ? SPHERE_LABELS.A : SPHERE_LABELS.B;
    for (let k = 0; k < perSphere[s]; k++, i++) {
      const d = sampleDirection(distribution, k, perSphere[s], rand, jitter);
      const p = [c[0] + radius * d[0], c[1] + radius * d[1], c[2] + radius * d[2]];
      writeGaussian(gaussians, sh, i, p, iso, opacity, colors[s]);
      labels[i] = label;
    }
  }

  return {
    gaussians,
    sh,
    shDegree: 0,
    count: n,
    labels,
    centers,
    radius,
    bounds: sphereBounds(centers, radius),
    names: { [SPHERE_LABELS.A]: "esfera A", [SPHERE_LABELS.B]: "esfera B" },
  };
}

/**
 * A single axis-aligned gaussian, for unit/e2e tests (e.g. analytic depth checks).
 *
 * @param {{position?: number[], scale?: number|number[], opacity?: number, color?: number[]}} [options]
 * @returns {{gaussians: Float32Array, sh: Float32Array, shDegree: 0, count: 1, labels: Uint32Array,
 *   bounds: {min:number[], max:number[], center:number[], radius:number}}}
 */
export function makeSingleGaussian({
  position = [0, 0, 0],
  scale = 0.05,
  opacity = 1,
  color = [0.2, 0.9, 0.2],
} = {}) {
  assertVec3("position", position);
  const s = typeof scale === "number" ? [scale, scale, scale] : scale;
  assertVec3("scale", s);
  s.forEach((v) => assertPositive("scale", v));
  assertUnit("opacity", opacity);
  assertVec3("color", color);
  const gaussians = new Float32Array(GAUSSIAN_STRIDE);
  const sh = new Float32Array(SH_STRIDE);
  writeGaussian(gaussians, sh, 0, position, s, opacity, color);
  const r = Math.max(s[0], s[1], s[2]) * 3;
  return {
    gaussians,
    sh,
    shDegree: 0,
    count: 1,
    labels: new Uint32Array(1),
    bounds: {
      min: position.map((v) => v - r),
      max: position.map((v) => v + r),
      center: position.slice(),
      radius: r,
    },
  };
}
