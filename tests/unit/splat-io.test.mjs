/**
 * Unit tests for shared/splat-io.js (run: npm test).
 * Only node:test + node:assert; fixtures are built in memory by tests/helpers/make-ply.mjs.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import {
  detectFormat,
  describePly,
  toGaussianCloud,
  toSplat32,
  packedToPly,
  packedToSplat44,
  boundsFromGaussians,
  thinDiskScale,
  plyVariantFromProperties,
  THIN_DISK_RATIO,
  GAUSSIAN_STRIDE,
  SH_STRIDE,
  SPLAT32_ROW,
} from "../../shared/splat-io.js";

import {
  SH_C0,
  gaussianPropertyNames,
  makeGaussianPly,
  makePointCloudPly,
  makePly,
  makeSplat32,
  makeSplat44,
  makeRng,
  restPerColor,
} from "../helpers/make-ply.mjs";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const logit = (p) => Math.log(p / (1 - p));
const f32 = Math.fround;

function assertNear(actual, expected, eps, message = "") {
  const diff = Math.abs(actual - expected);
  assert.ok(
    diff <= eps,
    `${message} expected ${expected} got ${actual} (diff ${diff} > ${eps})`
  );
}

function assertNearRel(actual, expected, rel, message = "") {
  const eps = Math.abs(expected) * rel + 1e-12;
  assertNear(actual, expected, eps, message);
}

function gaussianAt(cloud, i) {
  const g = cloud.gaussians.subarray(i * GAUSSIAN_STRIDE, (i + 1) * GAUSSIAN_STRIDE);
  return {
    x: g[0], y: g[1], z: g[2], opacity: g[3],
    sx: g[4], sy: g[5], sz: g[6], pad: g[7],
    qw: g[8], qx: g[9], qy: g[10], qz: g[11],
  };
}

function shAt(cloud, i) {
  return cloud.sh.subarray(i * SH_STRIDE, (i + 1) * SH_STRIDE);
}

function packedRow(packed, i) {
  const f = new Float32Array(packed.buffer, packed.byteOffset + i * SPLAT32_ROW, 6);
  const u = packed.subarray(i * SPLAT32_ROW + 24, (i + 1) * SPLAT32_ROW);
  return { x: f[0], y: f[1], z: f[2], sx: f[3], sy: f[4], sz: f[5], rgba: [...u.subarray(0, 4)], quat: [...u.subarray(4, 8)] };
}

const SAMPLE_3DGS = [
  {
    x: 1.5, y: -2.25, z: 0.125,
    fDc: [0.4, -0.3, 1.1],
    opacity: logit(0.8),
    logScale: [Math.log(0.5), Math.log(0.25), Math.log(0.125)],
    rot: [2, 0, 0, 0],
  },
  {
    x: -3, y: 4, z: 5,
    fDc: [-1, 0, 0.5],
    opacity: 0,
    logScale: [Math.log(0.02), Math.log(0.03), Math.log(0.04)],
    rot: [1, 1, 1, 1],
  },
];

// ---------------------------------------------------------------------------
// detectFormat
// ---------------------------------------------------------------------------

describe("detectFormat", () => {
  test("PLY by magic bytes without a filename", () => {
    const buf = makeGaussianPly(SAMPLE_3DGS);
    assert.equal(detectFormat(buf), "ply");
    assert.equal(detectFormat(buf, "whatever.bin"), "ply");
  });

  test("PLY by .ply extension even without magic", () => {
    const buf = new Uint8Array(64).buffer;
    assert.equal(detectFormat(buf, "scene.PLY"), "ply");
  });

  test("32-byte and 44-byte .splat by size", () => {
    const row = { x: 0, y: 0, z: 0, scale: [1, 1, 1], rgba: [255, 0, 0, 255], quat: [1, 0, 0, 0] };
    assert.equal(detectFormat(makeSplat32([row, row, row]), "a.splat"), "splat32"); // 96 bytes
    assert.equal(detectFormat(makeSplat44([row]), "a.splat"), "splat44"); // 44 bytes
    // no filename at all: sizes still decide
    assert.equal(detectFormat(makeSplat32([row])), "splat32");
    assert.equal(detectFormat(makeSplat44([row])), "splat44");
  });

  test("ambiguous size (352 bytes) resolved by the 44-byte quaternion heuristic", () => {
    const row = { x: 10, y: 20, z: 30, scale: [1, 1, 1], rgba: [1, 2, 3, 4], quat: [0.5, 0.5, 0.5, 0.5] };
    const rows44 = Array.from({ length: 8 }, () => row); // 8 * 44 = 352 = 11 * 32
    assert.equal(detectFormat(makeSplat44(rows44), "x.splat"), "splat44");
    const rows32 = Array.from({ length: 11 }, () => row); // floats 7..10 of first 44 bytes are garbage
    assert.equal(detectFormat(makeSplat32(rows32), "x.splat"), "splat32");
  });

  test("accepts typed arrays as input", () => {
    const buf = makeGaussianPly(SAMPLE_3DGS);
    assert.equal(detectFormat(new Uint8Array(buf)), "ply");
  });

  test("accepts a SharedArrayBuffer as input", () => {
    const buf = makeGaussianPly(SAMPLE_3DGS);
    const shared = new SharedArrayBuffer(buf.byteLength);
    new Uint8Array(shared).set(new Uint8Array(buf));
    assert.equal(detectFormat(shared), "ply");
    assert.equal(toSplat32(shared, "s.ply").count, 2);
    assert.equal(toGaussianCloud(shared, "s.ply").count, 2);
  });

  test("throws on unrecognised sizes", () => {
    assert.throws(() => detectFormat(new ArrayBuffer(10)), /Unrecognized splat format/);
  });
});

// ---------------------------------------------------------------------------
// describePly / variant classification
// ---------------------------------------------------------------------------

describe("describePly", () => {
  test("3DGS SH3 with normals", () => {
    const rest = new Array(45).fill(0);
    const buf = makeGaussianPly(
      [{ x: 0, y: 0, z: 0, fRest: rest }],
      { shDegree: 3, normals: true }
    );
    const info = describePly(buf);
    assert.equal(info.vertexCount, 1);
    assert.deepEqual(info.properties, gaussianPropertyNames({ shDegree: 3, normals: true }));
    assert.equal(info.shDegree, 3);
    assert.equal(info.variant, "3dgs");
    assert.equal(info.encoding, "binary_le");
  });

  test("2DGS, point cloud and ascii encodings", () => {
    const two = describePly(makeGaussianPly([{ x: 0, y: 0, z: 0 }], { variant: "2dgs", normals: true }));
    assert.equal(two.variant, "2dgs");
    assert.ok(two.properties.includes("scale_1"));
    assert.ok(!two.properties.includes("scale_2"));

    const pc = describePly(makePointCloudPly([{ x: 0, y: 0, z: 0, rgb: [1, 2, 3] }], { encoding: "ascii" }));
    assert.equal(pc.variant, "pointcloud");
    assert.equal(pc.shDegree, 0);
    assert.equal(pc.encoding, "ascii");
    assert.deepEqual(pc.properties, ["x", "y", "z", "red", "green", "blue"]);
  });

  test("SH degree from f_rest count", () => {
    for (const deg of [0, 1, 2, 3]) {
      const fRest = new Array(3 * restPerColor(deg)).fill(0);
      const info = describePly(makeGaussianPly([{ x: 0, y: 0, z: 0, fRest }], { shDegree: deg }));
      assert.equal(info.shDegree, deg, `degree ${deg}`);
    }
  });

  test("rejects non-PLY buffers", () => {
    assert.throws(() => describePly(new ArrayBuffer(64)), /Not a PLY/);
  });

  test("plyVariantFromProperties edge cases", () => {
    assert.equal(plyVariantFromProperties(["x", "y", "z"]), "pointcloud");
    assert.equal(plyVariantFromProperties(["x", "scale_0", "scale_1"]), "2dgs");
    assert.equal(plyVariantFromProperties(["x", "scale_0", "scale_1", "scale_2"]), "3dgs");
    assert.equal(plyVariantFromProperties(["x", "scale_0"]), "3dgs"); // isotropic
  });
});

// ---------------------------------------------------------------------------
// toGaussianCloud on 3DGS PLYs
// ---------------------------------------------------------------------------

describe("toGaussianCloud 3DGS", () => {
  test("SH0 binary_le: layout, opacity sigmoid, exp scales, quaternion normalisation", () => {
    const cloud = toGaussianCloud(makeGaussianPly(SAMPLE_3DGS), "scene.ply");
    assert.equal(cloud.format, "ply");
    assert.equal(cloud.variant, "3dgs");
    assert.equal(cloud.shDegree, 0);
    assert.equal(cloud.count, 2);
    assert.equal(cloud.gaussians.length, 2 * GAUSSIAN_STRIDE);
    assert.equal(cloud.sh.length, 2 * SH_STRIDE);

    const g0 = gaussianAt(cloud, 0);
    assert.equal(g0.x, f32(1.5));
    assert.equal(g0.y, f32(-2.25));
    assert.equal(g0.z, f32(0.125));
    assertNear(g0.opacity, 0.8, 1e-6, "opacity sigmoid");
    assertNearRel(g0.sx, 0.5, 1e-6, "sx");
    assertNearRel(g0.sy, 0.25, 1e-6, "sy");
    assertNearRel(g0.sz, 0.125, 1e-6, "sz");
    assert.equal(g0.pad, 0);
    assert.deepEqual([g0.qw, g0.qx, g0.qy, g0.qz], [1, 0, 0, 0], "rot (2,0,0,0) normalises to identity");

    const g1 = gaussianAt(cloud, 1);
    assertNear(g1.opacity, 0.5, 1e-6, "logit 0 -> 0.5");
    for (const q of [g1.qw, g1.qx, g1.qy, g1.qz]) assertNear(q, 0.5, 1e-6, "rot (1,1,1,1)");
    assertNear(Math.hypot(g1.qw, g1.qx, g1.qy, g1.qz), 1, 1e-6, "unit quaternion");

    const sh0 = shAt(cloud, 0);
    assert.equal(sh0[0], f32(0.4));
    assert.equal(sh0[1], f32(-0.3));
    assert.equal(sh0[2], f32(1.1));
    for (let k = 3; k < SH_STRIDE; k++) assert.equal(sh0[k], 0, `sh[${k}] must be 0 for SH0`);
  });

  test("SH3: f_rest per-colour blocks land at sh[3 + 3k + c]", () => {
    const K = restPerColor(3); // 15
    // encode channel c, coefficient k as 100*c + k + 1 (all distinct, exactly representable)
    const fRest = [];
    for (let c = 0; c < 3; c++) for (let k = 0; k < K; k++) fRest.push(100 * c + k + 1);
    const cloud = toGaussianCloud(
      makeGaussianPly([{ x: 0, y: 0, z: 0, fDc: [7, 8, 9], fRest }], { shDegree: 3, normals: true })
    );
    assert.equal(cloud.shDegree, 3);
    const sh = shAt(cloud, 0);
    assert.deepEqual([sh[0], sh[1], sh[2]], [7, 8, 9]);
    for (let k = 0; k < K; k++) {
      for (let c = 0; c < 3; c++) {
        assert.equal(sh[3 + 3 * k + c], 100 * c + k + 1, `coefficient k=${k} channel ${c}`);
      }
    }
  });

  test("SH1 and SH2 leave the higher bands at zero", () => {
    for (const deg of [1, 2]) {
      const K = restPerColor(deg);
      const fRest = Array.from({ length: 3 * K }, (_, i) => i + 1);
      const cloud = toGaussianCloud(makeGaussianPly([{ x: 0, y: 0, z: 0, fRest }], { shDegree: deg }));
      assert.equal(cloud.shDegree, deg);
      const sh = shAt(cloud, 0);
      for (let k = 0; k < K; k++) {
        for (let c = 0; c < 3; c++) assert.equal(sh[3 + 3 * k + c], c * K + k + 1, `deg ${deg} k=${k} c=${c}`);
      }
      for (let i = 3 + 3 * K; i < SH_STRIDE; i++) assert.equal(sh[i], 0, `deg ${deg} sh[${i}]`);
    }
  });

  test("binary_be and ascii encodings decode to the same values", () => {
    const le = toGaussianCloud(makeGaussianPly(SAMPLE_3DGS, { encoding: "binary_le" }));
    const be = toGaussianCloud(makeGaussianPly(SAMPLE_3DGS, { encoding: "binary_be" }));
    const ascii = toGaussianCloud(makeGaussianPly(SAMPLE_3DGS, { encoding: "ascii" }));
    assert.deepEqual([...be.gaussians], [...le.gaussians]);
    assert.deepEqual([...be.sh], [...le.sh]);
    assert.equal(ascii.count, 2);
    assert.equal(ascii.variant, "3dgs");
    for (let i = 0; i < le.gaussians.length; i++) {
      assertNear(ascii.gaussians[i], le.gaussians[i], 1e-6, `ascii gaussians[${i}]`);
    }
    for (let i = 0; i < le.sh.length; i++) assertNear(ascii.sh[i], le.sh[i], 1e-6, `ascii sh[${i}]`);
  });

  test("missing rot_* gives identity, missing opacity gives 1", () => {
    const buf = makeGaussianPly([{ x: 1, y: 2, z: 3 }], { rotation: false, opacity: false });
    const info = describePly(buf);
    assert.ok(!info.properties.includes("rot_0"));
    assert.ok(!info.properties.includes("opacity"));
    const g = gaussianAt(toGaussianCloud(buf), 0);
    assert.deepEqual([g.qw, g.qx, g.qy, g.qz], [1, 0, 0, 0]);
    assert.equal(g.opacity, 1);

    const row = packedRow(toSplat32(buf).packed, 0);
    assert.deepEqual(row.quat, [255, 128, 128, 128]);
    assert.equal(row.rgba[3], 255);
  });

  test("accepts a Uint8Array view and a truncated body throws a clear error", () => {
    const buf = makeGaussianPly(SAMPLE_3DGS);
    assert.equal(toGaussianCloud(new Uint8Array(buf)).count, 2);
    const truncated = buf.slice(0, buf.byteLength - 4);
    assert.throws(() => toGaussianCloud(truncated), /PLY body truncated/);
    assert.throws(() => toSplat32(truncated), /PLY body truncated/);
    const asciiBuf = makeGaussianPly(SAMPLE_3DGS, { encoding: "ascii" });
    const text = new TextDecoder().decode(asciiBuf);
    const oneRow = new TextEncoder().encode(text.split("\n").slice(0, -2).join("\n") + "\n").buffer;
    assert.throws(() => toGaussianCloud(oneRow), /ASCII body truncated/);
  });
});

// ---------------------------------------------------------------------------
// 2DGS
// ---------------------------------------------------------------------------

describe("2DGS PLY (scale_0, scale_1 only)", () => {
  const records = [
    { x: 0.5, y: 1, z: -1, fDc: [0.1, 0.2, 0.3], opacity: logit(0.9), logScale: [Math.log(0.2), Math.log(0.05)], rot: [0, 1, 0, 0] },
    { x: 2, y: 2, z: 2, fDc: [0, 0, 0], opacity: logit(0.25), logScale: [Math.log(0.01), Math.log(0.03)], rot: [1, 0, 0, 0] },
  ];
  const buf = makeGaussianPly(records, { variant: "2dgs", normals: true });

  test("thinDiskScale is relative to the smaller disk axis", () => {
    assertNearRel(thinDiskScale(0.2, 0.05), 0.05 * THIN_DISK_RATIO, 1e-12);
    assertNearRel(thinDiskScale(0.05, 0.2), 0.05 * THIN_DISK_RATIO, 1e-12);
    assert.ok(thinDiskScale(0, 0) > 0, "never zero");
  });

  test("float cloud: variant 2dgs, third scale synthesised as a thin slab", () => {
    const cloud = toGaussianCloud(buf, "disk.ply");
    assert.equal(cloud.variant, "2dgs");
    assert.equal(cloud.format, "ply");
    assert.equal(cloud.count, 2);
    const g0 = gaussianAt(cloud, 0);
    assertNearRel(g0.sx, 0.2, 1e-6);
    assertNearRel(g0.sy, 0.05, 1e-6);
    assertNearRel(g0.sz, 0.05 * THIN_DISK_RATIO, 1e-5, "sz thin");
    assert.ok(g0.sz > 0 && g0.sz < 1e-3 * Math.min(g0.sx, g0.sy));
    assertNear(g0.opacity, 0.9, 1e-6);
    assert.deepEqual([g0.qw, g0.qx, g0.qy, g0.qz], [0, 1, 0, 0]);
    const g1 = gaussianAt(cloud, 1);
    assertNearRel(g1.sz, 0.01 * THIN_DISK_RATIO, 1e-5, "sz uses min(sx, sy)");
  });

  test("packed path: same thin third scale and variant", () => {
    const parsed = toSplat32(buf, "disk.ply");
    assert.equal(parsed.variant, "2dgs");
    assert.equal(parsed.format, "ply");
    assert.equal(parsed.count, 2);
    // rows are sorted by volume*opacity; record 0 is the bigger one
    const row = packedRow(parsed.packed, 0);
    assert.equal(row.x, f32(0.5));
    assertNearRel(row.sx, 0.2, 1e-6);
    assertNearRel(row.sy, 0.05, 1e-6);
    assertNearRel(row.sz, 0.05 * THIN_DISK_RATIO, 1e-5);
    // sigmoid(fround(logit(0.9))) * 255 sits on a .5 boundary: allow one byte of rounding
    assertNear(row.rgba[3], 0.9 * 255, 1, "alpha byte");
  });

  test("2DGS without rot_* and without opacity still loads", () => {
    const noRot = makeGaussianPly([records[0]], { variant: "2dgs", rotation: false, opacity: false });
    const cloud = toGaussianCloud(noRot);
    assert.equal(cloud.variant, "2dgs");
    const g = gaussianAt(cloud, 0);
    assert.deepEqual([g.qw, g.qx, g.qy, g.qz], [1, 0, 0, 0]);
    assert.equal(g.opacity, 1);
  });
});

// ---------------------------------------------------------------------------
// degenerate rows (non-finite log-scales from diverged gaussians)
// ---------------------------------------------------------------------------

describe("non-finite scale_* rows are hidden identically by both parsers", () => {
  const names = gaussianPropertyNames();
  const properties = names.map((name) => ({ name, type: "float" }));
  const record = (over) => {
    const v = {
      x: 1, y: 2, z: 3, f_dc_0: 0.5, f_dc_1: 0.5, f_dc_2: 0.5, opacity: logit(0.95),
      scale_0: Math.log(0.1), scale_1: Math.log(0.1), scale_2: Math.log(0.1),
      rot_0: 1, rot_1: 0, rot_2: 0, rot_3: 0,
      ...over,
    };
    return names.map((n) => v[n]);
  };
  const healthy = record({ x: 7, scale_0: Math.log(0.05), scale_1: Math.log(0.05), scale_2: Math.log(0.05) });

  const cases = [
    ["NaN log-scale", { scale_0: NaN }],
    ["+Infinity log-scale", { scale_1: Infinity }],
    ["log-scale that overflows float32 after exp", { scale_2: 100 }],
  ];

  for (const [label, over] of cases) {
    for (const encoding of ["binary_le", "ascii"]) {
      test(`${label} (${encoding}): scales 0, opacity 0, sorted last, sibling untouched`, () => {
        // degenerate row first so the packed sort has to move it behind the healthy one
        const buf = makePly({ properties, rows: [record(over), healthy], encoding });

        const cloud = toGaussianCloud(buf, "diverged.ply");
        assert.equal(cloud.count, 2);
        const bad = gaussianAt(cloud, 0);
        assert.deepEqual([bad.sx, bad.sy, bad.sz], [0, 0, 0], "float scales");
        assert.equal(bad.opacity, 0, "float opacity");
        assert.equal(bad.x, 1);
        const good = gaussianAt(cloud, 1);
        assertNearRel(good.sx, 0.05, 1e-6);
        assertNear(good.opacity, 0.95, 1e-6);
        for (const v of cloud.gaussians) assert.ok(Number.isFinite(v), "no NaN/Infinity in the float cloud");

        const parsed = toSplat32(buf, "diverged.ply");
        assert.equal(parsed.count, 2);
        const first = packedRow(parsed.packed, 0);
        assert.equal(first.x, 7, "healthy row sorts first");
        assertNearRel(first.sx, 0.05, 1e-6);
        assertNear(first.rgba[3], 0.95 * 255, 1);
        const hidden = packedRow(parsed.packed, 1);
        assert.equal(hidden.x, 1);
        assert.deepEqual([hidden.sx, hidden.sy, hidden.sz], [0, 0, 0], "packed scales");
        assert.equal(hidden.rgba[3], 0, "packed alpha byte");
        assert.deepEqual(hidden.rgba.slice(0, 3), packedRow(parsed.packed, 0).rgba.slice(0, 3), "colour still decoded");
      });
    }
  }

  test("2DGS: a NaN in either disk axis hides the row", () => {
    const twoNames = gaussianPropertyNames({ variant: "2dgs" });
    const twoProps = twoNames.map((name) => ({ name, type: "float" }));
    const row = twoNames.map((n) => ({
      x: 0, y: 0, z: 0, f_dc_0: 0, f_dc_1: 0, f_dc_2: 0, opacity: 2,
      scale_0: Math.log(0.2), scale_1: NaN, rot_0: 1, rot_1: 0, rot_2: 0, rot_3: 0,
    }[n]));
    const buf = makePly({ properties: twoProps, rows: [row] });
    const g = gaussianAt(toGaussianCloud(buf), 0);
    assert.deepEqual([g.sx, g.sy, g.sz, g.opacity], [0, 0, 0, 0]);
    assert.equal(packedRow(toSplat32(buf).packed, 0).rgba[3], 0);
  });
});

// ---------------------------------------------------------------------------
// point clouds
// ---------------------------------------------------------------------------

describe("point-cloud PLY (red/green/blue, no scale_*)", () => {
  const points = [
    { x: 0, y: 0, z: 0, rgb: [255, 0, 0] },
    { x: 1, y: 0, z: 0, rgb: [0, 128, 0] },
    { x: 0, y: 1, z: 0, rgb: [1, 1, 1] },
  ];

  for (const encoding of ["binary_le", "ascii"]) {
    test(`float cloud (${encoding}): DC = (c/255 - 0.5) / SH_C0, fixed scale, identity rotation`, () => {
      const cloud = toGaussianCloud(makePointCloudPly(points, { encoding }));
      assert.equal(cloud.variant, "pointcloud");
      assert.equal(cloud.shDegree, 0);
      assert.equal(cloud.count, 3);
      points.forEach((p, i) => {
        const g = gaussianAt(cloud, i);
        assert.equal(g.x, p.x); assert.equal(g.y, p.y); assert.equal(g.z, p.z);
        assert.equal(g.opacity, 1);
        assertNear(g.sx, 0.01, 1e-9); assertNear(g.sy, 0.01, 1e-9); assertNear(g.sz, 0.01, 1e-9);
        assert.deepEqual([g.qw, g.qx, g.qy, g.qz], [1, 0, 0, 0]);
        const sh = shAt(cloud, i);
        for (let c = 0; c < 3; c++) {
          assertNear(sh[c], (p.rgb[c] / 255 - 0.5) / SH_C0, 1e-6, `point ${i} channel ${c}`);
        }
      });
    });
  }

  test("packed path keeps the exact bytes (uchar 1 is not treated as 1.0)", () => {
    const parsed = toSplat32(makePointCloudPly(points), "cloud.ply");
    assert.equal(parsed.variant, "pointcloud");
    for (let i = 0; i < points.length; i++) {
      const row = packedRow(parsed.packed, i);
      assert.deepEqual(row.rgba, [...points[i].rgb, 255], `point ${i}`);
      assert.equal(row.x, points[i].x);
      assertNear(row.sx, 0.01, 1e-9);
      assert.deepEqual(row.quat, [255, 128, 128, 128]);
    }
  });

  test("float red/green/blue in [0,1] are taken as-is", () => {
    const properties = ["x", "y", "z", "red", "green", "blue"].map((name) => ({ name, type: "float" }));
    const buf = makePly({ properties, rows: [[0, 0, 0, 0.5, 1, 0]] });
    const sh = shAt(toGaussianCloud(buf), 0);
    assertNear(sh[0], (0.5 - 0.5) / SH_C0, 1e-6);
    assertNear(sh[1], (1 - 0.5) / SH_C0, 1e-6);
    assertNear(sh[2], (0 - 0.5) / SH_C0, 1e-6);
  });
});

// ---------------------------------------------------------------------------
// 32-byte / 44-byte splat paths and round trips
// ---------------------------------------------------------------------------

describe("splat32 / splat44", () => {
  const rows = [
    { x: 1, y: 2, z: 3, scale: [0.5, 0.25, 0.125], rgba: [200, 100, 50, 255], quat: [1, 0, 0, 0] },
    { x: -1, y: 0.5, z: 8, scale: [0.02, 0.02, 0.02], rgba: [0, 0, 0, 51], quat: [0.5, 0.5, 0.5, 0.5] },
  ];

  test("toGaussianCloud on a 32-byte .splat", () => {
    const cloud = toGaussianCloud(makeSplat32(rows), "model.splat");
    assert.equal(cloud.format, "splat32");
    assert.equal(cloud.variant, "3dgs");
    assert.equal(cloud.shDegree, 0);
    assert.equal(cloud.count, 2);
    const g0 = gaussianAt(cloud, 0);
    assert.deepEqual([g0.x, g0.y, g0.z], [1, 2, 3]);
    assert.deepEqual([g0.sx, g0.sy, g0.sz], [0.5, 0.25, 0.125]);
    assert.equal(g0.opacity, 1);
    assertNear(g0.qw, 1, 1 / 128); assertNear(g0.qx, 0, 1 / 128);
    const sh0 = shAt(cloud, 0);
    assertNear(sh0[0], (200 / 255 - 0.5) / SH_C0, 1e-6);
    assertNear(sh0[1], (100 / 255 - 0.5) / SH_C0, 1e-6);
    assertNear(sh0[2], (50 / 255 - 0.5) / SH_C0, 1e-6);
    const g1 = gaussianAt(cloud, 1);
    assertNear(g1.opacity, 51 / 255, 1e-6);
    for (const q of [g1.qw, g1.qx, g1.qy, g1.qz]) assertNear(q, 0.5, 1 / 128);
  });

  test("toGaussianCloud on a 44-byte .splat matches the 32-byte result", () => {
    const c32 = toGaussianCloud(makeSplat32(rows), "a.splat");
    const c44 = toGaussianCloud(makeSplat44(rows), "a.splat");
    assert.equal(c44.format, "splat44");
    assert.equal(c44.count, 2);
    for (let i = 0; i < c32.gaussians.length; i++) {
      assertNear(c44.gaussians[i], c32.gaussians[i], 1 / 128, `gaussians[${i}]`);
    }
    assert.deepEqual([...c44.sh], [...c32.sh]);
  });

  test("packedToSplat44 keeps floats/bytes and unpacks the quaternion", () => {
    const packed = toSplat32(makeSplat32(rows), "a.splat").packed;
    const out = packedToSplat44(packed);
    assert.equal(out.byteLength, rows.length * 44);
    const f = new Float32Array(out);
    const u = new Uint8Array(out);
    rows.forEach((row, i) => {
      const o = i * 11;
      assert.deepEqual([f[o], f[o + 1], f[o + 2]], [row.x, row.y, row.z]);
      assert.deepEqual([f[o + 3], f[o + 4], f[o + 5]], row.scale.map(f32));
      assert.deepEqual([...u.subarray(i * 44 + 24, i * 44 + 28)], row.rgba);
      const n = Math.hypot(...row.quat);
      for (let c = 0; c < 4; c++) assertNear(f[o + 7 + c], row.quat[c] / n, 1 / 128, `row ${i} quat ${c}`);
    });
    assert.equal(detectFormat(out, "a.splat"), "splat44");
    const back = toGaussianCloud(out, "a.splat");
    assert.equal(back.count, rows.length);
  });

  test("round trip: PLY -> toSplat32 -> packedToPly -> toGaussianCloud within quantisation", () => {
    // decreasing volume*opacity so the packed sort keeps the record order
    const rng = makeRng(7);
    const records = [];
    for (let i = 0; i < 6; i++) {
      const s = 0.5 / (i + 1);
      records.push({
        x: i * 1.25, y: rng() * 4 - 2, z: rng() * 4 - 2,
        fDc: [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1],
        opacity: logit(0.95 - i * 0.1),
        logScale: [Math.log(s), Math.log(s * 0.8), Math.log(s * 0.6)],
        rot: [rng(), rng() - 0.5, rng() - 0.5, rng() - 0.5],
      });
    }
    const original = toGaussianCloud(makeGaussianPly(records), "orig.ply");
    const packed = toSplat32(makeGaussianPly(records), "orig.ply");
    assert.equal(packed.count, records.length);
    for (let i = 0; i < records.length; i++) {
      assert.equal(packedRow(packed.packed, i).x, f32(records[i].x), "sort keeps decreasing-volume order");
    }

    const plyBytes = packedToPly(packed.packed);
    const info = describePly(plyBytes.buffer);
    assert.equal(info.vertexCount, records.length);
    assert.equal(info.variant, "3dgs");
    assert.equal(info.shDegree, 0);

    const round = toGaussianCloud(plyBytes.buffer, "round.ply");
    assert.equal(round.count, records.length);
    const colourTol = 0.5 / 255 / SH_C0 + 1e-5;
    for (let i = 0; i < records.length; i++) {
      const a = gaussianAt(original, i);
      const b = gaussianAt(round, i);
      assert.deepEqual([b.x, b.y, b.z], [a.x, a.y, a.z], `position ${i}`);
      assertNearRel(b.sx, a.sx, 1e-5, `sx ${i}`);
      assertNearRel(b.sy, a.sy, 1e-5, `sy ${i}`);
      assertNearRel(b.sz, a.sz, 1e-5, `sz ${i}`);
      assertNear(b.opacity, a.opacity, 0.5 / 255 + 1e-4, `opacity ${i}`);
      for (const k of ["qw", "qx", "qy", "qz"]) assertNear(b[k], a[k], 1 / 128 + 1e-6, `${k} ${i}`);
      const shA = shAt(original, i);
      const shB = shAt(round, i);
      for (let c = 0; c < 3; c++) assertNear(shB[c], shA[c], colourTol, `sh dc ${i}.${c}`);
    }
  });

  test("toSplat32 sorts PLY rows by volume x opacity (largest first)", () => {
    const small = { x: 1, y: 0, z: 0, logScale: [Math.log(0.01), Math.log(0.01), Math.log(0.01)], opacity: logit(0.9) };
    const big = { x: 2, y: 0, z: 0, logScale: [Math.log(0.5), Math.log(0.5), Math.log(0.5)], opacity: logit(0.9) };
    const parsed = toSplat32(makeGaussianPly([small, big]));
    assert.equal(packedRow(parsed.packed, 0).x, 2);
    assert.equal(packedRow(parsed.packed, 1).x, 1);
  });
});

// ---------------------------------------------------------------------------
// bounds and downsampling
// ---------------------------------------------------------------------------

describe("boundsFromGaussians", () => {
  test("small clouds use the exact extents", () => {
    const g = new Float32Array(3 * GAUSSIAN_STRIDE);
    const pts = [[-1, 0, 2], [3, 4, -6], [0, 0, 0]];
    pts.forEach((p, i) => g.set(p, i * GAUSSIAN_STRIDE));
    const b = boundsFromGaussians(g);
    assert.deepEqual(b.min, [-1, 0, -6]);
    assert.deepEqual(b.max, [3, 4, 2]);
    assert.deepEqual(b.center, [1, 2, -2]);
    assert.equal(b.radius, 4);
  });

  test("large clouds: min/max are exact, centre/radius ignore the 5% tails", () => {
    const n = 100;
    const g = new Float32Array(n * GAUSSIAN_STRIDE);
    for (let i = 0; i < n; i++) g.set([i, 0, 0], i * GAUSSIAN_STRIDE); // x = 0..99
    g[0] = -1000; // outlier at index 0
    g[(n - 1) * GAUSSIAN_STRIDE] = 1000; // outlier at the end
    const b = boundsFromGaussians(g);
    assert.equal(b.min[0], -1000);
    assert.equal(b.max[0], 1000);
    assert.ok(b.radius < 60, `radius ${b.radius} should ignore outliers`);
    assert.ok(Math.abs(b.center[0] - 49.5) < 6, `center ${b.center[0]}`);
  });

  test("empty input gives the unit fallback", () => {
    const b = boundsFromGaussians(new Float32Array(0));
    assert.deepEqual(b.min, [-1, -1, -1]);
    assert.deepEqual(b.max, [1, 1, 1]);
    assert.deepEqual(b.center, [0, 0, 0]);
    assert.equal(b.radius, 1);
  });

  test("toSplat32 bounds follow the packed positions", () => {
    const rows = [
      { x: -2, y: 1, z: 0, scale: [1, 1, 1], rgba: [0, 0, 0, 255] },
      { x: 4, y: -3, z: 6, scale: [1, 1, 1], rgba: [0, 0, 0, 255] },
    ];
    const b = toSplat32(makeSplat32(rows), "a.splat").bounds;
    assert.deepEqual(b.min, [-2, -3, 0]);
    assert.deepEqual(b.max, [4, 1, 6]);
    assert.deepEqual(b.center, [1, -1, 3]);
    assert.equal(b.radius, 3);
  });
});

describe("toSplat32 compression option", () => {
  const rows = Array.from({ length: 10 }, (_, i) => ({
    x: i, y: 0, z: 0, scale: [1, 1, 1], rgba: [i, 0, 0, 255], quat: [1, 0, 0, 0],
  }));
  const buf = makeSplat32(rows);

  test("factor 1 (or absent/zero) keeps every row", () => {
    assert.equal(toSplat32(buf, "a.splat").count, 10);
    assert.equal(toSplat32(buf, "a.splat", { compression: 1 }).count, 10);
    assert.equal(toSplat32(buf, "a.splat", { compression: 0 }).count, 10);
  });

  test("factor 2 keeps every other row, evenly spaced", () => {
    const parsed = toSplat32(buf, "a.splat", { compression: 2 });
    assert.equal(parsed.count, 5);
    assert.equal(parsed.packed.byteLength, 5 * SPLAT32_ROW);
    for (let i = 0; i < 5; i++) assert.equal(packedRow(parsed.packed, i).x, 2 * i);
  });

  test("factor is clamped to 10 and never drops below one row", () => {
    assert.equal(toSplat32(buf, "a.splat", { compression: 10 }).count, 1);
    assert.equal(toSplat32(buf, "a.splat", { compression: 100 }).count, 1);
    const two = makeSplat32(rows.slice(0, 2));
    assert.equal(toSplat32(two, "a.splat", { compression: 5 }).count, 1);
  });

  test("applies to PLY input as well", () => {
    const records = Array.from({ length: 8 }, (_, i) => ({ x: i, y: 0, z: 0 }));
    assert.equal(toSplat32(makeGaussianPly(records), "s.ply", { compression: 4 }).count, 2);
  });
});
