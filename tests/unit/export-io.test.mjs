/**
 * Unit tests for shared/export-io.js (plan F5 encoders) against the decoders in
 * shared/splat-io.js. Run: npm test.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import { encodePly, encodeSplat32, exportFileName } from "../../shared/export-io.js";
import { describePly, readPlyColumns, toGaussianCloud } from "../../shared/splat-io.js";
import { makeTwoSpheres } from "../../shared/synthetic.js";

const near = (a, b, eps) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b} (±${eps})`);

describe("encodeSplat32", () => {
  test("round-trips positions, scales, opacity, colour and rotation through toGaussianCloud", () => {
    const scene = makeTwoSpheres();
    const bytes = encodeSplat32(scene);
    assert.equal(bytes.length, scene.count * 32);
    const back = toGaussianCloud(bytes.buffer, "x.splat");
    assert.equal(back.count, scene.count);
    for (const i of [0, 7, 1999, 2000, 3999]) {
      for (let k = 0; k < 3; k++) near(back.gaussians[i * 12 + k], scene.gaussians[i * 12 + k], 1e-6);
      for (let k = 4; k < 7; k++) near(back.gaussians[i * 12 + k], scene.gaussians[i * 12 + k], 1e-6);
      near(back.gaussians[i * 12 + 3], scene.gaussians[i * 12 + 3], 1 / 255);
      for (let k = 0; k < 3; k++) near(back.sh[i * 48 + k], scene.sh[i * 48 + k], 0.01);
      for (let k = 8; k < 12; k++) near(back.gaussians[i * 12 + k], scene.gaussians[i * 12 + k], 0.01);
    }
  });
});

describe("encodePly", () => {
  test("3DGS PLY with instance_id/class_id/confidence round-trips and keeps SH degree", () => {
    const scene = makeTwoSpheres();
    const labels = scene.labels;
    const classIds = Uint32Array.from(labels, (l) => l * 10);
    const confidences = Float32Array.from(labels, (l) => (l ? 0.5 + l / 10 : 0));
    const buf = encodePly(scene, { labels, classIds, confidences, comment: "prueba F5" });
    const info = describePly(buf);
    assert.equal(info.vertexCount, scene.count);
    assert.equal(info.shDegree, 0);
    assert.equal(info.encoding, "binary_le");
    assert.ok(info.properties.includes("instance_id") && info.properties.includes("class_id") && info.properties.includes("confidence"));
    const back = toGaussianCloud(buf, "x.ply");
    assert.equal(back.count, scene.count);
    for (const i of [0, 1234, 3999]) {
      for (let k = 0; k < 3; k++) near(back.gaussians[i * 12 + k], scene.gaussians[i * 12 + k], 1e-6);
      near(back.gaussians[i * 12 + 3], scene.gaussians[i * 12 + 3], 1e-5);
      for (let k = 4; k < 7; k++) near(back.gaussians[i * 12 + k], scene.gaussians[i * 12 + k], 1e-6);
      for (let k = 8; k < 12; k++) near(back.gaussians[i * 12 + k], scene.gaussians[i * 12 + k], 1e-6);
      for (let k = 0; k < 3; k++) near(back.sh[i * 48 + k], scene.sh[i * 48 + k], 1e-6);
    }
    const cols = readPlyColumns(buf, ["instance_id", "class_id", "confidence", "missing"]);
    assert.deepEqual(Object.keys(cols).sort(), ["class_id", "confidence", "instance_id"]);
    assert.ok(cols.instance_id instanceof Uint32Array, "ids come back as u32 although stored as float");
    assert.ok(cols.confidence instanceof Float32Array);
    assert.deepEqual(Array.from(cols.instance_id.subarray(0, 3)), Array.from(labels.subarray(0, 3)));
    assert.equal(cols.instance_id[3999], labels[3999]);
    assert.equal(cols.class_id[0], labels[0] * 10);
    near(cols.confidence[0], 0.5 + labels[0] / 10, 1e-6);
    assert.throws(() => encodePly(scene, { labels: new Uint32Array(3) }), /un valor por gaussiana/);
  });

  test("SH degree 3 coefficients survive (colour-major f_rest layout)", () => {
    const count = 2;
    const gaussians = new Float32Array(count * 12);
    const sh = new Float32Array(count * 48);
    for (let i = 0; i < count; i++) {
      gaussians.set([i, 0, 0, 0.7, 0.1, 0.1, 0.1, 0, 1, 0, 0, 0], i * 12);
      for (let k = 0; k < 48; k++) sh[i * 48 + k] = (i + 1) * 0.01 * k;
    }
    const buf = encodePly({ gaussians, sh, shDegree: 3, count });
    const info = describePly(buf);
    assert.equal(info.shDegree, 3);
    assert.equal(info.properties.filter((p) => p.startsWith("f_rest_")).length, 45);
    const back = toGaussianCloud(buf, "x.ply");
    for (let i = 0; i < count; i++) for (let k = 0; k < 48; k++) near(back.sh[i * 48 + k], sh[i * 48 + k], 1e-6);
  });

  test("exportFileName", () => {
    assert.equal(exportFileName("alarm_clock_generated.splat", 3, "spz"), "alarm_clock_generated_instancia-3.spz");
    assert.equal(exportFileName("model.splat", null, "ply"), "model_escena.ply");
    assert.equal(exportFileName("", null, "splat"), "escena_escena.splat");
  });
});
