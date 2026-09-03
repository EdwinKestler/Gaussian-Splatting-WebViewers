/**
 * Unit tests for shared/edit-ops.js (plan F5: ops.jsonl, replay, undo, baking). Run: npm test.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import {
  EditLog, EditSession, bakeSession, composeTransform, indicesFromRanges, mat4Multiply, mat4RotationAxis,
  mat4Translation, opsFromJsonl, opsToJsonl, rangesFromIndices, replay, rotationQuatFromMat4, sessionFingerprint,
  transformPoint,
} from "../../shared/edit-ops.js";
import { makeTwoSpheres } from "../../shared/synthetic.js";

const near = (a, b, eps = 1e-5) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

function tinyCloud() {
  // 4 gaussians on the x axis, labels [1,1,2,0]
  const gaussians = new Float32Array(4 * 12);
  for (let i = 0; i < 4; i++) {
    gaussians.set([i, 0, 0, 1, 0.1, 0.2, 0.3, 0, 1, 0, 0, 0], i * 12);
  }
  const sh = new Float32Array(4 * 48);
  for (let i = 0; i < 4; i++) sh[i * 48] = i;
  return { gaussians, sh, shDegree: 0, labels: new Uint32Array([1, 1, 2, 0]), names: { 1: "a", 2: "b" } };
}

describe("matrices", () => {
  test("composeTransform rotates about the pivot and then translates", () => {
    const m = composeTransform({ translate: [1, 0, 0], rotateAxis: [0, 0, 1], rotateDeg: 90, scale: 2, pivot: [1, 1, 0] });
    // pivot maps to pivot + translate
    const p = transformPoint(m, [1, 1, 0]);
    near(p[0], 2); near(p[1], 1); near(p[2], 0);
    // a point 1 unit +x from the pivot: scaled to 2, rotated 90° → +y, then translated
    const q = transformPoint(m, [2, 1, 0]);
    near(q[0], 2); near(q[1], 3); near(q[2], 0);
  });
  test("rotationQuatFromMat4 recovers rotation and scale", () => {
    const m = mat4Multiply(mat4RotationAxis([0, 1, 0], 90), mat4Multiply(mat4Translation([0, 0, 0]), new Float32Array([3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1])));
    const { quat, scale } = rotationQuatFromMat4(m);
    near(scale[0], 3); near(scale[1], 3); near(scale[2], 3);
    near(Math.abs(quat[0]), Math.SQRT1_2); near(Math.abs(quat[2]), Math.SQRT1_2);
  });
});

describe("ranges", () => {
  test("rangesFromIndices / indicesFromRanges round-trip", () => {
    const r = rangesFromIndices([5, 1, 2, 3, 9, 9, 10]);
    assert.deepEqual(r, [[1, 3], [5, 5], [9, 10]]);
    assert.deepEqual(Array.from(indicesFromRanges(r)), [1, 2, 3, 5, 9, 10]);
    assert.deepEqual(rangesFromIndices([]), []);
  });
});

describe("EditSession ops", () => {
  test("asignar, transformar, borrar, fusionar, renombrar mutate the state as specified", () => {
    const s = new EditSession(tinyCloud());
    assert.deepEqual(s.labelSet(), [1, 2]);
    assert.equal(s.nextLabel(), 3);
    s.apply({ op: "asignar", id_instancia: 3, rangos: [[3, 3]] });
    assert.deepEqual(Array.from(s.labels), [1, 1, 2, 3]);
    s.apply({ op: "transformar", id_instancia: 1, xform: Array.from(mat4Translation([0, 5, 0])) });
    assert.ok(s.xforms.has(1));
    s.apply({ op: "transformar", id_instancia: 1, xform: Array.from(mat4Translation([0, 0, 0])) });
    assert.ok(!s.xforms.has(1), "identity removes the transform");
    s.apply({ op: "borrar", id_instancia: 2 });
    assert.ok(s.deleted.has(2));
    s.apply({ op: "restaurar", id_instancia: 2 });
    assert.ok(!s.deleted.has(2));
    s.apply({ op: "fusionar", origen: 3, destino: 2 });
    assert.deepEqual(Array.from(s.labels), [1, 1, 2, 2]);
    s.apply({ op: "renombrar", id_instancia: 2, nombre_es: "mesa" });
    assert.equal(s.names.get(2), "mesa");
    assert.throws(() => s.apply({ op: "asignar", id_instancia: 1, rangos: [[9, 9]] }), /fuera de rango/);
    assert.throws(() => s.apply({ op: "volar" }), /desconocida/);
    assert.throws(() => s.apply({ op: "transformar", id_instancia: 1, xform: [1, 2] }), /16/);
  });

  test("duplicar appends copies with a new label and keeps origen (invariant index)", () => {
    const s = new EditSession(tinyCloud());
    const r = s.apply({ op: "duplicar", id_instancia: 1, nueva: 4, xform: Array.from(mat4Translation([0, 0, 1])) });
    assert.deepEqual(r, { label: 4, count: 2, start: 4 });
    assert.equal(s.count, 6);
    assert.deepEqual(Array.from(s.labels), [1, 1, 2, 0, 4, 4]);
    assert.deepEqual(Array.from(s.origen), [0, 1, 2, 3, 0, 1]);
    assert.equal(s.gaussians[5 * 12], 1, "copied gaussian keeps its position (transform is in xforms)");
    assert.equal(s.sh[5 * 48], 1);
    assert.equal(s.names.get(4), "a (copia)");
    assert.ok(s.xforms.has(4));
  });
});

describe("replay / EditLog", () => {
  test("replaying ops.jsonl reproduces the same state; undo/redo rebuild it", () => {
    const base = tinyCloud();
    const log = new EditLog(base);
    log.push({ op: "asignar", id_instancia: 5, rangos: rangesFromIndices([3]) });
    log.push({ op: "transformar", id_instancia: 5, xform: Array.from(composeTransform({ translate: [1, 2, 3] })) });
    log.push({ op: "duplicar", id_instancia: 5, nueva: 6 });
    const fp = sessionFingerprint(log.session);
    const jsonl = log.toJsonl();
    assert.equal(jsonl.split("\n").filter(Boolean).length, 3);
    const ops = opsFromJsonl(jsonl);
    assert.equal(ops[0].op, "asignar");
    assert.ok(ops[0].fecha, "ops are stamped");
    const again = replay(base, ops);
    assert.equal(sessionFingerprint(again), fp);
    assert.deepEqual(Array.from(again.labels), Array.from(log.session.labels));
    assert.equal(opsToJsonl([]), "");
    assert.throws(() => opsFromJsonl('{"op":"nada"}'), /desconocida/);

    const undone = log.undo();
    assert.equal(undone.op, "duplicar");
    assert.equal(log.session.count, 4);
    assert.notEqual(sessionFingerprint(log.session), fp);
    log.redo();
    assert.equal(sessionFingerprint(log.session), fp);
    log.undo(); log.undo(); log.undo();
    assert.equal(log.undo(), null);
    assert.deepEqual(Array.from(log.session.labels), [1, 1, 2, 0]);
    log.push({ op: "borrar", id_instancia: 1 });
    assert.equal(log.redo(), null, "a new op clears the redo stack");
  });
});

describe("bakeSession", () => {
  test("applies transforms to centre, scale and rotation; drops deleted; selects one instance", () => {
    const base = tinyCloud();
    const s = new EditSession(base);
    const m = composeTransform({ translate: [10, 0, 0], rotateAxis: [0, 0, 1], rotateDeg: 90, scale: 2, pivot: [0, 0, 0] });
    s.apply({ op: "transformar", id_instancia: 1, xform: Array.from(m) });
    s.apply({ op: "borrar", id_instancia: 2 });
    const scene = bakeSession(s);
    assert.equal(scene.count, 3, "instance 2 dropped, fondo kept");
    assert.deepEqual(Array.from(scene.labels), [1, 1, 0]);
    // gaussian 1 (x=1) → scaled 2 → rotated → (0,2,0) → translated → (10,2,0)
    near(scene.gaussians[12], 10); near(scene.gaussians[13], 2); near(scene.gaussians[14], 0);
    near(scene.gaussians[16], 0.2); near(scene.gaussians[17], 0.4); near(scene.gaussians[18], 0.6);
    const q = scene.gaussians.subarray(20, 24);
    near(Math.abs(q[0]), Math.cos(Math.PI / 4)); near(Math.abs(q[3]), Math.sin(Math.PI / 4));
    assert.equal(scene.origen[2], 3);
    const one = bakeSession(s, { label: 1 });
    assert.equal(one.count, 2);
    assert.deepEqual(Array.from(one.labels), [1, 1]);
    const noBg = bakeSession(s, { includeBackground: false });
    assert.equal(noBg.count, 2);
    const hidden = bakeSession(s, { hidden: new Set([1]) });
    assert.equal(hidden.count, 1);
  });

  test("synthetic spheres: moving sphere B shifts its bounds and leaves A untouched", () => {
    const scene = makeTwoSpheres();
    const s = new EditSession({ gaussians: scene.gaussians, sh: scene.sh, shDegree: scene.shDegree, labels: scene.labels });
    s.apply({ op: "transformar", id_instancia: 2, xform: Array.from(mat4Translation([0, 0, 1.5])) });
    const before = bakeSession(new EditSession({ gaussians: scene.gaussians, labels: scene.labels }), { label: 2 });
    const after = bakeSession(s, { label: 2 });
    const cz = (b) => { let z = 0; for (let i = 0; i < b.count; i++) z += b.gaussians[i * 12 + 2]; return z / b.count; };
    near(cz(after) - cz(before), 1.5, 1e-4);
    const a = bakeSession(s, { label: 1 });
    assert.equal(a.count, 2000);
  });
});
