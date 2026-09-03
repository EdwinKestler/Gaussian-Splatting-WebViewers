import { describe, test } from "node:test";
import assert from "node:assert/strict";

import { repairMesh, scaleMeshToMillimeters, validateMesh } from "../../shared/mesh-ops.js";
import { encode3mf, read3mfFiles, THREE_MF } from "../../shared/three-mf.js";

function tetrahedron(indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]) {
  return {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    normals: new Float32Array(12),
    colors: new Float32Array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
    indices: new Uint32Array(indices),
  };
}

describe("mesh validation and repair", () => {
  test("recognizes a closed, consistently oriented tetrahedron", () => {
    const report = validateMesh(tetrahedron());
    assert.equal(report.geometryValid, true);
    assert.equal(report.manifold, true);
    assert.equal(report.oriented, true);
    assert.equal(report.watertight, true);
    assert.equal(report.printable, true);
    assert.equal(report.boundaryEdges, 0);
    assert.equal(report.nonManifoldEdges, 0);
    assert.ok(Math.abs(report.signedVolume - 1 / 6) < 1e-8);
  });

  test("fills a small simple hole and restores outward winding", () => {
    const open = tetrahedron([0, 2, 1, 0, 1, 3, 0, 3, 2]);
    const before = validateMesh(open);
    assert.equal(before.boundaryEdges, 3);
    assert.equal(before.watertight, false);
    const repaired = repairMesh(open, { maxHoleEdges: 8 });
    assert.equal(repaired.repair.holesFound, 1);
    assert.equal(repaired.repair.holesFilled, 1);
    assert.equal(repaired.after.watertight, true);
    assert.equal(repaired.after.printable, true);
    assert.equal(repaired.after.boundaryEdges, 0);
    assert.ok(repaired.after.signedVolume > 0);
  });

  test("welds duplicate vertices and removes duplicate/degenerate triangles", () => {
    const mesh = tetrahedron();
    mesh.positions = new Float32Array([...mesh.positions, 0.00000001, 0, 0]);
    mesh.normals = new Float32Array(mesh.positions.length);
    mesh.colors = new Float32Array([...mesh.colors, 1, 0, 0]);
    mesh.indices = new Uint32Array([...mesh.indices, 4, 2, 1, 0, 0, 1]);
    const repaired = repairMesh(mesh, { fillHoles: false, weldTolerance: 1e-5 });
    assert.equal(repaired.repair.weldedVertices, 1);
    assert.equal(repaired.repair.removedDuplicateTriangles, 1);
    assert.equal(repaired.repair.removedDegenerateTriangles, 1);
    assert.equal(repaired.after.watertight, true);
    assert.equal(repaired.mesh.vertexCount, 4);
  });

  test("reports a non-manifold edge and does not call it printable", () => {
    const mesh = tetrahedron([...tetrahedron().indices, 0, 1, 2]);
    // Reversed duplicate face is identified separately, so attach a fifth vertex.
    mesh.positions = new Float32Array([...mesh.positions, 0, -1, 0]);
    mesh.normals = new Float32Array(mesh.positions.length);
    mesh.colors = new Float32Array(mesh.positions.length).fill(0.5);
    mesh.indices = new Uint32Array([...tetrahedron().indices, 0, 1, 4]);
    const report = validateMesh(mesh);
    assert.equal(report.nonManifoldEdges, 1);
    assert.equal(report.printable, false);
  });

  test("scales the longest dimension to millimetres and rests on the build plane", () => {
    const scaled = scaleMeshToMillimeters(tetrahedron(), { maxDimensionMm: 80 });
    assert.equal(scaled.unit, "millimeter");
    assert.equal(scaled.scale, 80);
    assert.deepEqual(scaled.bboxMm.min, [0, 0, 0]);
    assert.deepEqual(scaled.bboxMm.max, [80, 80, 80]);
    assert.equal(scaled.validation.printable, true);
  });
});

describe("3MF export", () => {
  test("writes the OPC parts, millimetre model, resources and build item", () => {
    const scaled = scaleMeshToMillimeters(tetrahedron(), { maxDimensionMm: 60 }).mesh;
    const one = new Uint8Array(encode3mf(scaled, { name: 'Clock & gear "A"' }));
    const two = new Uint8Array(encode3mf(scaled, { name: 'Clock & gear "A"' }));
    assert.deepEqual(one, two, "same mesh produces a deterministic package");
    assert.equal(new DataView(one.buffer).getUint32(0, true), 0x04034b50);
    assert.equal(new DataView(one.buffer).getUint32(one.length - 22, true), 0x06054b50);
    const files = read3mfFiles(one);
    assert.deepEqual([...files.keys()], ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]);
    const decode = (path) => new TextDecoder().decode(files.get(path));
    assert.ok(decode("[Content_Types].xml").includes(THREE_MF.MODEL_CONTENT_TYPE));
    assert.ok(decode("_rels/.rels").includes(THREE_MF.START_PART_REL));
    const model = decode("3D/3dmodel.model");
    assert.match(model, new RegExp(`xmlns="${THREE_MF.MODEL_NS}"`));
    assert.match(model, /<model unit="millimeter"/);
    assert.match(model, /Clock &amp; gear &quot;A&quot;/);
    assert.equal((model.match(/<vertex /g) || []).length, 4);
    assert.equal((model.match(/<triangle /g) || []).length, 4);
    assert.match(model, /<build><item objectid="2"\/><\/build>/);
  });

  test("rejects an open mesh by default", () => {
    const open = tetrahedron([0, 2, 1, 0, 1, 3, 0, 3, 2]);
    assert.throws(() => encode3mf(open), /print check failed.*3 boundary edges/);
  });

  test("reports malformed geometry before packaging", () => {
    const broken = tetrahedron();
    broken.indices = new Uint32Array([...broken.indices, 0, 0, 1]);
    assert.throws(() => encode3mf(broken), /1 degenerate triangles/);
  });
});
