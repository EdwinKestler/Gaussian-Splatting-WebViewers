/**
 * Unit tests for shared/naming.js (plan F4: framing, names, text search). Run: npm test.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import { applyNames, frameBounds, instanceBounds, matchScore, normalizeText, searchInstances } from "../../shared/naming.js";

describe("instanceBounds / frameBounds", () => {
  test("bounds of one label and camera distance that frames it", () => {
    const labels = new Uint32Array([1, 1, 2, 0]);
    const g = new Float32Array(4 * 12);
    g.set([0, 0, 0], 0);
    g.set([2, 1, 0], 12);
    g.set([9, 9, 9], 24);
    const b = instanceBounds(labels, g, 1);
    assert.deepEqual(b.min, [0, 0, 0]);
    assert.deepEqual(b.max, [2, 1, 0]);
    assert.deepEqual(b.center, [1, 0.5, 0]);
    assert.ok(Math.abs(b.radius - Math.hypot(2, 1, 0) / 2) < 1e-9);
    assert.equal(b.count, 2);
    assert.equal(instanceBounds(labels, g, 7), null);
    const fov = (50 * Math.PI) / 180;
    const f = frameBounds(b, fov, 1.4);
    assert.deepEqual(f.target, b.center);
    assert.ok(Math.abs(f.radius - (b.radius * 1.4) / Math.sin(fov / 2)) < 1e-9);
    assert.throws(() => frameBounds(b, 0), /fov/);
    assert.throws(() => instanceBounds(labels, new Float32Array(5), 1), /12 floats/);
  });
});

describe("text search", () => {
  const rows = [
    { label: 1, nombre: "office chair", nombre_es: "silla de oficina", categoria: "mobiliario", descripcion_es: "silla giratoria gris" },
    { label: 2, nombre: "table", nombre_es: "mesa", categoria: "mobiliario" },
    { label: 3, name: "grupo 3" },
  ];
  test("normalizeText strips accents and case", () => {
    assert.equal(normalizeText("  Sillón  ROJO "), "sillon rojo");
  });
  test("matchScore ranks exact > word > substring and requires every word", () => {
    assert.ok(matchScore(rows[1], "mesa") > matchScore(rows[0], "silla"));
    assert.ok(matchScore(rows[0], "silla") > matchScore(rows[0], "sill"));
    assert.equal(matchScore(rows[0], "silla verde"), 0, "una palabra sin coincidencia anula la búsqueda");
    assert.equal(matchScore(rows[0], ""), 0);
    assert.ok(matchScore(rows[0], "Oficina") > 0);
  });
  test("searchInstances returns matches best first", () => {
    assert.deepEqual(searchInstances(rows, "mobiliario").map((r) => r.label), [1, 2]);
    assert.deepEqual(searchInstances(rows, "grupo").map((r) => r.label), [3]);
    assert.deepEqual(searchInstances(rows, "mesa").map((r) => r.label), [2]);
    assert.deepEqual(searchInstances(rows, "nada"), []);
  });
});

describe("applyNames", () => {
  test("merges results into the registry, keeps names on failures", () => {
    const entries = new Map([
      [1, { name: "grupo 1", count: 10 }],
      [2, { name: "grupo 2", count: 5 }],
    ]);
    const res = applyNames(entries, [
      { id_instancia: 1, ok: true, nombre: "clock", nombre_es: "reloj", categoria: "decoracion", confianza: 0.91, descripcion_es: "reloj despertador" },
      { id_instancia: 2, ok: false, error: "timeout" },
      { id_instancia: 9, ok: true, nombre: "ghost" },
    ]);
    assert.deepEqual(res, { applied: 1, failed: 1 });
    assert.equal(entries.get(1).name, "reloj");
    assert.equal(entries.get(1).nombre, "clock");
    assert.equal(entries.get(1).categoria, "decoracion");
    assert.equal(entries.get(1).confianza, 0.91);
    assert.equal(entries.get(2).name, "grupo 2");
    assert.equal(entries.get(2).error, "timeout");
  });
});
