/**
 * Unit tests for shared/schemas.js and shared/schemas/instancias.schema.json
 * (plan §3.3: every generated JSON is validated against a schema). Run: npm test.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import { assertValid, loadSchema, validateAgainst } from "../../shared/schemas.js";
import { buildInstancesJson } from "../../shared/lift.js";

describe("instancias.schema.json", () => {
  test("a JSON built by buildInstancesJson (with embeddings and malla) is valid", async () => {
    const schema = await loadSchema("instancias");
    const json = buildInstancesJson({
      escena: "s.splat",
      fecha: "2026-09-03T00:00:00Z",
      fuente: { formato: "splat", sh_grado: 0 },
      metodo: { mascaras: "sam2", sesgo_fondo: 0.3, umbral_iou: 0.5, difusion_iter: 5, k_buffer: 24, vistas: 2 },
      labels: new Uint32Array([1, 1, 2, 0]),
      gaussians: new Float32Array(4 * 12),
      names: ["", { nombre: "clock", nombre_es: "reloj", categoria: "decoracion", confianza: 0.9, malla: "artifacts/mallas/s/1.glb" }],
      colors: ["", [1, 0, 0]],
      views: [{ indice: 0, instancias: [1, 2] }],
      embeddings: { modelo: "Xenova/clip-vit-base-patch32", dimension: 2, vectors: { 1: new Float32Array([0.5, 0.5]) } },
    });
    assert.deepEqual(validateAgainst(schema, json), []);
    assert.equal(assertValid(schema, json), json);
  });

  test("reports missing keys, wrong types and out-of-range values in Spanish", async () => {
    const schema = await loadSchema("instancias");
    const bad = { version: 1, escena: "s", fecha: "f", fuente: { n_gaussianas: -1 }, metodo: { mascaras: "x" }, n_instancias: 1, instancias: [{ id_instancia: 0, nombre_es: 3, n_gaussianas: 2, confianza: 1.5, bbox: { min: [0, 0], max: [0, 0, 0] } }] };
    const errors = validateAgainst(schema, bad);
    assert.ok(errors.some((e) => e.includes("$.fuente.n_gaussianas") && e.includes("mínimo")));
    assert.ok(errors.some((e) => e.includes("$.metodo.levantamiento: falta")));
    assert.ok(errors.some((e) => e.includes("$.instancias[0].id_instancia") && e.includes("mínimo 1")));
    assert.ok(errors.some((e) => e.includes("$.instancias[0].nombre_es: se esperaba string")));
    assert.ok(errors.some((e) => e.includes("$.instancias[0].confianza") && e.includes("máximo")));
    assert.ok(errors.some((e) => e.includes("$.instancias[0].bbox.min") && e.includes("elementos <")));
    assert.throws(() => assertValid(schema, bad, "instancias.json"), /no cumple el esquema/);
  });

  test("nullable types and enums", () => {
    const schema = { type: "object", properties: { a: { type: ["string", "null"] }, b: { enum: ["x", "y"] }, c: { type: "array", items: { type: "integer" }, maxItems: 2 } } };
    assert.deepEqual(validateAgainst(schema, { a: null, b: "x", c: [1, 2] }), []);
    const errors = validateAgainst(schema, { a: 1, b: "z", c: [1, 2.5, 3] });
    assert.equal(errors.length, 4, errors.join("; "));
  });
});
