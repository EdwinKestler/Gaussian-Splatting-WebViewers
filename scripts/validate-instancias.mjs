#!/usr/bin/env node
// One-shot: validate one or more instancias.json files against shared/schemas/instancias.schema.json
// and check that a sibling etiquetas.u32 has fuente.n_gaussianas entries.
//   node scripts/validate-instancias.mjs artifacts/segmentaciones/<escena>/<fecha>/instancias.json …
import { readFile, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { loadSchema, validateAgainst } from "../shared/schemas.js";

const files = process.argv.slice(2);
if (!files.length) {
  console.error("uso: node scripts/validate-instancias.mjs <instancias.json>…");
  process.exit(2);
}
const schema = await loadSchema("instancias");
let failed = 0;
for (const f of files) {
  const json = JSON.parse(await readFile(f, "utf8"));
  const errors = validateAgainst(schema, json);
  const labelsPath = join(dirname(f), "etiquetas.u32");
  try {
    const s = await stat(labelsPath);
    if (s.size !== json.fuente.n_gaussianas * 4) errors.push(`etiquetas.u32: ${s.size / 4} valores, se esperaban ${json.fuente.n_gaussianas}`);
  } catch {
    errors.push("etiquetas.u32: no existe junto al JSON");
  }
  if (errors.length) {
    failed++;
    console.log(`✗ ${f}\n  ${errors.join("\n  ")}`);
  } else console.log(`✓ ${f}: ${json.n_instancias} instancias, ${json.fuente.n_gaussianas} gaussianas, método ${json.metodo.mascaras}`);
}
process.exit(failed ? 1 : 0);
