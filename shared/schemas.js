/**
 * Minimal JSON Schema checker for the project's own schemas (plan §3.3:
 * "todo JSON se valida contra un esquema"). Supports the subset used in
 * shared/schemas/*.json: type (string or list), required, properties, items,
 * minItems/maxItems, minimum/maximum, enum. Returns a list of Spanish error
 * strings ("ruta: problema"); an empty list means valid.
 */

function typeOf(v) {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  if (Number.isInteger(v)) return "integer";
  return typeof v;
}

function matchesType(v, t) {
  const actual = typeOf(v);
  if (t === "number") return actual === "number" || actual === "integer";
  return actual === t;
}

export function validateAgainst(schema, value, path = "$", errors = []) {
  if (!schema || typeof schema !== "object") return errors;
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => matchesType(value, t))) {
      errors.push(`${path}: se esperaba ${types.join("|")}, hay ${typeOf(value)}`);
      return errors;
    }
  }
  if (value === null || value === undefined) return errors;
  if (schema.enum && !schema.enum.includes(value)) errors.push(`${path}: valor fuera de ${JSON.stringify(schema.enum)}`);
  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) errors.push(`${path}: ${value} < mínimo ${schema.minimum}`);
    if (schema.maximum !== undefined && value > schema.maximum) errors.push(`${path}: ${value} > máximo ${schema.maximum}`);
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) errors.push(`${path}: ${value.length} elementos < ${schema.minItems}`);
    if (schema.maxItems !== undefined && value.length > schema.maxItems) errors.push(`${path}: ${value.length} elementos > ${schema.maxItems}`);
    if (schema.items) value.forEach((v, i) => validateAgainst(schema.items, v, `${path}[${i}]`, errors));
  } else if (typeof value === "object") {
    for (const key of schema.required || []) {
      if (!(key in value)) errors.push(`${path}.${key}: falta`);
    }
    for (const [key, sub] of Object.entries(schema.properties || {})) {
      if (key in value) validateAgainst(sub, value[key], `${path}.${key}`, errors);
    }
  }
  return errors;
}

/** Load a schema from shared/schemas/ (Node: fs; browser: fetch relative to this module). */
export async function loadSchema(name) {
  const url = new URL(`./schemas/${name}.schema.json`, import.meta.url);
  if (typeof fetch === "function" && url.protocol !== "file:") return (await fetch(url)).json();
  const { readFile } = await import("node:fs/promises");
  return JSON.parse(await readFile(url, "utf8"));
}

/** Throws with every problem listed when `json` does not follow the schema. */
export function assertValid(schema, json, what = "JSON") {
  const errors = validateAgainst(schema, json);
  if (errors.length) throw new Error(`${what} no cumple el esquema:\n  ${errors.join("\n  ")}`);
  return json;
}
