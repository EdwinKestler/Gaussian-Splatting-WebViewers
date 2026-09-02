/**
 * Test helpers: build small PLY / .splat buffers in memory.
 *
 * Nothing here touches the filesystem; every function returns an ArrayBuffer
 * that can be fed straight into shared/splat-io.js.
 *
 * Gaussian record shape (all optional except x/y/z):
 *   {
 *     x, y, z,
 *     fDc:      [r, g, b]              SH DC coefficients (not colours)
 *     fRest:    [...]                  3 * restPerColour(shDegree) numbers,
 *                                      per-colour blocks: R[0..k) G[0..k) B[0..k)
 *     opacity:  logit                  pre-sigmoid, as stored on disk
 *     logScale: [s0, s1, (s2)]         pre-exp, as stored on disk
 *     rot:      [w, x, y, z]           raw quaternion (may be unnormalised)
 *     normal:   [nx, ny, nz]
 *   }
 */

export const SH_C0 = 0.28209479177387814;

/** f_rest coefficients per colour channel for SH degree 0..3. */
export const REST_PER_COLOR = { 0: 0, 1: 3, 2: 8, 3: 15 };

const TYPE_SIZE = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

export function restPerColor(shDegree) {
  const k = REST_PER_COLOR[shDegree];
  if (k === undefined) throw new Error(`Unsupported SH degree ${shDegree}`);
  return k;
}

/**
 * Ordered property names of an INRIA-style Gaussian PLY.
 * `variant` is "3dgs" (scale_0..2) or "2dgs" (scale_0..1, hbb1 layout).
 */
export function gaussianPropertyNames({
  shDegree = 0,
  variant = "3dgs",
  normals = false,
  opacity = true,
  rotation = true,
} = {}) {
  const names = ["x", "y", "z"];
  if (normals) names.push("nx", "ny", "nz");
  names.push("f_dc_0", "f_dc_1", "f_dc_2");
  const nRest = 3 * restPerColor(shDegree);
  for (let i = 0; i < nRest; i++) names.push(`f_rest_${i}`);
  if (opacity) names.push("opacity");
  const nScale = variant === "2dgs" ? 2 : 3;
  for (let i = 0; i < nScale; i++) names.push(`scale_${i}`);
  if (rotation) names.push("rot_0", "rot_1", "rot_2", "rot_3");
  return names;
}

function formatLine(encoding) {
  if (encoding === "binary_le") return "format binary_little_endian 1.0";
  if (encoding === "binary_be") return "format binary_big_endian 1.0";
  if (encoding === "ascii") return "format ascii 1.0";
  throw new Error(`Unknown PLY encoding ${encoding}`);
}

export function makePlyHeader({ vertexCount, encoding = "binary_le", properties, comments = [] }) {
  const lines = ["ply", formatLine(encoding)];
  for (const c of comments) lines.push(`comment ${c}`);
  lines.push(`element vertex ${vertexCount}`);
  for (const p of properties) lines.push(`property ${p.type} ${p.name}`);
  lines.push("end_header");
  return lines.join("\n") + "\n";
}

function writeNumeric(view, offset, type, value, le) {
  switch (type) {
    case "float":
    case "float32": view.setFloat32(offset, value, le); break;
    case "double":
    case "float64": view.setFloat64(offset, value, le); break;
    case "int":
    case "int32": view.setInt32(offset, value, le); break;
    case "uint":
    case "uint32": view.setUint32(offset, value, le); break;
    case "short":
    case "int16": view.setInt16(offset, value, le); break;
    case "ushort":
    case "uint16": view.setUint16(offset, value, le); break;
    case "char":
    case "int8": view.setInt8(offset, value); break;
    case "uchar":
    case "uint8": view.setUint8(offset, value); break;
    default: throw new Error(`Unsupported PLY property type: ${type}`);
  }
}

function isIntegerType(type) {
  return !(type === "float" || type === "float32" || type === "double" || type === "float64");
}

/**
 * Generic PLY writer.
 *   properties: [{ name, type }]
 *   rows:       number[][] (one array per vertex, same order as properties)
 */
export function makePly({ properties, rows, encoding = "binary_le", comments = [] }) {
  for (const row of rows) {
    if (row.length !== properties.length) {
      throw new Error(`Row has ${row.length} values but ${properties.length} properties are declared`);
    }
  }
  const header = new TextEncoder().encode(
    makePlyHeader({ vertexCount: rows.length, encoding, properties, comments })
  );
  let body;
  if (encoding === "ascii") {
    const text = rows
      .map((row) =>
        row
          .map((v, i) => (isIntegerType(properties[i].type) ? String(Math.round(v)) : String(v)))
          .join(" ")
      )
      .join("\n") + "\n";
    body = new TextEncoder().encode(text);
  } else {
    const le = encoding === "binary_le";
    const rowSize = properties.reduce((acc, p) => acc + TYPE_SIZE[p.type], 0);
    body = new Uint8Array(rows.length * rowSize);
    const view = new DataView(body.buffer);
    let offset = 0;
    for (const row of rows) {
      for (let i = 0; i < properties.length; i++) {
        writeNumeric(view, offset, properties[i].type, row[i], le);
        offset += TYPE_SIZE[properties[i].type];
      }
    }
  }
  const out = new Uint8Array(header.length + body.length);
  out.set(header, 0);
  out.set(body, header.length);
  return out.buffer;
}

/** Fill one Gaussian record into the property order given by `names`. */
function gaussianRow(rec, names, shDegree) {
  const k = restPerColor(shDegree);
  const fRest = rec.fRest || new Array(3 * k).fill(0);
  if (fRest.length !== 3 * k) {
    throw new Error(`fRest has ${fRest.length} values, expected ${3 * k} for SH degree ${shDegree}`);
  }
  const fDc = rec.fDc || [0, 0, 0];
  const logScale = rec.logScale || [0, 0, 0];
  const rot = rec.rot || [1, 0, 0, 0];
  const normal = rec.normal || [0, 0, 0];
  const values = {
    x: rec.x, y: rec.y, z: rec.z,
    nx: normal[0], ny: normal[1], nz: normal[2],
    f_dc_0: fDc[0], f_dc_1: fDc[1], f_dc_2: fDc[2],
    opacity: rec.opacity ?? 0,
    scale_0: logScale[0], scale_1: logScale[1], scale_2: logScale[2],
    rot_0: rot[0], rot_1: rot[1], rot_2: rot[2], rot_3: rot[3],
  };
  for (let i = 0; i < fRest.length; i++) values[`f_rest_${i}`] = fRest[i];
  return names.map((n) => {
    if (!(n in values)) throw new Error(`No value for property ${n}`);
    const v = values[n];
    if (!Number.isFinite(v)) throw new Error(`Property ${n} is not finite: ${v}`);
    return v;
  });
}

/**
 * Build an INRIA-style Gaussian PLY (3DGS or 2DGS) from records.
 * options: { shDegree, variant, encoding, normals, opacity, rotation }
 */
export function makeGaussianPly(records, options = {}) {
  const shDegree = options.shDegree ?? 0;
  const names = gaussianPropertyNames({ shDegree, ...options });
  const properties = names.map((name) => ({ name, type: "float" }));
  const rows = records.map((rec) => gaussianRow(rec, names, shDegree));
  return makePly({ properties, rows, encoding: options.encoding ?? "binary_le" });
}

/**
 * Plain point-cloud PLY: x y z as float, red green blue as uchar (0..255).
 * points: [{ x, y, z, rgb: [r, g, b] }]
 */
export function makePointCloudPly(points, options = {}) {
  const properties = [
    { name: "x", type: "float" },
    { name: "y", type: "float" },
    { name: "z", type: "float" },
    { name: "red", type: "uchar" },
    { name: "green", type: "uchar" },
    { name: "blue", type: "uchar" },
  ];
  const rows = points.map((p) => [p.x, p.y, p.z, p.rgb[0], p.rgb[1], p.rgb[2]]);
  return makePly({ properties, rows, encoding: options.encoding ?? "binary_le" });
}

/**
 * 32-byte .splat rows: xyz f32, scale f32, rgba u8, quat u8 (wxyz -> [0,255]).
 * rows: [{ x, y, z, scale: [sx, sy, sz], rgba: [r, g, b, a] (0..255), quat: [w, x, y, z] }]
 */
export function makeSplat32(rows) {
  const out = new ArrayBuffer(rows.length * 32);
  const f = new Float32Array(out);
  const u = new Uint8Array(out);
  rows.forEach((row, i) => {
    const o = i * 8;
    f[o] = row.x; f[o + 1] = row.y; f[o + 2] = row.z;
    f[o + 3] = row.scale[0]; f[o + 4] = row.scale[1]; f[o + 5] = row.scale[2];
    const b = i * 32;
    for (let c = 0; c < 4; c++) u[b + 24 + c] = row.rgba[c];
    const q = row.quat || [1, 0, 0, 0];
    const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    // same mapping (and clamp) as packQuatU8 in shared/splat-io.js: +1 -> 255, -1 -> 0
    for (let c = 0; c < 4; c++) u[b + 28 + c] = Math.max(0, Math.min(255, Math.round((q[c] / n) * 128 + 128)));
  });
  return out;
}

/**
 * 44-byte .splat rows: xyz f32, scale f32, rgba u8, quat f32 (wxyz).
 * Same row shape as makeSplat32.
 */
export function makeSplat44(rows) {
  const out = new ArrayBuffer(rows.length * 44);
  const f = new Float32Array(out);
  const u = new Uint8Array(out);
  rows.forEach((row, i) => {
    const o = i * 11;
    f[o] = row.x; f[o + 1] = row.y; f[o + 2] = row.z;
    f[o + 3] = row.scale[0]; f[o + 4] = row.scale[1]; f[o + 5] = row.scale[2];
    const b = i * 44;
    for (let c = 0; c < 4; c++) u[b + 24 + c] = row.rgba[c];
    const q = row.quat || [1, 0, 0, 0];
    for (let c = 0; c < 4; c++) f[o + 7 + c] = q[c];
  });
  return out;
}

/** Deterministic pseudo-random generator for reproducible fixtures. */
export function makeRng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return (s >>> 0) / 4294967296;
  };
}
