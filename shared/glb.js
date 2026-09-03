/**
 * Plan F6 — minimal binary glTF 2.0 (GLB) writer for an indexed triangle mesh
 * with normals and per-vertex colours. One buffer, one mesh, one node, one
 * scene; accessors carry min/max for POSITION as the spec requires.
 */

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;
const FLOAT = 5126;
const UINT32 = 5125;

function pad4(n) {
  return (n + 3) & ~3;
}

/**
 * @param {{positions:Float32Array, normals?:Float32Array, colors?:Float32Array, indices:Uint32Array}} mesh
 * @param {{name?:string, generator?:string, extras?:object}} options
 * @returns {ArrayBuffer}
 */
export function encodeGlb(mesh, { name = "malla", generator = "Gaussian-Splatting-WebViewers F6", extras = null } = {}) {
  const { positions, normals = null, colors = null, indices } = mesh;
  if (!(positions instanceof Float32Array) || positions.length % 3) throw new Error("positions: Float32Array de 3 floats por vértice");
  if (!(indices instanceof Uint32Array) || indices.length % 3) throw new Error("indices: Uint32Array de 3 índices por triángulo");
  const nv = positions.length / 3;
  if (normals && normals.length !== positions.length) throw new Error("normals debe tener el tamaño de positions");
  if (colors && colors.length !== positions.length) throw new Error("colors debe tener 3 floats por vértice");
  for (let i = 0; i < indices.length; i++) if (indices[i] >= nv) throw new Error(`índice ${indices[i]} fuera de rango`);

  const views = [];
  const accessors = [];
  const parts = [];
  let offset = 0;
  const addView = (array, target, byteStride = null) => {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    const view = { buffer: 0, byteOffset: offset, byteLength: bytes.byteLength, target };
    if (byteStride) view.byteStride = byteStride;
    views.push(view);
    parts.push(bytes);
    offset += pad4(bytes.byteLength);
    return views.length - 1;
  };
  const minMax = (arr) => {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < arr.length; i += 3) {
      for (let a = 0; a < 3; a++) {
        if (arr[i + a] < min[a]) min[a] = arr[i + a];
        if (arr[i + a] > max[a]) max[a] = arr[i + a];
      }
    }
    return { min, max };
  };
  const attributes = {};
  const posView = addView(positions, 34962, 12);
  accessors.push({ bufferView: posView, componentType: FLOAT, count: nv, type: "VEC3", ...minMax(positions) });
  attributes.POSITION = accessors.length - 1;
  if (normals) {
    accessors.push({ bufferView: addView(normals, 34962, 12), componentType: FLOAT, count: nv, type: "VEC3" });
    attributes.NORMAL = accessors.length - 1;
  }
  if (colors) {
    accessors.push({ bufferView: addView(colors, 34962, 12), componentType: FLOAT, count: nv, type: "VEC3" });
    attributes.COLOR_0 = accessors.length - 1;
  }
  accessors.push({ bufferView: addView(indices, 34963), componentType: UINT32, count: indices.length, type: "SCALAR" });
  const indexAccessor = accessors.length - 1;

  const json = {
    asset: { version: "2.0", generator },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name }],
    meshes: [{ name, primitives: [{ attributes, indices: indexAccessor, mode: 4 }] }],
    buffers: [{ byteLength: offset }],
    bufferViews: views,
    accessors,
  };
  if (extras) json.extras = extras;
  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonLen = pad4(jsonBytes.length);
  const binLen = offset;
  const total = 12 + 8 + jsonLen + 8 + binLen;
  const out = new ArrayBuffer(total);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);
  dv.setUint32(0, GLB_MAGIC, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonLen, true);
  dv.setUint32(16, CHUNK_JSON, true);
  u8.set(jsonBytes, 20);
  for (let i = 20 + jsonBytes.length; i < 20 + jsonLen; i++) u8[i] = 0x20; // pad JSON with spaces
  const binStart = 20 + jsonLen;
  dv.setUint32(binStart, binLen, true);
  dv.setUint32(binStart + 4, CHUNK_BIN, true);
  let o = binStart + 8;
  for (const p of parts) {
    u8.set(p, o);
    o += pad4(p.byteLength);
  }
  return out;
}

/** Parse the GLB header + JSON chunk (for tests and HUD summaries). */
export function decodeGlbHeader(buffer) {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== GLB_MAGIC) throw new Error("no es un GLB");
  const version = dv.getUint32(4, true);
  const length = dv.getUint32(8, true);
  const jsonLen = dv.getUint32(12, true);
  if (dv.getUint32(16, true) !== CHUNK_JSON) throw new Error("falta el chunk JSON");
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen)));
  const binStart = 20 + jsonLen;
  const binLen = dv.getUint32(binStart, true);
  if (dv.getUint32(binStart + 4, true) !== CHUNK_BIN) throw new Error("falta el chunk BIN");
  return { version, length, json, binOffset: binStart + 8, binLength: binLen };
}
