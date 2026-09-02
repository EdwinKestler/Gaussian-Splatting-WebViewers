/**
 * Shared Gaussian splat I/O.
 *
 * Understands:
 *   - INRIA / 3DGS .ply (binary LE, binary BE, ASCII), SH degree 0..3
 *   - 2DGS .ply (hbb1/2d-gaussian-splatting: scale_0, scale_1 only)
 *   - plain point-cloud .ply (x y z + red green blue, no scale_*)
 *   - 32-byte .splat  (antimatter15 / Viewer 1 / converter)
 *   - 44-byte .splat  (early GaussianSplats3D / Viewer 2)
 *
 * Always emits a packed 32-byte buffer:
 *   xyz f32, scale f32, rgba u8, quat u8 (wxyz, mapped from [-1,1] to [0,255])
 *
 * Every parsed result carries `format` ("ply" | "splat32" | "splat44") and
 * `variant` ("3dgs" | "2dgs" | "pointcloud"); .splat files are always "3dgs".
 */

export const SPLAT32_ROW = 32;
const SH_C0 = 0.28209479177387814;

/** Linear scale used for PLYs without scale_* (plain point clouds). */
const POINT_SCALE = 0.01;

/**
 * 2DGS PLYs store only two log-scales: each primitive is a flat disk spanned by
 * the first two rotated axes, with the third axis as its normal. The renderers
 * here build a full 3x3 covariance from three scales, so the missing third
 * scale is synthesised as a very thin slab *relative to the disk itself*:
 * sz = min(sx, sy) * THIN_DISK_RATIO. A relative value (instead of a fixed
 * exp(-1e6)-style constant) keeps the disk thin at any scene scale while
 * surviving the log/exp round trip of packedToPly (which floors scales at 1e-8)
 * and float32 storage without degenerating the covariance.
 */
export const THIN_DISK_RATIO = 1e-4;
const THIN_DISK_FLOOR = 1e-8;

const BYTE_TYPES = new Set(["uchar", "uint8", "char", "int8"]);

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/** Third (normal) scale for a 2DGS disk of linear extents sx, sy. */
export function thinDiskScale(sx, sy) {
  const s = Math.min(Math.abs(sx), Math.abs(sy));
  return Math.max(s * THIN_DISK_RATIO, THIN_DISK_FLOOR);
}

/**
 * Classify a PLY vertex layout from its property names:
 *   "3dgs"       scale_0..scale_2 (a lone scale_0 is treated as isotropic 3DGS)
 *   "2dgs"       scale_0 and scale_1 but no scale_2
 *   "pointcloud" no scale_* at all
 */
export function plyVariantFromProperties(names) {
  const set = new Set(names);
  if (!set.has("scale_0")) return "pointcloud";
  if (set.has("scale_1") && !set.has("scale_2")) return "2dgs";
  return "3dgs";
}

function isPlyMagic(u8) {
  return u8.length >= 3 && u8[0] === 0x70 && u8[1] === 0x6c && u8[2] === 0x79;
}

/** Accept ArrayBuffer or any ArrayBufferView; always return an ArrayBuffer. */
function asArrayBuffer(input) {
  if (input instanceof ArrayBuffer) return input;
  if (ArrayBuffer.isView(input)) {
    if (input.byteOffset === 0 && input.byteLength === input.buffer.byteLength) return input.buffer;
    return input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength);
  }
  throw new TypeError("Expected an ArrayBuffer or a typed array");
}

/** Per-property byte offsets, types and row size for a binary PLY vertex. */
function indexProperties(properties) {
  const offsets = {};
  const types = {};
  let rowSize = 0;
  for (const p of properties) {
    offsets[p.name] = rowSize;
    types[p.name] = p.type;
    rowSize += TYPE_SIZE[p.type] || 4;
  }
  const has = (name) => Object.prototype.hasOwnProperty.call(offsets, name);
  return { offsets, types, rowSize, has };
}

function countRestCoefficients(has) {
  let n = 0;
  while (has(`f_rest_${n}`)) n += 1;
  return n;
}

/** f_rest_* count -> SH degree (3 per colour channel = degree 1, 8 = 2, 15 = 3). */
function shDegreeFromRestCount(nRest) {
  const perColor = nRest / 3;
  if (perColor >= 15) return 3;
  if (perColor >= 8) return 2;
  if (perColor >= 3) return 1;
  return 0;
}

/**
 * Linear (exp) scales for one vertex. `get(name)` returns the raw log-scale.
 *   3dgs: exp(scale_0..2); a lone scale_0 is broadcast (isotropic)
 *   2dgs: exp(scale_0..1) + synthesised thin third axis
 *   pointcloud: fixed POINT_SCALE
 */
function linearScales(get, has, variant) {
  if (variant === "pointcloud") return [POINT_SCALE, POINT_SCALE, POINT_SCALE];
  const sx = Math.exp(get("scale_0"));
  if (variant === "2dgs") {
    const sy = Math.exp(get("scale_1"));
    return [sx, sy, thinDiskScale(sx, sy)];
  }
  const sy = has("scale_1") ? Math.exp(get("scale_1")) : sx;
  const sz = has("scale_2") ? Math.exp(get("scale_2")) : sy;
  return [sx, sy, sz];
}

/**
 * red/green/blue -> [0,1]. Byte-typed properties are always /255; float colours
 * are assumed to be in [0,1] unless any channel exceeds 1 (then /255).
 */
function unitColor(r, g, b, type) {
  if (BYTE_TYPES.has(type) || r > 1 || g > 1 || b > 1) return [r / 255, g / 255, b / 255];
  return [r, g, b];
}

/** Round to the nearest byte (Uint8Array assignment would otherwise truncate). */
function clampByte(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

/** Throw a clear error when the body is shorter than the header promises. */
function assertBinaryBody(buffer, headerEnd, vertexCount, rowSize) {
  const available = buffer.byteLength - headerEnd;
  const needed = vertexCount * rowSize;
  if (available < needed) {
    throw new Error(
      `PLY body truncated: header declares ${vertexCount} vertices x ${rowSize} bytes ` +
        `(${needed} bytes) but only ${available} bytes follow end_header`
    );
  }
}

function asciiRows(buffer, headerEnd, vertexCount) {
  const text = new TextDecoder().decode(new Uint8Array(buffer, headerEnd));
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  if (lines.length < vertexCount) {
    throw new Error(
      `PLY ASCII body truncated: header declares ${vertexCount} vertices but only ${lines.length} rows follow end_header`
    );
  }
  return lines;
}

function looksLikeSplat44(buffer) {
  if (buffer.byteLength < 44 || buffer.byteLength % 44 !== 0) return false;
  const f = new Float32Array(buffer.slice(0, 44));
  const n = Math.hypot(f[7], f[8], f[9], f[10]);
  return n > 0.4 && n < 1.6 && Number.isFinite(n);
}

export function detectFormat(input, filename = "") {
  const buffer = asArrayBuffer(input);
  const name = String(filename).toLowerCase();
  const u8 = new Uint8Array(buffer);
  if (name.endsWith(".ply") || isPlyMagic(u8)) {
    return "ply";
  }
  const len = buffer.byteLength;
  const div32 = len % 32 === 0;
  const div44 = len % 44 === 0;
  if (name.endsWith(".splat")) {
    if (div44 && !div32) return "splat44";
    if (div32 && !div44) return "splat32";
    if (div44 && looksLikeSplat44(buffer)) return "splat44";
    if (div32) return "splat32";
  }
  if (div44 && !div32) return "splat44";
  if (div32) return "splat32";
  throw new Error(`Unrecognized splat format (${len} bytes, name="${filename}")`);
}

function boundsFromPacked(packed) {
  const f = new Float32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const count = packed.byteLength / SPLAT32_ROW;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const o = i * 8;
    const x = f[o], y = f[o + 1], z = f[o + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) {
    minX = minY = minZ = -1;
    maxX = maxY = maxZ = 1;
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
    radius: Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5 || 1,
  };
}

function downsamplePacked(packed, compression) {
  const factor = Math.max(1, Math.min(10, compression | 0 || 1));
  if (factor === 1) return packed;
  const count = packed.byteLength / SPLAT32_ROW;
  const keep = Math.max(1, Math.floor(count / factor));
  const out = new Uint8Array(keep * SPLAT32_ROW);
  if (keep === count) {
    out.set(packed.subarray(0, keep * SPLAT32_ROW));
    return out;
  }
  for (let i = 0; i < keep; i++) {
    const src = Math.floor((i * count) / keep);
    out.set(packed.subarray(src * SPLAT32_ROW, (src + 1) * SPLAT32_ROW), i * SPLAT32_ROW);
  }
  return out;
}

function packQuatU8(out, offset, w, x, y, z) {
  const n = Math.hypot(w, x, y, z) || 1;
  out[offset] = Math.max(0, Math.min(255, Math.round((w / n) * 128 + 128)));
  out[offset + 1] = Math.max(0, Math.min(255, Math.round((x / n) * 128 + 128)));
  out[offset + 2] = Math.max(0, Math.min(255, Math.round((y / n) * 128 + 128)));
  out[offset + 3] = Math.max(0, Math.min(255, Math.round((z / n) * 128 + 128)));
}

function splat32ToPacked(buffer) {
  return new Uint8Array(buffer.slice(0));
}

function splat44ToPacked(buffer) {
  const count = Math.floor(buffer.byteLength / 44);
  const srcF = new Float32Array(buffer);
  const srcU = new Uint8Array(buffer);
  const packed = new Uint8Array(count * SPLAT32_ROW);
  const dstF = new Float32Array(packed.buffer);
  for (let i = 0; i < count; i++) {
    const sf = i * 11;
    const df = i * 8;
    const sb = i * 44;
    const db = i * SPLAT32_ROW;
    dstF[df] = srcF[sf];
    dstF[df + 1] = srcF[sf + 1];
    dstF[df + 2] = srcF[sf + 2];
    dstF[df + 3] = srcF[sf + 3];
    dstF[df + 4] = srcF[sf + 4];
    dstF[df + 5] = srcF[sf + 5];
    packed[db + 24] = srcU[sb + 24];
    packed[db + 25] = srcU[sb + 25];
    packed[db + 26] = srcU[sb + 26];
    packed[db + 27] = srcU[sb + 27];
    packQuatU8(packed, db + 28, srcF[sf + 7], srcF[sf + 8], srcF[sf + 9], srcF[sf + 10]);
  }
  return packed;
}

function decodePlyHeader(buffer) {
  const u8 = new Uint8Array(buffer);
  const decoder = new TextDecoder("ascii");
  let headerText = "";
  let headerEnd = -1;
  const chunk = 256;
  for (let offset = 0; offset < Math.min(u8.length, 256 * 1024); offset += chunk) {
    headerText += decoder.decode(u8.subarray(offset, Math.min(u8.length, offset + chunk)));
    const idx = headerText.search(/end_header\r?\n/);
    if (idx >= 0) {
      const match = headerText.match(/end_header\r?\n/);
      headerEnd = idx + match[0].length;
      headerText = headerText.slice(0, headerEnd);
      break;
    }
  }
  if (headerEnd < 0) throw new Error("PLY header is missing end_header");

  const lines = headerText.split(/\r?\n/);
  let format = "ascii";
  let vertexCount = 0;
  const properties = [];
  let inVertex = false;

  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("format ")) {
      if (line.includes("binary_little_endian")) format = "binary_le";
      else if (line.includes("binary_big_endian")) format = "binary_be";
      else format = "ascii";
    } else if (line.startsWith("element ")) {
      const parts = line.split(/\s+/);
      inVertex = parts[1] === "vertex";
      if (inVertex) vertexCount = parseInt(parts[2], 10) || 0;
    } else if (inVertex && line.startsWith("property ")) {
      const parts = line.split(/\s+/);
      properties.push({ type: parts[1], name: parts[2] });
    }
  }
  if (!vertexCount) throw new Error("PLY has no vertex element");
  if (!properties.length) throw new Error("PLY vertex element has no properties");
  return { format, vertexCount, properties, headerEnd, headerText };
}

const TYPE_SIZE = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

function readNumeric(view, offset, type, le) {
  switch (type) {
    case "float":
    case "float32": return view.getFloat32(offset, le);
    case "double":
    case "float64": return view.getFloat64(offset, le);
    case "int":
    case "int32": return view.getInt32(offset, le);
    case "uint":
    case "uint32": return view.getUint32(offset, le);
    case "short":
    case "int16": return view.getInt16(offset, le);
    case "ushort":
    case "uint16": return view.getUint16(offset, le);
    case "char":
    case "int8": return view.getInt8(offset);
    case "uchar":
    case "uint8": return view.getUint8(offset);
    default: throw new Error(`Unsupported PLY property type: ${type}`);
  }
}

/** Parse a PLY into 32-byte rows (sorted by volume x opacity). Returns { packed, variant }. */
function plyToPacked(buffer) {
  const { format, vertexCount, properties, headerEnd } = decodePlyHeader(buffer);
  const { offsets, types, rowSize, has } = indexProperties(properties);
  const variant = plyVariantFromProperties(properties.map((p) => p.name));
  const raw = new Array(vertexCount);

  if (format === "ascii") {
    const lines = asciiRows(buffer, headerEnd, vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      const parts = lines[i].trim().split(/\s+/);
      const rec = {};
      for (let p = 0; p < properties.length; p++) {
        rec[properties[p].name] = parseFloat(parts[p]);
      }
      raw[i] = rec;
    }
  } else {
    const le = format === "binary_le";
    assertBinaryBody(buffer, headerEnd, vertexCount, rowSize);
    const view = new DataView(buffer, headerEnd);
    for (let i = 0; i < vertexCount; i++) {
      const base = i * rowSize;
      const rec = {};
      for (const p of properties) {
        rec[p.name] = readNumeric(view, base + offsets[p.name], p.type, le);
      }
      raw[i] = rec;
    }
  }

  const scales = new Array(vertexCount);
  const sizeList = new Float32Array(vertexCount);
  const order = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    order[i] = i;
    const r = raw[i];
    const get = (name) => (Number.isFinite(r[name]) ? r[name] : 0);
    scales[i] = linearScales(get, has, variant);
    if (variant === "pointcloud") {
      sizeList[i] = 0;
      continue;
    }
    const [sx, sy, sz] = scales[i];
    const op = has("opacity") ? sigmoid(get("opacity")) : 1;
    sizeList[i] = sx * sy * sz * op;
  }
  order.sort((a, b) => sizeList[b] - sizeList[a]);

  const packed = new Uint8Array(vertexCount * SPLAT32_ROW);
  const dstF = new Float32Array(packed.buffer);

  for (let j = 0; j < vertexCount; j++) {
    const src = order[j];
    const r = raw[src];
    const df = j * 8;
    const db = j * SPLAT32_ROW;
    dstF[df] = r.x || 0;
    dstF[df + 1] = r.y || 0;
    dstF[df + 2] = r.z || 0;
    dstF[df + 3] = scales[src][0];
    dstF[df + 4] = scales[src][1];
    dstF[df + 5] = scales[src][2];
    if (has("f_dc_0")) {
      packed[db + 24] = clampByte((0.5 + SH_C0 * r.f_dc_0) * 255);
      packed[db + 25] = clampByte((0.5 + SH_C0 * r.f_dc_1) * 255);
      packed[db + 26] = clampByte((0.5 + SH_C0 * r.f_dc_2) * 255);
    } else if (has("red")) {
      const [cr, cg, cb] = unitColor(r.red, r.green, r.blue, types.red);
      packed[db + 24] = clampByte(cr * 255);
      packed[db + 25] = clampByte(cg * 255);
      packed[db + 26] = clampByte(cb * 255);
    } else {
      packed[db + 24] = 255;
      packed[db + 25] = 32;
      packed[db + 26] = 32;
    }
    packed[db + 27] = has("opacity") ? clampByte(sigmoid(r.opacity) * 255) : 255;
    if (has("rot_0")) {
      packQuatU8(packed, db + 28, r.rot_0, r.rot_1, r.rot_2, r.rot_3);
    } else {
      packed[db + 28] = 255;
      packed[db + 29] = 128;
      packed[db + 30] = 128;
      packed[db + 31] = 128;
    }
  }
  return { packed, variant };
}

export function toSplat32(input, filename = "", options = {}) {
  const buffer = asArrayBuffer(input);
  const compression = options.compression ?? 1;
  const fmt = detectFormat(buffer, filename);
  let packed;
  let variant = "3dgs";
  if (fmt === "ply") ({ packed, variant } = plyToPacked(buffer));
  else if (fmt === "splat44") packed = splat44ToPacked(buffer);
  else packed = splat32ToPacked(buffer);
  packed = downsamplePacked(packed, compression);
  const count = packed.byteLength / SPLAT32_ROW;
  return {
    format: fmt,
    variant,
    packed,
    count,
    bounds: boundsFromPacked(packed),
  };
}

export function packedToSplat44(packed) {
  const count = packed.byteLength / SPLAT32_ROW;
  const out = new ArrayBuffer(count * 44);
  const srcF = new Float32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const dstF = new Float32Array(out);
  const dstU = new Uint8Array(out);
  for (let i = 0; i < count; i++) {
    const sf = i * 8;
    const df = i * 11;
    const sb = i * SPLAT32_ROW;
    const db = i * 44;
    dstF[df] = srcF[sf];
    dstF[df + 1] = srcF[sf + 1];
    dstF[df + 2] = srcF[sf + 2];
    dstF[df + 3] = srcF[sf + 3];
    dstF[df + 4] = srcF[sf + 4];
    dstF[df + 5] = srcF[sf + 5];
    dstU[db + 24] = packed[sb + 24];
    dstU[db + 25] = packed[sb + 25];
    dstU[db + 26] = packed[sb + 26];
    dstU[db + 27] = packed[sb + 27];
    dstF[df + 7] = (packed[sb + 28] - 128) / 128;
    dstF[df + 8] = (packed[sb + 29] - 128) / 128;
    dstF[df + 9] = (packed[sb + 30] - 128) / 128;
    dstF[df + 10] = (packed[sb + 31] - 128) / 128;
  }
  return out;
}

export function packedToPly(packed) {
  const count = packed.byteLength / SPLAT32_ROW;
  const srcF = new Float32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const header =
    "ply\n" +
    "format binary_little_endian 1.0\n" +
    `element vertex ${count}\n` +
    "property float x\nproperty float y\nproperty float z\n" +
    "property float scale_0\nproperty float scale_1\nproperty float scale_2\n" +
    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n" +
    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n" +
    "property float opacity\n" +
    "end_header\n";
  const headerBytes = new TextEncoder().encode(header);
  const body = new ArrayBuffer(count * 14 * 4);
  const f = new Float32Array(body);
  const logit = (p) => {
    const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
    return Math.log(c / (1 - c));
  };
  for (let i = 0; i < count; i++) {
    const sf = i * 8;
    const sb = i * SPLAT32_ROW;
    const o = i * 14;
    f[o] = srcF[sf];
    f[o + 1] = srcF[sf + 1];
    f[o + 2] = srcF[sf + 2];
    f[o + 3] = Math.log(Math.max(srcF[sf + 3], 1e-8));
    f[o + 4] = Math.log(Math.max(srcF[sf + 4], 1e-8));
    f[o + 5] = Math.log(Math.max(srcF[sf + 5], 1e-8));
    f[o + 6] = (packed[sb + 28] - 128) / 128;
    f[o + 7] = (packed[sb + 29] - 128) / 128;
    f[o + 8] = (packed[sb + 30] - 128) / 128;
    f[o + 9] = (packed[sb + 31] - 128) / 128;
    f[o + 10] = (packed[sb + 24] / 255 - 0.5) / SH_C0;
    f[o + 11] = (packed[sb + 25] / 255 - 0.5) / SH_C0;
    f[o + 12] = (packed[sb + 26] / 255 - 0.5) / SH_C0;
    f[o + 13] = logit(packed[sb + 27] / 255);
  }
  const out = new Uint8Array(headerBytes.length + body.byteLength);
  out.set(headerBytes, 0);
  out.set(new Uint8Array(body), headerBytes.length);
  return out;
}

/** 12 floats: xyz, opacity, scale xyz, pad, quat wxyz */
export const GAUSSIAN_STRIDE = 12;
/** 16 RGB coefficients, tightly packed (48 floats). */
export const SH_STRIDE = 48;

export function boundsFromGaussians(gaussians) {
  const count = gaussians.length / GAUSSIAN_STRIDE;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  const zs = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const x = gaussians[i * 12], y = gaussians[i * 12 + 1], z = gaussians[i * 12 + 2];
    xs[i] = x; ys[i] = y; zs[i] = z;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) {
    minX = minY = minZ = -1;
    maxX = maxY = maxZ = 1;
  }
  const pct = (arr, t) => {
    arr.sort();
    const i = Math.min(arr.length - 1, Math.max(0, Math.floor(t * (arr.length - 1))));
    return arr[i];
  };
  let fminX = minX, fminY = minY, fminZ = minZ;
  let fmaxX = maxX, fmaxY = maxY, fmaxZ = maxZ;
  if (count >= 32) {
    fminX = pct(xs, 0.05); fmaxX = pct(xs, 0.95);
    fminY = pct(ys, 0.05); fmaxY = pct(ys, 0.95);
    fminZ = pct(zs, 0.05); fmaxZ = pct(zs, 0.95);
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(fminX + fmaxX) / 2, (fminY + fmaxY) / 2, (fminZ + fmaxZ) / 2],
    radius: Math.max(fmaxX - fminX, fmaxY - fminY, fmaxZ - fminZ) * 0.5 || 1,
  };
}

function packedToCloud(packed) {
  const n = packed.byteLength / SPLAT32_ROW;
  const srcF = new Float32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const gaussians = new Float32Array(n * GAUSSIAN_STRIDE);
  const sh = new Float32Array(n * SH_STRIDE);
  for (let i = 0; i < n; i++) {
    const sf = i * 8;
    const sb = i * SPLAT32_ROW;
    const g = i * 12;
    gaussians[g] = srcF[sf];
    gaussians[g + 1] = srcF[sf + 1];
    gaussians[g + 2] = srcF[sf + 2];
    gaussians[g + 3] = packed[sb + 27] / 255;
    gaussians[g + 4] = srcF[sf + 3];
    gaussians[g + 5] = srcF[sf + 4];
    gaussians[g + 6] = srcF[sf + 5];
    const qw = (packed[sb + 28] - 128) / 128;
    const qx = (packed[sb + 29] - 128) / 128;
    const qy = (packed[sb + 30] - 128) / 128;
    const qz = (packed[sb + 31] - 128) / 128;
    const qn = Math.hypot(qw, qx, qy, qz) || 1;
    gaussians[g + 8] = qw / qn;
    gaussians[g + 9] = qx / qn;
    gaussians[g + 10] = qy / qn;
    gaussians[g + 11] = qz / qn;
    const s = i * SH_STRIDE;
    sh[s] = (packed[sb + 24] / 255 - 0.5) / SH_C0;
    sh[s + 1] = (packed[sb + 25] / 255 - 0.5) / SH_C0;
    sh[s + 2] = (packed[sb + 26] / 255 - 0.5) / SH_C0;
  }
  return {
    gaussians,
    sh,
    shDegree: 0,
    count: n,
    bounds: boundsFromGaussians(gaussians),
    format: "splat32",
    variant: "3dgs",
  };
}

function plyToCloud(buffer) {
  const { format, vertexCount, properties, headerEnd } = decodePlyHeader(buffer);
  const { offsets, types, rowSize, has } = indexProperties(properties);
  const variant = plyVariantFromProperties(properties.map((p) => p.name));

  const nRest = countRestCoefficients(has);
  const nCoeffsPerColor = nRest / 3;
  const shDegree = shDegreeFromRestCount(nRest);

  const gaussians = new Float32Array(vertexCount * GAUSSIAN_STRIDE);
  const sh = new Float32Array(vertexCount * SH_STRIDE);

  const fill = (i, get) => {
    const g = i * 12;
    gaussians[g] = get("x");
    gaussians[g + 1] = get("y");
    gaussians[g + 2] = get("z");
    gaussians[g + 3] = has("opacity") ? sigmoid(get("opacity")) : 1;
    const [sx, sy, sz] = linearScales(get, has, variant);
    gaussians[g + 4] = sx;
    gaussians[g + 5] = sy;
    gaussians[g + 6] = sz;
    if (has("rot_0")) {
      const qw = get("rot_0");
      const qx = get("rot_1");
      const qy = get("rot_2");
      const qz = get("rot_3");
      const qn = Math.hypot(qw, qx, qy, qz) || 1;
      gaussians[g + 8] = qw / qn;
      gaussians[g + 9] = qx / qn;
      gaussians[g + 10] = qy / qn;
      gaussians[g + 11] = qz / qn;
    } else {
      gaussians[g + 8] = 1;
    }
    const s = i * SH_STRIDE;
    if (has("f_dc_0")) {
      sh[s] = get("f_dc_0");
      sh[s + 1] = get("f_dc_1");
      sh[s + 2] = get("f_dc_2");
    } else if (has("red")) {
      const [r, gch, b] = unitColor(get("red"), get("green"), get("blue"), types.red);
      sh[s] = (r - 0.5) / SH_C0;
      sh[s + 1] = (gch - 0.5) / SH_C0;
      sh[s + 2] = (b - 0.5) / SH_C0;
    }
    const maxK = Math.min(15, nCoeffsPerColor | 0);
    for (let k = 0; k < maxK; k++) {
      sh[s + 3 + k * 3] = get(`f_rest_${k}`);
      sh[s + 4 + k * 3] = get(`f_rest_${nCoeffsPerColor + k}`);
      sh[s + 5 + k * 3] = get(`f_rest_${2 * nCoeffsPerColor + k}`);
    }
  };

  if (format === "ascii") {
    const lines = asciiRows(buffer, headerEnd, vertexCount);
    const names = properties.map((p) => p.name);
    for (let i = 0; i < vertexCount; i++) {
      const parts = lines[i].trim().split(/\s+/);
      const rec = {};
      for (let p = 0; p < names.length; p++) rec[names[p]] = parseFloat(parts[p]);
      fill(i, (name) => (Number.isFinite(rec[name]) ? rec[name] : 0));
    }
  } else {
    const le = format === "binary_le";
    assertBinaryBody(buffer, headerEnd, vertexCount, rowSize);
    const view = new DataView(buffer, headerEnd);
    for (let i = 0; i < vertexCount; i++) {
      const base = i * rowSize;
      fill(i, (name) =>
        has(name) ? readNumeric(view, base + offsets[name], types[name], le) : 0
      );
    }
  }

  return {
    gaussians,
    sh,
    shDegree,
    count: vertexCount,
    bounds: boundsFromGaussians(gaussians),
    format: "ply",
    variant,
  };
}

/**
 * Header-only summary of a PLY for HUDs and tests (no vertex data is decoded):
 *   { vertexCount, properties: [names], shDegree, variant, encoding }
 * `encoding` is "binary_le" | "binary_be" | "ascii". Throws if the buffer does
 * not start with the "ply" magic or its header is malformed.
 */
export function describePly(input) {
  const buffer = asArrayBuffer(input);
  if (!isPlyMagic(new Uint8Array(buffer))) {
    throw new Error("Not a PLY buffer (missing 'ply' magic)");
  }
  const header = decodePlyHeader(buffer);
  const properties = header.properties.map((p) => p.name);
  const set = new Set(properties);
  const has = (name) => set.has(name);
  return {
    vertexCount: header.vertexCount,
    properties,
    shDegree: shDegreeFromRestCount(countRestCoefficients(has)),
    variant: plyVariantFromProperties(properties),
    encoding: header.format,
  };
}

/**
 * Full-precision 3DGS cloud: float covariance + SH0–3.
 * Compact .splat files only recover SH degree 0.
 */
export function toGaussianCloud(input, filename = "") {
  const buffer = asArrayBuffer(input);
  const fmt = detectFormat(buffer, filename);
  if (fmt === "ply") return plyToCloud(buffer);
  const packed = fmt === "splat44" ? splat44ToPacked(buffer) : splat32ToPacked(buffer);
  const cloud = packedToCloud(packed);
  cloud.format = fmt;
  return cloud;
}
