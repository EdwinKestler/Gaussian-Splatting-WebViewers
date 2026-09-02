import { toSplat32 } from "../shared/splat-io.js";

const GAUSSFORGE_URL =
  "https://cdn.jsdelivr.net/npm/@gaussforge/wasm@0.6.0/dist/index.web.js";
const SH_C0 = 0.28209479177387814;

let forge = null;
let forgeTried = false;
let lastInput = null;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));
}

function clampByte(v) {
  return Math.max(0, Math.min(255, v | 0));
}

function packQuatU8(out, offset, w, x, y, z) {
  const n = Math.hypot(w, x, y, z) || 1;
  out[offset] = clampByte(Math.round((w / n) * 128 + 128));
  out[offset + 1] = clampByte(Math.round((x / n) * 128 + 128));
  out[offset + 2] = clampByte(Math.round((y / n) * 128 + 128));
  out[offset + 3] = clampByte(Math.round((z / n) * 128 + 128));
}

function boundsFromPacked(packed) {
  const f = new Float32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const count = packed.byteLength / 32;
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

function downsample(packed, sh1, compression) {
  const factor = Math.max(1, Math.min(10, compression | 0 || 1));
  if (factor === 1) return { packed, sh1 };
  const count = packed.byteLength / 32;
  const keep = Math.max(1, Math.floor(count / factor));
  const out = new Uint8Array(keep * 32);
  let outSh = null;
  if (sh1 && sh1.length >= count * 9) outSh = new Float32Array(keep * 9);
  for (let i = 0; i < keep; i++) {
    const src = Math.floor((i * count) / keep);
    out.set(packed.subarray(src * 32, src * 32 + 32), i * 32);
    if (outSh) outSh.set(sh1.subarray(src * 9, src * 9 + 9), i * 9);
  }
  return { packed: out, sh1: outSh };
}

function detectGaussFormat(buffer, filename = "") {
  const name = String(filename).toLowerCase();
  if (name.endsWith(".compressed.ply")) return "compressed.ply";
  if (name.endsWith(".ply")) return "ply";
  if (name.endsWith(".spz")) return "spz";
  if (name.endsWith(".ksplat")) return "ksplat";
  if (name.endsWith(".sog")) return "sog";
  if (name.endsWith(".splat")) return "splat";
  const u8 = new Uint8Array(buffer);
  if (u8.length >= 4 && u8[0] === 0x70 && u8[1] === 0x6c && u8[2] === 0x79) return "ply";
  if (u8.length >= 2 && u8[0] === 0x1f && u8[1] === 0x8b) return "spz";
  if (u8.length >= 4 && u8[0] === 0x50 && u8[1] === 0x4b) return "sog";
  if (buffer.byteLength % 32 === 0) return "splat";
  return "ply";
}

function irToPacked(ir) {
  const n = ir.numPoints | 0;
  const packed = new Uint8Array(n * 32);
  const f = new Float32Array(packed.buffer);
  const pos = ir.positions;
  const scl = ir.scales;
  const rot = ir.rotations;
  const alp = ir.alphas;
  const col = ir.colors;
  for (let i = 0; i < n; i++) {
    const df = i * 8;
    const db = i * 32;
    f[df] = pos[i * 3];
    f[df + 1] = pos[i * 3 + 1];
    f[df + 2] = pos[i * 3 + 2];
    f[df + 3] = Math.exp(scl[i * 3]);
    f[df + 4] = Math.exp(scl[i * 3 + 1]);
    f[df + 5] = Math.exp(scl[i * 3 + 2]);
    packed[db + 24] = clampByte((0.5 + SH_C0 * col[i * 3]) * 255);
    packed[db + 25] = clampByte((0.5 + SH_C0 * col[i * 3 + 1]) * 255);
    packed[db + 26] = clampByte((0.5 + SH_C0 * col[i * 3 + 2]) * 255);
    packed[db + 27] = clampByte(sigmoid(alp[i]) * 255);
    packQuatU8(packed, db + 28, rot[i * 4], rot[i * 4 + 1], rot[i * 4 + 2], rot[i * 4 + 3]);
  }
  let sh1 = null;
  const degree = (ir.meta && ir.meta.shDegree) || 0;
  if (degree >= 1 && ir.sh && ir.sh.length >= n * 9) {
    sh1 = new Float32Array(n * 9);
    const stride = ((degree + 1) * (degree + 1) - 1) * 3;
    for (let i = 0; i < n; i++) {
      sh1.set(ir.sh.subarray(i * stride, i * stride + 9), i * 9);
    }
  }
  return { packed, sh1, shDegree: degree };
}

async function getForge() {
  if (forge) return forge;
  if (forgeTried) return null;
  forgeTried = true;
  const mod = await import(GAUSSFORGE_URL);
  forge = await mod.createGaussForge();
  return forge;
}

async function decodeWithForge(buffer, name, compression) {
  const gf = await getForge();
  if (!gf) throw new Error("GaussForge WASM unavailable");
  const format = detectGaussFormat(buffer, name);
  const bytes = new Uint8Array(buffer instanceof Uint8Array ? buffer : buffer);
  lastInput = { bytes: new Uint8Array(bytes), format, name };
  const result = await gf.read(bytes, format);
  if (result.error) throw new Error(result.error);
  const ir = result.data;
  if (!ir || !ir.numPoints) throw new Error("GaussForge returned an empty cloud");
  const packedFull = irToPacked(ir);
  const reduced = downsample(packedFull.packed, packedFull.sh1, compression);
  return {
    decoder: "gaussforge",
    format,
    shDegree: packedFull.shDegree,
    count: reduced.packed.byteLength / 32,
    packed: reduced.packed,
    sh1: reduced.sh1,
    bounds: boundsFromPacked(reduced.packed),
    warning: result.warning || "",
    version: gf.getVersion ? gf.getVersion() : "",
  };
}

function decodeFallback(buffer, name, compression) {
  const result = toSplat32(buffer, name, { compression: compression || 1 });
  return {
    decoder: "builtin",
    format: result.format,
    shDegree: 0,
    count: result.count,
    packed: result.packed,
    sh1: null,
    bounds: result.bounds,
    warning: "",
    version: "",
  };
}

self.onmessage = async (event) => {
  const data = event.data || {};
  const id = data.id;

  if (data.type === "convert") {
    try {
      const gf = await getForge();
      if (!gf) throw new Error("GaussForge WASM unavailable");
      if (!lastInput) throw new Error("No decoded model to export");
      const converted = await gf.convert(lastInput.bytes, lastInput.format, data.outFormat || "ply");
      if (converted.error) throw new Error(converted.error);
      self.postMessage(
        { id, ok: true, type: "convert", outFormat: data.outFormat, bytes: converted.data },
        [converted.data.buffer]
      );
    } catch (err) {
      self.postMessage({
        id,
        ok: false,
        type: "convert",
        error: err && err.message ? err.message : String(err),
      });
    }
    return;
  }

  try {
    let decoded;
    try {
      decoded = await decodeWithForge(data.buffer, data.name || "", data.compression || 1);
    } catch (forgeErr) {
      decoded = decodeFallback(data.buffer, data.name || "", data.compression || 1);
      decoded.warning = `GaussForge: ${forgeErr.message || forgeErr}; used built-in decoder`;
    }
    const transfer = [decoded.packed.buffer];
    if (decoded.sh1) transfer.push(decoded.sh1.buffer);
    self.postMessage({ id, ok: true, ...decoded }, transfer);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err && err.message ? err.message : String(err),
    });
  }
};
