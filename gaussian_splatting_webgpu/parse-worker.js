import { toGaussianCloud, boundsFromGaussians } from "../shared/splat-io.js";

// Copia vendorizada (vendor/gaussforge/, ver NOTICE.md): funciona sin red.
const GAUSSFORGE_VENDOR_URL = new URL("../vendor/gaussforge/index.web.js", import.meta.url).href;
// Respaldo por CDN, sólo si la copia vendorizada no carga.
const GAUSSFORGE_URL =
  "https://cdn.jsdelivr.net/npm/@gaussforge/wasm@0.6.0/dist/index.web.js";
// Orden de intento; el último recurso es el decodificador integrado (splat-io.js).
const GAUSSFORGE_SOURCES = [
  { source: "vendor", url: GAUSSFORGE_VENDOR_URL },
  { source: "cdn", url: GAUSSFORGE_URL },
];

let forgePromise = null; // carga en curso o resuelta; se comparte entre mensajes concurrentes
let forgeSource = ""; // "vendor" | "cdn" una vez cargado
let forgeLoadError = "";
let lastInput = null;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));
}

function downsampleCloud(gaussians, sh, compression) {
  const factor = Math.max(1, Math.min(10, compression | 0 || 1));
  const count = gaussians.length / 12;
  if (factor === 1) return { gaussians, sh, count };
  const keep = Math.max(1, Math.floor(count / factor));
  const outG = new Float32Array(keep * 12);
  const outSh = new Float32Array(keep * 48);
  for (let i = 0; i < keep; i++) {
    const src = Math.floor((i * count) / keep);
    outG.set(gaussians.subarray(src * 12, src * 12 + 12), i * 12);
    outSh.set(sh.subarray(src * 48, src * 48 + 48), i * 48);
  }
  return { gaussians: outG, sh: outSh, count: keep };
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

function shDegreeFromIr(ir) {
  const n = ir.numPoints | 0;
  const metaDeg = (ir.meta && ir.meta.shDegree) || 0;
  const rest = ir.sh && ir.sh.length ? ir.sh.length : 0;
  const per = n ? rest / n : 0;
  const fromLen = per >= 45 ? 3 : per >= 24 ? 2 : per >= 9 ? 1 : 0;
  return Math.min(3, Math.max(metaDeg, fromLen));
}

function irToCloud(ir) {
  const n = ir.numPoints | 0;
  const gaussians = new Float32Array(n * 12);
  const sh = new Float32Array(n * 48);
  const pos = ir.positions;
  const scl = ir.scales;
  const rot = ir.rotations;
  const alp = ir.alphas;
  const col = ir.colors;
  const rest = ir.sh;
  const degree = shDegreeFromIr(ir);
  const restStride = degree > 0 ? ((degree + 1) * (degree + 1) - 1) * 3 : 0;
  for (let i = 0; i < n; i++) {
    const g = i * 12;
    gaussians[g] = pos[i * 3];
    gaussians[g + 1] = pos[i * 3 + 1];
    gaussians[g + 2] = pos[i * 3 + 2];
    gaussians[g + 3] = sigmoid(alp[i]);
    gaussians[g + 4] = Math.exp(scl[i * 3]);
    gaussians[g + 5] = Math.exp(scl[i * 3 + 1]);
    gaussians[g + 6] = Math.exp(scl[i * 3 + 2]);
    const qw = rot[i * 4], qx = rot[i * 4 + 1], qy = rot[i * 4 + 2], qz = rot[i * 4 + 3];
    const qn = Math.hypot(qw, qx, qy, qz) || 1;
    gaussians[g + 8] = qw / qn;
    gaussians[g + 9] = qx / qn;
    gaussians[g + 10] = qy / qn;
    gaussians[g + 11] = qz / qn;
    const s = i * 48;
    sh[s] = col[i * 3];
    sh[s + 1] = col[i * 3 + 1];
    sh[s + 2] = col[i * 3 + 2];
    if (restStride && rest && rest.length >= (i + 1) * restStride) {
      sh.set(rest.subarray(i * restStride, i * restStride + Math.min(restStride, 45)), s + 3);
    }
  }
  return { gaussians, sh, shDegree: degree, count: n };
}

function finishCloud(cloud, extra) {
  const reduced = downsampleCloud(cloud.gaussians, cloud.sh, extra.compression);
  return {
    decoder: extra.decoder,
    decoderSource: extra.decoderSource || "",
    format: extra.format,
    shDegree: cloud.shDegree,
    count: reduced.count,
    gaussians: reduced.gaussians,
    sh: reduced.sh,
    bounds: boundsFromGaussians(reduced.gaussians),
    warning: extra.warning || "",
    version: extra.version || "",
  };
}

function errorText(err) {
  return err && err.message ? err.message : String(err);
}

async function loadForgeFrom(entry) {
  const mod = await import(entry.url);
  if (typeof mod.createGaussForge !== "function") {
    throw new Error("el módulo no exporta createGaussForge");
  }
  return mod.createGaussForge();
}

// Intenta cada origen en orden; devuelve la instancia o lanza con todos los motivos.
async function loadForge() {
  const errors = [];
  for (const entry of GAUSSFORGE_SOURCES) {
    try {
      const gf = await loadForgeFrom(entry);
      const version = gf.getVersion ? gf.getVersion() : "?";
      console.info(`[parse-worker] GaussForge ${version} cargado desde ${entry.source} (${entry.url})`);
      forgeSource = entry.source;
      return gf;
    } catch (err) {
      const msg = errorText(err);
      errors.push(`${entry.source}: ${msg}`);
      console.warn(`[parse-worker] GaussForge no cargó desde ${entry.source} (${entry.url}): ${msg}`);
    }
  }
  forgeLoadError = errors.join("; ");
  console.warn(`[parse-worker] GaussForge no disponible; se usará el decodificador integrado (${forgeLoadError})`);
  throw new Error(`GaussForge WASM unavailable (${forgeLoadError})`);
}

// Memoriza la promesa (no un booleano): los mensajes que llegan mientras el
// módulo aún se importa esperan la misma carga en vez de caer al decodificador
// integrado. Si falla, la promesa rechazada se conserva y todos ven el mismo motivo.
function getForge() {
  if (!forgePromise) forgePromise = loadForge();
  return forgePromise;
}

async function decodeWithForge(buffer, name, compression) {
  const gf = await getForge();
  const format = detectGaussFormat(buffer, name);
  const bytes = new Uint8Array(buffer instanceof Uint8Array ? buffer : buffer);
  lastInput = { bytes: new Uint8Array(bytes), format, name };
  const result = await gf.read(bytes, format);
  if (result.error) throw new Error(result.error);
  const ir = result.data;
  if (!ir || !ir.numPoints) throw new Error("GaussForge returned an empty cloud");
  return finishCloud(irToCloud(ir), {
    decoder: "gaussforge",
    decoderSource: forgeSource,
    format,
    compression,
    warning: result.warning || "",
    version: gf.getVersion ? gf.getVersion() : "",
  });
}

function decodeFallback(buffer, name, compression) {
  const cloud = toGaussianCloud(buffer, name);
  return finishCloud(cloud, {
    decoder: "builtin",
    format: cloud.format,
    compression,
  });
}

self.onmessage = async (event) => {
  const data = event.data || {};
  const id = data.id;

  if (data.type === "convert") {
    try {
      const gf = await getForge();
      if (!lastInput) throw new Error("No decoded model to export");
      const converted = await gf.convert(lastInput.bytes, lastInput.format, data.outFormat || "ply");
      if (converted.error) throw new Error(converted.error);
      self.postMessage(
        { id, ok: true, type: "convert", outFormat: data.outFormat, bytes: converted.data },
        [converted.data.buffer]
      );
    } catch (err) {
      self.postMessage({ id, ok: false, type: "convert", error: errorText(err) });
    }
    return;
  }

  try {
    let decoded;
    try {
      decoded = await decodeWithForge(data.buffer, data.name || "", data.compression || 1);
    } catch (forgeErr) {
      console.warn(`[parse-worker] GaussForge falló con ${data.name || "?"}: ${errorText(forgeErr)}`);
      decoded = decodeFallback(data.buffer, data.name || "", data.compression || 1);
      decoded.warning = `GaussForge: ${errorText(forgeErr)}; used built-in decoder`;
    }
    const transfer = [decoded.gaussians.buffer, decoded.sh.buffer];
    self.postMessage({ id, ok: true, ...decoded }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: errorText(err) });
  }
};
