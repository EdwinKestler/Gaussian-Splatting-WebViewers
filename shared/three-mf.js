import { validateMesh } from "./mesh-ops.js";

const encoder = new TextEncoder();
const MODEL_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";
const REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships";
const START_PART_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel";
const MODEL_CONTENT_TYPE = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml";

const xmlEscape = (value) => String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const number = (value) => Number(value).toPrecision(9).replace(/(?:\.0+|(?:(\.\d*?)0+))$/, "$1");

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function concat(parts) {
  const size = parts.reduce((sum, part) => sum + part.byteLength, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.byteLength; }
  return out;
}

function zipStored(entries) {
  const locals = [], central = [];
  let offset = 0;
  for (const [path, content] of entries) {
    const name = encoder.encode(path);
    const data = typeof content === "string" ? encoder.encode(content) : content;
    const crc = crc32(data);
    const local = new Uint8Array(30 + name.length);
    const l = new DataView(local.buffer);
    l.setUint32(0, 0x04034b50, true); l.setUint16(4, 20, true); l.setUint16(6, 0x0800, true);
    l.setUint16(8, 0, true); l.setUint16(10, 0, true); l.setUint16(12, 0x0021, true);
    l.setUint32(14, crc, true); l.setUint32(18, data.length, true); l.setUint32(22, data.length, true);
    l.setUint16(26, name.length, true); l.setUint16(28, 0, true); local.set(name, 30);
    locals.push(local, data);

    const cd = new Uint8Array(46 + name.length);
    const c = new DataView(cd.buffer);
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
    c.setUint16(10, 0, true); c.setUint16(12, 0, true); c.setUint16(14, 0x0021, true);
    c.setUint32(16, crc, true); c.setUint32(20, data.length, true); c.setUint32(24, data.length, true);
    c.setUint16(28, name.length, true); c.setUint16(30, 0, true); c.setUint16(32, 0, true); c.setUint16(34, 0, true); c.setUint16(36, 0, true);
    c.setUint32(38, 0, true); c.setUint32(42, offset, true); cd.set(name, 46);
    central.push(cd);
    offset += local.length + data.length;
  }
  const centralSize = central.reduce((sum, part) => sum + part.length, 0);
  const end = new Uint8Array(22);
  const e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true); e.setUint16(4, 0, true); e.setUint16(6, 0, true);
  e.setUint16(8, entries.length, true); e.setUint16(10, entries.length, true);
  e.setUint32(12, centralSize, true); e.setUint32(16, offset, true); e.setUint16(20, 0, true);
  return concat([...locals, ...central, end]);
}

function averageColor(colors, vertices) {
  if (!colors || colors.length < vertices * 3) return "#B3B3B3";
  const sum = [0, 0, 0];
  for (let v = 0; v < vertices; v++) for (let a = 0; a < 3; a++) sum[a] += Math.max(0, Math.min(1, colors[v * 3 + a]));
  return `#${sum.map((v) => Math.round((v / vertices) * 255).toString(16).padStart(2, "0")).join("").toUpperCase()}`;
}

/** Encode one printable triangle mesh as a deterministic, uncompressed 3MF OPC package. */
export function encode3mf(mesh, { name = "Gaussian Splatting mesh", requireWatertight = true } = {}) {
  const validation = validateMesh(mesh);
  if (!validation.geometryValid) {
    throw new Error(
      `3MF geometry check failed: ${validation.nonFiniteVertices} non-finite vertices, ${validation.invalidIndexTriangles} invalid-index triangles, ${validation.degenerateTriangles} degenerate triangles, ${validation.duplicateTriangles} duplicate triangles`
    );
  }
  if (requireWatertight && !validation.printable) {
    throw new Error(`3MF print check failed: ${validation.boundaryEdges} boundary edges, ${validation.nonManifoldEdges} non-manifold edges, ${validation.inconsistentWindingEdges} winding conflicts`);
  }
  const vertices = [];
  for (let v = 0; v < validation.vertices; v++) {
    vertices.push(`<vertex x="${number(mesh.positions[v * 3])}" y="${number(mesh.positions[v * 3 + 1])}" z="${number(mesh.positions[v * 3 + 2])}"/>`);
  }
  const triangles = [];
  for (let t = 0; t < mesh.indices.length; t += 3) triangles.push(`<triangle v1="${mesh.indices[t]}" v2="${mesh.indices[t + 1]}" v3="${mesh.indices[t + 2]}"/>`);
  const model = `<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="${MODEL_NS}"><metadata name="Title">${xmlEscape(name)}</metadata><resources><basematerials id="1"><base name="Vertex color average" displaycolor="${averageColor(mesh.colors, validation.vertices)}"/></basematerials><object id="2" type="model" pid="1" pindex="0" name="${xmlEscape(name)}"><mesh><vertices>${vertices.join("")}</vertices><triangles>${triangles.join("")}</triangles></mesh></object></resources><build><item objectid="2"/></build></model>`;
  const contentTypes = `<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="${MODEL_CONTENT_TYPE}"/></Types>`;
  const relationships = `<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="${REL_NS}"><Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="${START_PART_REL}"/></Relationships>`;
  return zipStored([
    ["[Content_Types].xml", contentTypes],
    ["_rels/.rels", relationships],
    ["3D/3dmodel.model", model],
  ]).buffer;
}

/** Minimal reader for the STORE-only packages emitted by encode3mf (tests/inspection). */
export function read3mfFiles(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const files = new Map();
  let offset = 0;
  while (offset + 30 <= bytes.length) {
    const view = new DataView(bytes.buffer, bytes.byteOffset + offset, bytes.byteLength - offset);
    if (view.getUint32(0, true) !== 0x04034b50) break;
    const method = view.getUint16(8, true);
    if (method !== 0) throw new Error(`unsupported ZIP method ${method}`);
    const size = view.getUint32(18, true), nameLength = view.getUint16(26, true), extraLength = view.getUint16(28, true);
    const nameStart = offset + 30, dataStart = nameStart + nameLength + extraLength;
    const name = new TextDecoder().decode(bytes.subarray(nameStart, nameStart + nameLength));
    files.set(name, bytes.slice(dataStart, dataStart + size));
    offset = dataStart + size;
  }
  return files;
}

export const THREE_MF = { MODEL_NS, START_PART_REL, MODEL_CONTENT_TYPE };
