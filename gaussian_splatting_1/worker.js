import { toSplat32 } from "../shared/splat-io.js";

function sortSplats(matrices, view) {
  const vertexCount = matrices.length / 16;
  let maxDepth = -Infinity;
  let minDepth = Infinity;
  const depthList = new Float32Array(vertexCount);
  const sizeList = new Int32Array(depthList.buffer);
  for (let i = 0; i < vertexCount; i++) {
    const depth =
      view[0] * matrices[i * 16 + 12] -
      view[1] * matrices[i * 16 + 13] -
      view[2] * matrices[i * 16 + 14];
    depthList[i] = depth;
    if (depth > maxDepth) maxDepth = depth;
    if (depth < minDepth) minDepth = depth;
  }
  const depthInv = (256 * 256 - 1) / Math.max(maxDepth - minDepth, 1e-6);
  const counts0 = new Uint32Array(256 * 256);
  for (let i = 0; i < vertexCount; i++) {
    sizeList[i] = ((depthList[i] - minDepth) * depthInv) | 0;
    counts0[sizeList[i]]++;
  }
  const starts0 = new Uint32Array(256 * 256);
  for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
  const depthIndex = new Uint32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) depthIndex[starts0[sizeList[i]]++] = i;
  const sortedMatrices = new Float32Array(vertexCount * 16);
  for (let j = 0; j < vertexCount; j++) {
    const i = depthIndex[j];
    sortedMatrices.set(matrices.subarray(i * 16, i * 16 + 16), j * 16);
  }
  return sortedMatrices;
}

function packedToMatrices(packed) {
  const count = packed.byteLength / 32;
  const f = new Float32Array(packed.buffer, packed.byteOffset, packed.byteLength / 4);
  const u = packed;
  const matrices = new Float32Array(count * 16);
  const tmp = new Float32Array(16);
  for (let i = 0; i < count; i++) {
    const qx = (u[32 * i + 29] - 128) / 128;
    const qy = (u[32 * i + 30] - 128) / 128;
    const qz = (u[32 * i + 31] - 128) / 128;
    const qw = (u[32 * i + 28] - 128) / 128;
    const x = f[8 * i];
    const y = f[8 * i + 1];
    const z = f[8 * i + 2];
    const sx = f[8 * i + 3];
    const sy = f[8 * i + 4];
    const sz = f[8 * i + 5];
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const m00 = (1 - 2 * (yy + zz)) * sx;
    const m01 = (2 * (xy + wz)) * sx;
    const m02 = (2 * (xz - wy)) * sx;
    const m10 = (2 * (xy - wz)) * sy;
    const m11 = (1 - 2 * (xx + zz)) * sy;
    const m12 = (2 * (yz + wx)) * sy;
    const m20 = (2 * (xz + wy)) * sz;
    const m21 = (2 * (yz - wx)) * sz;
    const m22 = (1 - 2 * (xx + yy)) * sz;
    tmp[0] = m00 * m00 + m10 * m10 + m20 * m20;
    tmp[1] = m00 * m01 + m10 * m11 + m20 * m21;
    tmp[2] = m00 * m02 + m10 * m12 + m20 * m22;
    tmp[4] = tmp[1];
    tmp[5] = m01 * m01 + m11 * m11 + m21 * m21;
    tmp[6] = m01 * m02 + m11 * m12 + m21 * m22;
    tmp[8] = tmp[2];
    tmp[9] = tmp[6];
    tmp[10] = m02 * m02 + m12 * m12 + m22 * m22;
    tmp[12] = x;
    tmp[13] = y;
    tmp[14] = z;
    tmp[3] = u[32 * i + 24] / 255;
    tmp[7] = u[32 * i + 25] / 255;
    tmp[11] = u[32 * i + 26] / 255;
    tmp[15] = u[32 * i + 27] / 255;
    matrices.set(tmp, i * 16);
  }
  return matrices;
}

let matrices = null;

self.onmessage = (e) => {
  const data = e.data;
  if (data.buffer) {
    try {
      const parsed = toSplat32(data.buffer, data.name || "", {
        compression: data.compression || 1,
      });
      matrices = packedToMatrices(parsed.packed);
      const transferable = matrices.slice();
      self.postMessage(
        {
          ok: true,
          count: parsed.count,
          format: parsed.format,
          bounds: parsed.bounds,
          matrices: transferable.buffer,
        },
        [transferable.buffer]
      );
    } catch (err) {
      self.postMessage({ ok: false, error: err.message || String(err) });
    }
    return;
  }
  if (data.matrices) {
    matrices = new Float32Array(data.matrices);
  }
  if (data.view && matrices) {
    const view = new Float32Array(data.view);
    const sorted = sortSplats(matrices, view);
    self.postMessage({ sortedMatrices: sorted.buffer }, [sorted.buffer]);
  }
};
