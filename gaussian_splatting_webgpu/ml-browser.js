/**
 * In-browser models for F3/F4 through transformers.js (ONNX Runtime Web):
 *   - SAM 2 (or SlimSAM) promptable masks: one image embedding per view, one
 *     decoder run per point prompt.
 *   - CLIP image / text embeddings for per-instance semantic search.
 *
 * Both are optional: the viewer works without them (sidecar backends) and the
 * library is imported lazily, so the plain viewer never pays for it. Weights
 * come from vendor/ml/ when scripts/download-ml-models.sh has been run (works
 * offline; the directory is gitignored), otherwise from jsDelivr + the Hugging
 * Face Hub with the browser Cache API keeping them between sessions.
 */

export const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js";
export const SAM_MODELS = Object.freeze({
  "sam2-tiny": { id: "onnx-community/sam2.1-hiera-tiny-ONNX", kind: "sam2", dtype: "fp16" },
  "sam2-small": { id: "onnx-community/sam2.1-hiera-small-ONNX", kind: "sam2", dtype: "fp16" },
  slimsam: { id: "Xenova/slimsam-77-uniform", kind: "sam", dtype: "fp32" },
});
export const CLIP_MODEL = "Xenova/clip-vit-base-patch32";

/** Local copy made by scripts/download-ml-models.sh (served from the repo root). */
export const LOCAL_ML_URL = new URL("../vendor/ml/", import.meta.url).href;

let libPromise = null;
let localManifestPromise = null;

/** manifest.json of vendor/ml/ when the weights were downloaded locally, else null. */
export function localManifest(base = LOCAL_ML_URL) {
  if (!localManifestPromise) {
    localManifestPromise = fetch(base + "manifest.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
  }
  return localManifestPromise;
}

/**
 * Load transformers.js once. With a local vendor/ml/ the library, its ONNX
 * Runtime wasm files and the model weights all come from this server (works
 * offline); otherwise the CDN and the Hugging Face Hub are used and the
 * browser Cache API keeps the weights between sessions.
 */
export function loadTransformers(url = TRANSFORMERS_URL) {
  if (!libPromise) {
    libPromise = (async () => {
      const manifest = await localManifest();
      const entry = manifest ? LOCAL_ML_URL + (manifest.transformers?.entry || "transformers/transformers.min.js") : url;
      const mod = await import(entry);
      if (manifest) {
        mod.env.allowRemoteModels = false;
        mod.env.allowLocalModels = true;
        // Same-origin path, not a full URL: transformers.js only probes local
        // files (tokenizer_config.json etc.) when the path has no http(s) scheme.
        mod.env.localModelPath = new URL(LOCAL_ML_URL).pathname + "models/";
        mod.env.backends.onnx.wasm.wasmPaths = LOCAL_ML_URL + "transformers/";
        console.info(`[ml] transformers.js ${manifest.transformers?.version || ""} local (${(manifest.bytes / 1e6).toFixed(0)} MB en vendor/ml)`);
      } else {
        mod.env.allowLocalModels = false;
        mod.env.useBrowserCache = true;
      }
      mod.env.__local = !!manifest;
      return mod;
    })();
  }
  return libPromise;
}

/** Pick the ONNX Runtime device: WebGPU when it is a real adapter, else WASM. */
export async function pickDevice(preferred = "auto") {
  if (preferred !== "auto") return preferred;
  if (!navigator.gpu) return "wasm";
  try {
    const adapter = await navigator.gpu.requestAdapter();
    const arch = (adapter && adapter.info && adapter.info.architecture) || "";
    // SwiftShader (software) is slower than WASM for ONNX; keep WebGPU for real GPUs.
    return adapter && !/swiftshader|software/i.test(arch) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

/** RGBA pixels → transformers.js RawImage (RGB). */
function rawImageFromRgba(lib, rgba, width, height) {
  const rgb = new Uint8ClampedArray(width * height * 3);
  for (let p = 0, q = 0; p < width * height; p++, q += 3) {
    rgb[q] = rgba[p * 4];
    rgb[q + 1] = rgba[p * 4 + 1];
    rgb[q + 2] = rgba[p * 4 + 2];
  }
  return new lib.RawImage(rgb, width, height, 3);
}

/**
 * Promptable segmentation (SAM 2 / SlimSAM) for one view.
 * Usage:
 *   const sam = await BrowserSam.load({ model: "sam2-tiny" });
 *   await sam.setImage(rgba, W, H);            // encoder once per view
 *   const m = await sam.maskAt([[x, y]]);       // decoder per prompt (pixel coords of the view)
 *   const labels = await sam.labelMask(prompts) // one u32 label map from many point prompts
 */
export class BrowserSam {
  constructor(lib, model, processor, meta) {
    this.lib = lib;
    this.model = model;
    this.processor = processor;
    this.meta = meta;
    this.image = null;
    this.inputs = null;
    this.embeddings = null;
  }

  static async load({ model = "sam2-tiny", device = "auto", dtype = null, progress = null } = {}) {
    const meta = SAM_MODELS[model];
    if (!meta) throw new Error(`modelo SAM desconocido: ${model} (${Object.keys(SAM_MODELS).join(", ")})`);
    const lib = await loadTransformers();
    const dev = await pickDevice(device);
    const opts = { device: dev, dtype: dtype || meta.dtype, progress_callback: progress || undefined };
    const ModelClass = meta.kind === "sam2" ? lib.Sam2Model : lib.SamModel;
    let m;
    try {
      m = await ModelClass.from_pretrained(meta.id, opts);
    } catch (err) {
      if (dev === "webgpu") {
        console.warn(`[ml] ${meta.id} no cargó con WebGPU (${err.message}); reintentando con WASM`);
        m = await ModelClass.from_pretrained(meta.id, { ...opts, device: "wasm" });
        opts.device = "wasm";
      } else throw err;
    }
    const processor = await lib.AutoProcessor.from_pretrained(meta.id);
    const sam = new BrowserSam(lib, m, processor, { ...meta, key: model, device: opts.device, dtype: opts.dtype });
    return sam;
  }

  /** Encode one view (RGBA pixels, top-left origin). */
  async setImage(rgba, width, height) {
    const t0 = performance.now();
    this.image = rawImageFromRgba(this.lib, rgba, width, height);
    this.width = width;
    this.height = height;
    this.inputs = await this.processor(this.image);
    this.embeddings = await this.model.get_image_embeddings(this.inputs);
    this.msEncode = performance.now() - t0;
    return this.msEncode;
  }

  /**
   * Mask for one prompt: points [[x, y], ...] in view pixels (labels 1 = foreground).
   * @returns {{mask: Uint8Array (width*height, 1 = inside), score: number, area: number}}
   */
  async maskAt(points, labels = null) {
    if (!this.embeddings) throw new Error("setImage() primero");
    const pts = [[points.map(([x, y]) => [x, y])]]; // batch × point_batch × points × 2
    const lbl = [[labels || points.map(() => 1)]];
    const reshaped = this.processor.reshape_input_points(pts, this.inputs.original_sizes, this.inputs.reshaped_input_sizes);
    const input_labels = new this.lib.Tensor("int64", BigInt64Array.from(lbl.flat(2).map((v) => BigInt(v))), [1, 1, lbl[0][0].length]);
    const outputs = await this.model({ ...this.embeddings, input_points: reshaped, input_labels });
    const masks = await this.processor.post_process_masks(outputs.pred_masks, this.inputs.original_sizes, this.inputs.reshaped_input_sizes);
    const scores = outputs.iou_scores.data; // [1, 1, 3]
    let best = 0;
    for (let k = 1; k < scores.length; k++) if (scores[k] > scores[best]) best = k;
    // masks[0]: Tensor [point_batch, 3, H, W] (bool/uint8)
    const t = masks[0];
    const [, nMasks, H, W] = t.dims;
    const src = t.data;
    const off = best * H * W;
    const out = new Uint8Array(this.width * this.height);
    let area = 0;
    for (let y = 0; y < this.height; y++) {
      const sy = Math.min(H - 1, Math.floor((y * H) / this.height));
      for (let x = 0; x < this.width; x++) {
        const sx = Math.min(W - 1, Math.floor((x * W) / this.width));
        const v = src[off + sy * W + sx] ? 1 : 0;
        out[y * this.width + x] = v;
        area += v;
      }
    }
    return { mask: out, score: Number(scores[best]), area, candidates: nMasks };
  }

  /**
   * One u32 label map from several point prompts: near-duplicate masks (two
   * prompts on the same object) are merged, then masks are painted largest
   * first so smaller objects stay on top; low-score or empty masks are skipped.
   * @param {Array<{points:number[][], id?:number}>} prompts
   * @returns {{labels:Uint32Array, labelCount:number, duplicates:number, objects:Array<{id:number, score:number, area:number, prompt:number, prompts:number[]}>}}
   */
  async labelMask(prompts, { minScore = 0.5, minArea = 64, maxAreaFraction = 0.9, duplicateIou = 0.8 } = {}) {
    const results = [];
    for (let i = 0; i < prompts.length; i++) {
      const r = await this.maskAt(prompts[i].points, prompts[i].labels || null);
      const frac = r.area / (this.width * this.height);
      if (r.score < minScore || r.area < minArea || frac > maxAreaFraction) continue;
      results.push({ ...r, prompt: i });
    }
    const merged = mergeDuplicateMasks(results, duplicateIou);
    return { ...paintMasks(merged, this.width, this.height), duplicates: results.length - merged.length };
  }
}

/** Intersection over union of two binary masks (Uint8Array of the same length). */
export function maskIou(a, b) {
  let inter = 0;
  let union = 0;
  for (let p = 0; p < a.length; p++) {
    const x = a[p] !== 0;
    const y = b[p] !== 0;
    if (x && y) inter++;
    if (x || y) union++;
  }
  return union ? inter / union : 0;
}

/**
 * Drop masks that are near-duplicates (IoU ≥ threshold) of a better-scored one,
 * keeping the survivor's `prompts` list. Input order does not matter.
 * @param {Array<{mask:Uint8Array, score:number, area:number, prompt:number}>} masks
 */
export function mergeDuplicateMasks(masks, iouThreshold = 0.8) {
  const byScore = [...masks].sort((a, b) => b.score - a.score || b.area - a.area);
  const kept = [];
  for (const m of byScore) {
    const dup = kept.find((k) => maskIou(k.mask, m.mask) >= iouThreshold);
    if (dup) dup.prompts.push(m.prompt);
    else kept.push({ ...m, prompts: [m.prompt] });
  }
  return kept;
}

/** Paint masks largest first into one u32 label map (1..n); smaller objects stay on top. */
export function paintMasks(masks, width, height) {
  const ordered = [...masks].sort((a, b) => b.area - a.area);
  const labels = new Uint32Array(width * height);
  const objects = [];
  ordered.forEach((r, k) => {
    const id = k + 1;
    for (let p = 0; p < labels.length; p++) if (r.mask[p]) labels[p] = id;
    objects.push({ id, score: r.score, area: r.area, prompt: r.prompt, prompts: r.prompts || [r.prompt] });
  });
  return { labels, labelCount: objects.length + 1, objects };
}

/** CLIP image/text embeddings (unit-normalised Float32Array of 512). */
export class BrowserClip {
  constructor(lib, vision, text, processor, tokenizer, meta) {
    this.lib = lib;
    this.vision = vision;
    this.text = text;
    this.processor = processor;
    this.tokenizer = tokenizer;
    this.meta = meta;
  }

  static async load({ model = CLIP_MODEL, device = "auto", dtype = "q8", progress = null } = {}) {
    const lib = await loadTransformers();
    const dev = await pickDevice(device);
    const opts = { device: dev, dtype, progress_callback: progress || undefined };
    const load = async (d) => ({
      vision: await lib.CLIPVisionModelWithProjection.from_pretrained(model, { ...opts, device: d }),
      text: await lib.CLIPTextModelWithProjection.from_pretrained(model, { ...opts, device: d }),
    });
    let models;
    let used = dev;
    try {
      models = await load(dev);
    } catch (err) {
      if (dev !== "wasm") {
        console.warn(`[ml] CLIP no cargó con ${dev} (${err.message}); reintentando con WASM`);
        models = await load("wasm");
        used = "wasm";
      } else throw err;
    }
    const processor = await lib.AutoProcessor.from_pretrained(model);
    const tokenizer = await lib.AutoTokenizer.from_pretrained(model);
    return new BrowserClip(lib, models.vision, models.text, processor, tokenizer, { id: model, device: used, dtype });
  }

  static normalize(v) {
    let n = 0;
    for (let i = 0; i < v.length; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
    return out;
  }

  async embedImage(rgba, width, height) {
    const img = rawImageFromRgba(this.lib, rgba, width, height);
    const inputs = await this.processor(img);
    const out = await this.vision(inputs);
    return BrowserClip.normalize(out.image_embeds.data);
  }

  async embedText(text) {
    const inputs = this.tokenizer([text], { padding: true, truncation: true });
    const out = await this.text(inputs);
    return BrowserClip.normalize(out.text_embeds.data);
  }

  static cosine(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
}
