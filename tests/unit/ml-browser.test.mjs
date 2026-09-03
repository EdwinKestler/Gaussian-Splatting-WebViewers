/**
 * Unit tests for the pure helpers of gaussian_splatting_webgpu/ml-browser.js
 * (mask bookkeeping around SAM 2 in the browser). The models themselves are
 * covered by the opt-in e2e test (ML_E2E=1). Run: npm test.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import { maskIou, mergeDuplicateMasks, paintMasks, SAM_MODELS, CLIP_MODEL } from "../../gaussian_splatting_webgpu/ml-browser.js";

const m = (bits) => Uint8Array.from(bits);

describe("maskIou / mergeDuplicateMasks / paintMasks", () => {
  test("maskIou counts intersection over union", () => {
    assert.equal(maskIou(m([1, 1, 0, 0]), m([0, 1, 1, 0])), 1 / 3);
    assert.equal(maskIou(m([0, 0]), m([0, 0])), 0);
    assert.equal(maskIou(m([1, 1]), m([1, 1])), 1);
  });

  test("two prompts on the same object collapse into the better-scored mask", () => {
    const whole = { mask: m([1, 1, 1, 1, 0, 0]), score: 0.9, area: 4, prompt: 0 };
    const wholeAgain = { mask: m([1, 1, 1, 0, 0, 0]), score: 0.95, area: 3, prompt: 1 };
    const other = { mask: m([0, 0, 0, 0, 1, 1]), score: 0.7, area: 2, prompt: 2 };
    const kept = mergeDuplicateMasks([whole, wholeAgain, other], 0.7);
    assert.equal(kept.length, 2);
    assert.deepEqual(kept.map((k) => k.prompt), [1, 2], "el de mejor puntuación sobrevive");
    assert.deepEqual(kept[0].prompts, [1, 0]);
    assert.equal(mergeDuplicateMasks([whole, wholeAgain, other], 0.9).length, 3, "umbral alto: no se fusionan");
  });

  test("paintMasks paints largest first so small objects stay on top", () => {
    const big = { mask: m([1, 1, 1, 1]), score: 0.8, area: 4, prompt: 0 };
    const small = { mask: m([0, 1, 0, 0]), score: 0.9, area: 1, prompt: 1 };
    const out = paintMasks([small, big], 2, 2);
    assert.deepEqual(Array.from(out.labels), [1, 2, 1, 1]);
    assert.equal(out.labelCount, 3);
    assert.deepEqual(out.objects.map((o) => [o.id, o.prompt]), [[1, 0], [2, 1]]);
  });

  test("model registry", () => {
    assert.equal(SAM_MODELS["sam2-tiny"].id, "onnx-community/sam2.1-hiera-tiny-ONNX");
    assert.equal(CLIP_MODEL, "Xenova/clip-vit-base-patch32");
  });
});
