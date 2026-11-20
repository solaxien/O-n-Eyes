import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dpmStep, half, noise } from "../src/numerics.js";
import { colorEvidence, cvStressStudy, differenceMap, enhance } from "../src/vision.js";
import { chooseProfile, fallbackChain, nextFallback, previewSteps } from "../src/resources.js";
import { analyzePrompt, conditioningPrompt } from "../src/prompt.js";
const fixture = JSON.parse(
  await readFile(new URL("./scheduler-fixtures.json", import.meta.url), "utf8"),
);
for (const data of fixture.cases)
  test(`DPM++ ${data.steps} steps match independent Diffusers FP32 reference`, () => {
    let sample = new Float32Array(data.initial),
      previous = null;
    for (let i = 0; i < data.rows.length; i++) {
      ({ sample, previous } = dpmStep(
        sample,
        new Float32Array(data.rows[i].epsilon),
        data.sigmas,
        i,
        previous,
      ));
      const max = Math.max(
        ...sample.map((x, j) => Math.abs(x - data.rows[i].sample[j])),
      );
      assert.ok(max < 0.00005, `step ${i}: ${max}`);
    }
  });
test("FP16 conversion handles boundary values and ties", () => {
  for (const [x, expected] of [
    [0, 0],
    [-0, 0x8000],
    [1, 0x3c00],
    [1.5, 0x3e00],
    [-2, 0xc000],
    [65504, 0x7bff],
    [Infinity, 0x7c00],
    [2 ** -24, 1],
    [2 ** -25, 0],
    [1 + 2 ** -11, 0x3c00],
  ])
    assert.equal(half(x), expected);
  assert.ok((half(NaN) & 0x7fff) > 0x7c00);
});
test("Fresh seeds alter finite initial noise reproducibly", () => {
  assert.deepEqual(noise(38, 64), noise(38, 64));
  assert.notDeepEqual(noise(38, 64), noise(39, 64));
  assert.ok(noise(0, 4096).every(Number.isFinite));
});
function image(size, hot = false) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      out.fill(50 + x * 2, i, i + 3);
      out[i + 3] = 255;
    }
  if (hot) out.fill(255, (20 * size + 20) * 4, (20 * size + 20) * 4 + 3);
  return out;
}
test("An injected isolated hot pixel is corrected while a gradient remains usable", () => {
  const clean = image(64),
    raw = image(64, true),
    result = enhance(raw, 64, 64),
    i = (20 * 64 + 20) * 4;
  assert.ok(result.usable);
  assert.ok(result.applied.includes("Isolated speckle correction"));
  assert.equal(result.quality.after.kind, "no-reference-pixel-health-v1");
  assert.ok(result.quality.after.score >= 0 && result.quality.after.score <= 100);
  assert.ok(
    Math.abs(result.pixels[i] - clean[i]) < Math.abs(raw[i] - clean[i]) * 0.1,
  );
});
test("Uniform failed decode is flagged and retains original pixels", () => {
  const raw = new Uint8ClampedArray(32 * 32 * 4).fill(255),
    result = enhance(raw, 32, 32);
  assert.equal(result.usable, false);
  assert.deepEqual(result.pixels, raw);
});
test("Edge-aware detail recovery visibly changes a textured decode within the quality gate", () => {
  const size = 64,
    raw = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4,
        texture = ((x * 17 + y * 31) % 27) - 13,
        value = 128 + texture + (x % 8 < 4 ? 18 : -18);
      raw[i] = raw[i + 1] = raw[i + 2] = value;
      raw[i + 3] = 255;
    }
  const result = enhance(raw, size, size),
    difference = differenceMap(raw, result.pixels);
  assert.ok(result.usable);
  assert.ok(result.applied.includes("Edge-aware bilateral detail recovery"));
  assert.ok(difference.changed_fraction > 0.1);
  assert.ok(difference.mean_absolute_delta > 0.5);
});
test("Auto lowers work when measured time or memory class cannot fit its budget", () => {
  assert.equal(chooseProfile("auto", { memoryGiB: 8, hardwareConcurrency: 8, stepMs: 1100 }).steps, 6);
  assert.equal(chooseProfile("auto", { memoryGiB: 8, hardwareConcurrency: 8, stepMs: 1600 }).steps, 4);
  assert.equal(chooseProfile("auto", { memoryGiB: 4 }).steps, 4);
  assert.equal(chooseProfile("quality", { memoryGiB: 4 }).size, 256);
  assert.equal(
    chooseProfile("balanced", { memoryGiB: 8, hardwareConcurrency: 8, stepMs: 1600 }).steps,
    12,
  );
});
test("Complex prompts receive a larger auto budget only on capable devices", () => {
  assert.equal(
    chooseProfile("auto", {
      memoryGiB: 8,
      hardwareConcurrency: 8,
      complexity: 0.8,
    }).key,
    "balanced",
  );
  assert.equal(
    chooseProfile("auto", {
      memoryGiB: 4,
      hardwareConcurrency: 4,
      complexity: 0.8,
    }).key,
    "preview",
  );
});
test("Fallback chain always reduces local work and terminates", () => {
  assert.deepEqual(fallbackChain("quality"), ["balanced", "eco", "preview"]);
  assert.equal(nextFallback("preview"), null);
});
test("Process visualizer captures bounded real diffusion checkpoints", () => {
  assert.deepEqual(previewSteps(4), [1, 2, 3]);
  assert.deepEqual(previewSteps(6), [2, 3, 5]);
  assert.deepEqual(previewSteps(12), [3, 6, 9]);
  assert.deepEqual(previewSteps(20), [5, 10, 15]);
});
test("Prompt contract captures object attributes and a spatial relation", () => {
  const plan = analyzePrompt("Two red apples are to the left of a blue cup on a table.");
  assert.equal(plan.objects[0].label, "apple");
  assert.equal(plan.objects[0].count, 2);
  assert.equal(plan.objects[0].color, "red");
  assert.deepEqual(plan.relations, [
    { subject: "apple", relation: "left of", object: "cup" },
  ]);
  assert.match(conditioningPrompt(plan), /Layout: apple left of cup/);
});
test("Colour evidence is a cheap global diagnostic, not an object verifier", () => {
  const pixels = new Uint8ClampedArray(16 * 16 * 4).fill(255);
  for (let i = 0; i < pixels.length; i += 4) {
    pixels[i] = 230;
    pixels[i + 1] = 20;
    pixels[i + 2] = 15;
  }
  const evidence = colorEvidence(
    pixels,
    16,
    16,
    analyzePrompt("A red apple beside a blue cup."),
    1,
  );
  assert.equal(evidence.requested.find((item) => item.color === "red").detected, true);
  assert.equal(evidence.requested.find((item) => item.color === "blue").detected, false);
  assert.match(evidence.caveat, /does not verify object identity/);
});
test("Difference maps expose real CV deltas without changing the source images", () => {
  const reference = new Uint8ClampedArray(4 * 4 * 4).fill(255),
    candidate = new Uint8ClampedArray(reference);
  candidate[0] = 0;
  const difference = differenceMap(reference, candidate);
  assert.equal(difference.changed_pixels, 1);
  assert.equal(difference.changed_fraction, 1 / 16);
  assert.ok(difference.max_absolute_delta > 0);
  assert.ok(difference.pixels[0] > 0);
});
test("Controlled CV stress study makes conditional stages visibly testable", () => {
  const study = cvStressStudy(image(64), 64, 64);
  assert.ok(study.speckle.injected_pixels >= 8);
  assert.notDeepEqual(study.speckle.input, study.speckle.output);
  assert.notDeepEqual(study.color.input, study.color.output);
  assert.equal(study.quality_gate.usable, false);
});
