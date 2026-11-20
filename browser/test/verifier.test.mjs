import test from "node:test";
import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import { analyzePrompt } from "../src/prompt.js";
import {
  COCO_CLASSES,
  boxColor,
  decode,
  detectorClass,
  preprocess,
  relationHolds,
  verify,
} from "../src/verifier.js";
import {
  candidateScore,
  candidateSeed,
  pickBest,
  selectionPolicy,
  shouldContinue,
} from "../src/selection.js";

function solid(size, rgb) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < out.length; i += 4) out.set([...rgb, 255], i);
  return out;
}
function paint(pixels, size, [x1, y1, x2, y2], rgb) {
  for (let y = y1; y < y2; y++)
    for (let x = x1; x < x2; x++) pixels.set(rgb, (y * size + x) * 4);
}
const det = (label, score, box) => ({ label, score, box });

test("Prompt labels map to detector classes or are marked unverifiable", () => {
  assert.equal(COCO_CLASSES.length, 80);
  assert.equal(detectorClass("cup"), "cup");
  assert.equal(detectorClass("woman"), "person");
  assert.equal(detectorClass("television"), "tv");
  assert.equal(detectorClass("table"), "dining table");
  assert.equal(detectorClass("mountain"), null);
});

test("Letterbox preprocessing is BGR, CHW and padded with 114", () => {
  const size = 64,
    pixels = solid(size, [200, 100, 50]),
    { data, ratio } = preprocess(pixels, size, size, 128);
  assert.equal(ratio, 2);
  const plane = 128 * 128;
  assert.equal(data[0], 50); // B plane
  assert.equal(data[plane], 100); // G plane
  assert.equal(data[2 * plane], 200); // R plane
  const wide = preprocess(solid(64, [10, 10, 10]).subarray(0, 64 * 32 * 4), 64, 32, 128);
  assert.equal(wide.data[127 * 128], 114); // bottom rows are padding
});

test("YOLOX decoding recovers a box and applies per-class NMS", () => {
  const size = 64,
    rows = (64 / 8) ** 2 + (64 / 16) ** 2 + (64 / 32) ** 2,
    output = new Float32Array(rows * 85),
    cup = COCO_CLASSES.indexOf("cup");
  // Stride-8 cells (2, 3) and (3, 3): centres x = 20 and 22, both 16 px wide,
  // so they overlap heavily (IoU 0.78) and the weaker one must be removed.
  for (const [row, offset, objectness] of [[3 * 8 + 2, 0.5, 0.9], [3 * 8 + 3, -0.25, 0.6]]) {
    const o = row * 85;
    output[o] = offset;
    output[o + 1] = 0.5;
    output[o + 2] = Math.log(2); // width 16
    output[o + 3] = Math.log(2);
    output[o + 4] = objectness;
    output[o + 5 + cup] = 1;
  }
  const found = decode(output, { size, ratio: 1, width: 64, height: 64 });
  assert.equal(found.length, 1, "overlapping same-class boxes are merged");
  assert.equal(found[0].label, "cup");
  assert.deepEqual(found[0].box.map(Math.round), [12, 20, 28, 36]);
});

test("Box colour checks the object, not the whole image", () => {
  const size = 100,
    pixels = solid(size, [30, 60, 220]); // mostly blue background
  paint(pixels, size, [10, 10, 50, 50], [220, 30, 30]); // red object
  assert.equal(boxColor(pixels, size, size, [10, 10, 50, 50], "red").pass, true);
  assert.equal(boxColor(pixels, size, size, [10, 10, 50, 50], "blue").pass, false);
  assert.equal(boxColor(pixels, size, size, [60, 60, 95, 95], "blue").pass, true);
});

test("Relations use box centres with a margin; depth is not verifiable", () => {
  const a = { box: [0, 40, 20, 60] },
    b = { box: [70, 40, 90, 60] };
  assert.equal(relationHolds("left of", a, b, 100, 100), true);
  assert.equal(relationHolds("right of", a, b, 100, 100), false);
  assert.equal(relationHolds("beside", a, b, 100, 100), true);
  assert.equal(relationHolds("behind", a, b, 100, 100), null);
});

test("Verifier reports pass, fail and not-verifiable clauses honestly", () => {
  const size = 100,
    pixels = solid(size, [240, 240, 240]);
  paint(pixels, size, [5, 30, 35, 70], [220, 30, 30]); // red apple, left
  paint(pixels, size, [60, 30, 95, 70], [220, 40, 40]); // cup is red, not blue
  const analysis = analyzePrompt("a red apple to the left of a blue cup near a mountain"),
    report = verify(analysis, [det("apple", 0.9, [5, 30, 35, 70]), det("cup", 0.8, [60, 30, 95, 70])], pixels, size, size),
    status = (list, label) => list.find((c) => c.label === label)?.status;
  assert.equal(status(report.object_checks, "red apple"), "pass");
  assert.equal(status(report.object_checks, "blue cup"), "pass");
  assert.equal(status(report.object_checks, "mountain"), "not_verifiable");
  assert.equal(status(report.color_checks, "red apple"), "pass");
  assert.equal(status(report.color_checks, "blue cup"), "fail");
  assert.equal(status(report.relation_checks, "apple left of cup"), "pass");
  assert.equal(report.summary.failed, 1);
  assert.equal(report.summary.critical_failed, true);
  assert.equal(report.alignment, null);
});

test("Exact counts follow the count threshold", () => {
  const size = 64,
    pixels = solid(size, [200, 200, 0]),
    analysis = analyzePrompt("two bananas on a table"),
    three = [0.9, 0.8, 0.5].map((s, i) => det("banana", s, [i * 20, 0, i * 20 + 15, 15])),
    report = verify(analysis, three, pixels, size, size);
  assert.equal(report.count_checks[0].found, 3);
  assert.equal(report.count_checks[0].status, "fail");
  const two = verify(analysis, three.slice(0, 2), pixels, size, size);
  assert.equal(two.count_checks[0].status, "pass");
});

test("Selection policy keeps fast modes single-image and the reference unselected", () => {
  assert.equal(selectionPolicy("eco").maxCandidates, 1);
  assert.equal(selectionPolicy("eco").verify, false);
  assert.equal(selectionPolicy("baseline").maxCandidates, 1);
  assert.equal(selectionPolicy("balanced").maxCandidates, 2);
  assert.equal(selectionPolicy("quality").maxCandidates, 3);
  assert.equal(selectionPolicy("quality", { verifyRequested: false }).maxCandidates, 1);
  assert.equal(candidateSeed(38, 0), 38);
  assert.notEqual(candidateSeed(38, 1), 38);
  assert.equal(candidateSeed(38, 1), candidateSeed(38, 1));
  assert.ok(candidateSeed(4294967295, 2) <= 4294967295);
});

const candidate = (passed, failed, health = 80) => ({
  verifier: { summary: { checked: passed + failed, passed, failed, not_verifiable: 0, critical_failed: failed > 0, score: passed / (passed + failed) } },
  cv: { usable: true, quality: { after: { score: health } } },
});

test("Balanced retries once only after a failed clause", () => {
  const policy = selectionPolicy("balanced");
  assert.equal(shouldContinue(policy, [candidate(3, 0)], 8000).next, false);
  assert.equal(shouldContinue(policy, [candidate(1, 2)], 8000).next, true);
  assert.equal(shouldContinue(policy, [candidate(1, 2), candidate(1, 2)], 16000).next, false);
  assert.equal(shouldContinue(policy, [candidate(1, 2)], 30000).next, false, "budget");
  assert.equal(
    shouldContinue(policy, [{ verifier: { unavailable: true }, cv: {} }], 8000).next,
    false,
    "no claimed retry without a verifier",
  );
});

test("Quality keeps the best candidate within budget", () => {
  const policy = selectionPolicy("quality"),
    list = [candidate(1, 2, 90), candidate(2, 1, 60)];
  assert.equal(shouldContinue(policy, list, 60000).next, true);
  assert.equal(shouldContinue(policy, [...list, candidate(0, 3)], 90000).next, false);
  assert.equal(pickBest([...list, candidate(0, 3)]), 1);
  assert.ok(candidateScore(candidate(3, 0)) > candidateScore(candidate(2, 1)));
});

const model = process.env.P3_VERIFIER_MODEL || new URL("../../.models/verifier/verifier-yolox-nano.onnx", import.meta.url).pathname;
const hasModel = await access(model).then(() => true, () => false);
test(
  "JS detector path matches the Python ONNX reference on real phone outputs",
  { skip: !hasModel && "Set P3_VERIFIER_MODEL to the YOLOX-Nano ONNX file to run." },
  async () => {
    const ort = await import("onnxruntime-web");
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(new Uint8Array(await readFile(model)), {
        executionProviders: ["wasm"],
      }),
      reference = JSON.parse(await readFile(new URL("./fixtures/yolox-nano-reference.json", import.meta.url), "utf8"));
    for (const [name, expected] of Object.entries(reference)) {
      const pixels = new Uint8ClampedArray(await readFile(new URL(`./fixtures/${name}-256.rgba`, import.meta.url))),
        input = preprocess(pixels, 256, 256),
        result = await session.run({ images: new ort.Tensor("float32", input.data, [1, 3, 416, 416]) }),
        found = decode(result.output.data, { size: 416, ratio: input.ratio, width: 256, height: 256 });
      assert.deepEqual(found.map((d) => d.label), expected.map((d) => d.label), name);
      for (let i = 0; i < found.length; i++) {
        assert.ok(Math.abs(found[i].score - expected[i].score) < 0.02, `${name} score`);
        for (let k = 0; k < 4; k++)
          assert.ok(Math.abs(found[i].box[k] - expected[i].box[k]) < 2, `${name} box`);
      }
    }
  },
);

import { artifactReport, enhance } from "../src/vision.js";
function textured(size, base, cast = [0, 0, 0]) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4,
        v = base + Math.round((x + y) / 4) + ((x * 3 + y) % 3) + (x > size / 2 ? 25 : 0);
      out[i] = v + cast[0];
      out[i + 1] = v + cast[1];
      out[i + 2] = v + cast[2];
      out[i + 3] = 255;
    }
  return out;
}

test("Exposure stage brightens a dark decode with one gamma for all channels", () => {
  const raw = textured(64, 20),
    result = enhance(raw, 64, 64),
    exposure = result.stages.find((s) => s.stage === "exposure");
  assert.equal(exposure.status, "applied");
  assert.ok(exposure.changed_pixels > 0);
  const mean = (p) => p.reduce((s, v, i) => (i % 4 === 3 ? s : s + v), 0) / (p.length * 0.75);
  assert.ok(mean(result.pixels) > mean(raw) + 15);
  assert.ok(result.artifacts.checks.find((c) => c.name.startsWith("exposure")).flagged);
});

test("Well-exposed images record why each stage was skipped", () => {
  const result = enhance(textured(64, 110), 64, 64),
    byName = Object.fromEntries(result.stages.map((s) => [s.stage, s]));
  assert.equal(byName.exposure.status, "skipped");
  assert.match(byName.exposure.reason, /within 70 to 185/);
  assert.equal(byName.repair.status, "skipped");
  assert.ok(["applied", "skipped"].includes(byName.gate.status));
  for (const stage of result.stages) assert.ok(stage.reason.length > 10, stage.stage);
});

test("Artifact report flags a colour cast and an 8-pixel decoder grid", () => {
  const size = 64,
    cast = textured(size, 100, [18, 0, -10]);
  assert.ok(artifactReport(cast, size, size).checks.find((c) => c.name.startsWith("colour cast")).flagged);
  const grid = textured(size, 100);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x += 8) grid.fill(30, (y * size + x) * 4, (y * size + x) * 4 + 3);
  assert.ok(artifactReport(grid, size, size).checks.find((c) => c.name.startsWith("decoder grid")).flagged);
  assert.equal(artifactReport(textured(size, 100), size, size).checks.find((c) => c.name.startsWith("decoder grid")).flagged, false);
});

import { chooseProfile, deviceBudget, readinessItems } from "../src/resources.js";
test("Simulated budgets drive the real planner and stay labelled", () => {
  const actual = deviceBudget("actual", { memoryGiB: 8, hardwareConcurrency: 8 }),
    phone = deviceBudget("phone-4gb", { memoryGiB: 8, hardwareConcurrency: 8 }),
    failing = deviceBudget("force-fallback", { memoryGiB: 8, hardwareConcurrency: 8 });
  assert.equal(actual.simulated, false);
  assert.equal(phone.simulated, true);
  assert.equal(failing.failFirstAttempt, true);
  assert.equal(chooseProfile("auto", { ...actual, complexity: 0.9 }).key, "balanced");
  assert.equal(chooseProfile("auto", { ...phone, complexity: 0.9 }).key, "preview");
  assert.equal(chooseProfile("quality", phone).key, "eco");
});

test("Readiness report states each capability and the plan it implies", () => {
  const items = readinessItems({
      webgpu: true,
      fp16: false,
      quotaBytes: 500 * 2 ** 20,
      usageBytes: 0,
      requiredBytes: 896382552,
      budget: deviceBudget("phone-4gb"),
    }),
    get = (label) => items.find((i) => i.label === label);
  assert.equal(get("WebGPU").status, "pass");
  assert.equal(get("FP16 shaders").status, "fail");
  assert.equal(get("Storage").status, "fail");
  assert.match(get("Memory class and cores").detail, /simulated/);
  assert.match(get("Auto plan").detail, /Fast preview/);
  assert.equal(get("Prompt verifier").status, "not_verifiable");
});
