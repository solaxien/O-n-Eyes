import * as ort from "./vendor/ort.webgpu.min.mjs?v=1.31.1";
import { getAsset } from "./cache.js";
import { Tokenizer } from "./tokenizer.js";
import { toHalf, noise, dpmStep } from "./numerics.js";
import { enhance, png } from "./vision.js";
import { chooseProfile } from "./resources.js";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.logLevel = "warning";
// Even runtime fetches may read only the prepared application cache. This worker
// cannot send prompts, inference requests, or cache misses to a server.
globalThis.fetch = async (input) => {
  const url = new URL(
    typeof input === "string" || input instanceof URL ? input : input.url,
    location.href,
  );
  if (url.origin !== location.origin)
    throw Error("External network access is disabled during local generation.");
  const result = await caches.match(url.pathname);
  if (!result)
    throw Error(
      "A runtime asset is missing from the offline cache. Prepare the app again.",
    );
  return result;
};
const stage = (message, progress) =>
  postMessage({ type: "progress", message, progress });
const resident = new Map();
let retain = false,
  negativeEmbedding = null,
  dispatches = 0,
  observedStepMs;
if (typeof GPUComputePassEncoder !== "undefined") {
  const original = GPUComputePassEncoder.prototype.dispatchWorkgroups;
  GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
    dispatches++;
    return original.apply(this, args);
  };
}
async function session(proof, backend = "webgpu") {
  if (retain && resident.has(proof.graph)) return resident.get(proof.graph);
  const graphProof = proof.files.find((f) => f.path === proof.graph),
    graph = new Uint8Array(
      await (
        await getAsset(proof.graph, { sha256: graphProof.sha256 })
      ).arrayBuffer(),
    ),
    externalData = [];
  for (const file of proof.files.filter((f) => f.path !== proof.graph))
    externalData.push({
      path: file.path,
      data: await (await getAsset(file.path, { sha256: file.sha256 })).blob(),
    });
  const model = await ort.InferenceSession.create(graph, {
    executionProviders:
      backend === "webgpu"
        ? [{ name: "webgpu", storageBufferCacheMode: "simple" }]
        : ["wasm"],
    externalData,
    graphOptimizationLevel: "all",
    enableCpuMemArena: false,
    enableMemPattern: false,
  });
  if (retain) resident.set(proof.graph, model);
  return model;
}
async function run(model, feeds) {
  try {
    const output = (await model.run(feeds)).output;
    const result = new Float32Array(output.data);
    output.dispose();
    return result;
  } finally {
    for (const tensor of Object.values(feeds)) tensor.dispose();
  }
}
self.onmessage = async ({ data }) => {
  const started = performance.now(),
    times = {},
    loads = {};
  let active;
  retain = Boolean(data.keepLoaded);
  dispatches = 0;
  const warm = resident.size === 3;
  async function unload() {
    if (active && !retain) await active.release();
    active = null;
  }
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter || !adapter.features.has("shader-f16"))
      throw Error(
        "A WebGPU adapter with FP16 support is required. No server fallback is used.",
      );
    const profile = chooseProfile(data.mode, {
        memoryGiB: navigator.deviceMemory || 4,
        stepMs: observedStepMs,
        keepLoaded: retain,
      }),
      reason = profile.reason;
    const manifest = await (await getAsset("manifest.json")).json();
    const tokenizer = new Tokenizer(
      await (await getAsset("vocab.json")).json(),
      await (await getAsset("merges.txt")).text(),
    );
    const ids = tokenizer.encode(data.prompt);
    const plans = await (await getAsset("schedules.json")).json(),
      schedule = plans[profile.steps];
    stage("Loading the local CPU text encoder", 4);
    let tick = performance.now();
    active = await session(manifest.components.text, "wasm");
    loads.text = performance.now() - tick;
    tick = performance.now();
    const negative =
      negativeEmbedding ||
      (await run(active, {
        input_ids: new ort.Tensor("int32", tokenizer.encode(""), [1, 77]),
      }));
    if (retain) negativeEmbedding = negative;
    const positive = await run(active, {
      input_ids: new ort.Tensor("int32", ids, [1, 77]),
    });
    times.encode = performance.now() - tick;
    await unload();
    stage("Loading the adapted denoiser on this device", 12);
    tick = performance.now();
    active = await session(manifest.components.unet);
    loads.unet = performance.now() - tick;
    const side = profile.size / 8,
      shape = [1, 4, side, side];
    let latent = noise(data.seed, 4 * side * side),
      previous = null;
    const contexts = [toHalf(negative), toHalf(positive)];
    tick = performance.now();
    const steps = [];
    for (let step = 0; step < profile.steps; step++) {
      stage(
        `Local diffusion · step ${step + 1} of ${profile.steps}`,
        15 + (step / profile.steps) * 65,
      );
      const stepStart = performance.now(),
        sample = toHalf(latent),
        values = [];
      for (const context of contexts)
        values.push(
          await run(active, {
            sample: new ort.Tensor("float16", sample, shape),
            timestep: new ort.Tensor(
              "float32",
              new Float32Array([schedule.timesteps[step]]),
              [1],
            ),
            encoder_hidden_states: new ort.Tensor(
              "float16",
              context,
              [1, 77, 768],
            ),
          }),
        );
      const prediction = new Float32Array(latent.length);
      for (let i = 0; i < prediction.length; i++)
        prediction[i] =
          values[0][i] + manifest.guidance * (values[1][i] - values[0][i]);
      ({ sample: latent, previous } = dpmStep(
        latent,
        prediction,
        schedule.sigmas,
        step,
        previous,
      ));
      if (!latent.every(Number.isFinite))
        throw Error(
          "The local denoiser produced non-finite values. Try Low memory mode.",
        );
      steps.push(performance.now() - stepStart);
    }
    times.denoise = performance.now() - tick;
    observedStepMs =
      steps.slice(1).reduce((s, x) => s + x, 0) / Math.max(1, steps.length - 1);
    await unload();
    stage("Decoding the image locally", 83);
    tick = performance.now();
    active = await session(manifest.components.decoder);
    loads.decoder = performance.now() - tick;
    tick = performance.now();
    const decoded = await run(active, {
      latent: new ort.Tensor("float16", toHalf(latent), shape),
    });
    times.decode = performance.now() - tick;
    await unload();
    if (!decoded.every(Number.isFinite))
      throw Error(
        "The local decoder produced invalid pixels. Try Low memory mode.",
      );
    const pixels = new Uint8ClampedArray(profile.size * profile.size * 4),
      plane = profile.size * profile.size;
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++)
        pixels[i * 4 + c] = Math.max(
          0,
          Math.min(255, (decoded[c * plane + i] + 1) * 127.5),
        );
      pixels[i * 4 + 3] = 255;
    }
    stage("Local colour, edge and artifact checks", 94);
    tick = performance.now();
    const cv = enhance(pixels, profile.size, profile.size, data.omit || []);
    const raw = await png(pixels, profile.size, profile.size),
      enhanced = await png(cv.pixels, profile.size, profile.size);
    times.cv_resize_encode = performance.now() - tick;
    const metadata = {
      execution: "visitor browser",
      inferenceRequests: 0,
      worker_network_fetch: "disabled; runtime reads CacheStorage only",
      backend: "Local WASM text encoding; WebGPU diffusion and decoding",
      gpu: {
        vendor: adapter.info.vendor,
        architecture: adapter.info.architecture,
      },
      gpu_dispatches: dispatches,
      seed: data.seed,
      prompt: data.prompt,
      native_resolution: profile.size,
      display_resolution: 512,
      steps: profile.steps,
      unet_calls: profile.steps * 2,
      guidance: manifest.guidance,
      profile_reason: reason,
      adapter_sha256: manifest.adapter_sha256,
      model_revision: manifest.model.revision,
      model_bytes: Object.values(manifest.components)
        .flatMap((x) => x.files)
        .reduce((s, f) => s + f.bytes, 0),
      components_load_ms: loads,
      stages_ms: times,
      step_ms: steps,
      cached_generation_ms: performance.now() - started,
      initial_download_included: false,
      models_retained: retain,
      warm_models: warm,
      cv: {
        usable: cv.usable,
        applied: cv.applied,
        before: cv.before,
        after: cv.after,
        artifact_fraction: cv.artifact_fraction,
      },
      memory_note:
        "Browser deviceMemory is a coarse device class, not available GPU memory. Process memory is measured separately through ADB; this page cannot measure total GPU allocations.",
    };
    const variants = {};
    if (data.cvStudy) {
      metadata.cv_study = {};
      for (const component of ["repair", "edges", "color", "gate"]) {
        const begin = performance.now(),
          variant = enhance(pixels, profile.size, profile.size, [component]);
        variants["omit-" + component] = await png(
          variant.pixels,
          profile.size,
          profile.size,
        );
        metadata.cv_study["omit-" + component] = {
          ms: performance.now() - begin,
          applied: variant.applied,
          before: variant.before,
          after: variant.after,
          usable: variant.usable,
        };
      }
    }
    postMessage({ type: "result", raw, enhanced, metadata, variants });
  } catch (error) {
    postMessage({
      type: "error",
      message: error.message || String(error),
      elapsed_ms: performance.now() - started,
    });
  } finally {
    if (active) await active.release().catch(() => {});
  }
};
