import * as ort from "./vendor/ort.webgpu.min.mjs?v=1.31.1";
import { getAsset } from "./cache.js";
import { Tokenizer } from "./tokenizer.js";
import { toHalf, noise, dpmStep } from "./numerics.js";
import { colorEvidence, cvStressStudy, differenceMap, enhance, png } from "./vision.js";
import { analyzePrompt, conditioningPrompt, QUALITY_NEGATIVE, QUALITY_SUFFIX } from "./prompt.js";
import { chooseProfile, previewSteps } from "./resources.js";
import { decode as decodeDetections, preprocess as detectorInput, verify } from "./verifier.js";

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
  negativeText = null,
  detectorSession = null,
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
function decodedPixels(decoded, size) {
  const pixels = new Uint8ClampedArray(size * size * 4),
    plane = size * size;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++)
      pixels[i * 4 + c] = Math.max(
        0,
        Math.min(255, (decoded[c * plane + i] + 1) * 127.5),
      );
    pixels[i * 4 + 3] = 255;
  }
  return pixels;
}
async function verifyLocally(semantic, pixels, size) {
  const begin = performance.now();
  try {
    const index = await (await fetch("/verifier-index.json")).json();
    if (!detectorSession) {
      let bytes;
      try {
        bytes = new Uint8Array(
          await (await getAsset(index.path, { sha256: index.sha256 })).arrayBuffer(),
        );
      } catch {
        return {
          unavailable: true,
          reason: "The verifier model is not in the offline cache. Use Prepare to download it.",
          ms: performance.now() - begin,
        };
      }
      detectorSession = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    }
    const input = detectorInput(pixels, size, size, index.input_size),
      feeds = {
        [detectorSession.inputNames[0]]: new ort.Tensor("float32", input.data, [1, 3, input.size, input.size]),
      },
      results = await detectorSession.run(feeds),
      output = results[detectorSession.outputNames[0]],
      detections = decodeDetections(output.data, {
        size: input.size,
        ratio: input.ratio,
        width: size,
        height: size,
      });
    output.dispose();
    for (const tensor of Object.values(feeds)) tensor.dispose();
    const report = verify(semantic, detections, pixels, size, size);
    if (!retain) {
      await detectorSession.release();
      detectorSession = null;
    }
    return { ...report, model: index.model, model_sha256: index.sha256, ms: performance.now() - begin };
  } catch (error) {
    if (detectorSession && !retain) {
      await detectorSession.release().catch(() => {});
      detectorSession = null;
    }
    return {
      unavailable: true,
      reason: "The on-device verifier failed: " + (error.message || String(error)),
      ms: performance.now() - begin,
    };
  }
}
self.onmessage = async ({ data }) => {
  const started = performance.now(),
    times = {},
    loads = {};
  let active,
    profile,
    semantic,
    conditioning = data.prompt,
    conditioningFallback = false;
  const visualizeProcess = Boolean(data.processPreview),
    checkpoints = [],
    processEvent = (stageName, status, detail) => {
      if (visualizeProcess)
        postMessage({ type: "process", stage: stageName, status, detail });
    };
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
    semantic = analyzePrompt(data.prompt);
    processEvent("prompt", "active", "Parsing the prompt and selecting a local resource profile.");
    profile = chooseProfile(data.mode, {
        memoryGiB: navigator.deviceMemory || 4,
        hardwareConcurrency: navigator.hardwareConcurrency || 4,
        stepMs: observedStepMs,
        complexity: semantic.complexity,
        keepLoaded: retain,
        budgetMs: data.budgetMs,
      });
    processEvent(
      "prompt",
      "done",
      `${profile.label}: ${profile.size} px and ${profile.steps} diffusion steps.`,
    );
    const manifest = await (await getAsset("manifest.json")).json();
    const tokenizer = new Tokenizer(
      await (await getAsset("vocab.json")).json(),
      await (await getAsset("merges.txt")).text(),
    );
    if (data.structuredPrompt && semantic.needs_semantic_verification)
      conditioning = conditioningPrompt(semantic);
    let ids,
      qualitySuffixApplied = false;
    if (data.qualityHints)
      try {
        ids = tokenizer.encode(conditioning + QUALITY_SUFFIX);
        conditioning += QUALITY_SUFFIX;
        qualitySuffixApplied = true;
      } catch {
        ids = undefined; // Too long with the suffix; keep the user's wording.
      }
    if (!ids)
      try {
        ids = tokenizer.encode(conditioning);
      } catch (error) {
        if (conditioning !== data.prompt) {
          conditioning = data.prompt;
          conditioningFallback = true;
          ids = tokenizer.encode(conditioning);
        } else throw error;
      }
    const unconditionalText = data.qualityHints ? QUALITY_NEGATIVE : "";
    const plans = await (await getAsset("schedules.json")).json(),
      schedule = plans[profile.steps];
    processEvent("text", "active", "Encoding the prompt locally with CLIP.");
    stage("Loading the local CPU text encoder", 4);
    let tick = performance.now();
    active = await session(manifest.components.text, "wasm");
    loads.text = performance.now() - tick;
    tick = performance.now();
    const negative =
      (negativeText === unconditionalText && negativeEmbedding) ||
      (await run(active, {
        input_ids: new ort.Tensor("int32", tokenizer.encode(unconditionalText), [1, 77]),
      }));
    if (retain) {
      negativeEmbedding = negative;
      negativeText = unconditionalText;
    }
    const positive = await run(active, {
      input_ids: new ort.Tensor("int32", ids, [1, 77]),
    });
    times.encode = performance.now() - tick;
    await unload();
    processEvent("text", "done", "Text embedding is ready; the encoder session was released.");
    processEvent("denoise", "active", `Starting ${profile.steps} local diffusion steps.`);
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
    const steps = [],
      checkpointSteps = visualizeProcess ? new Set(previewSteps(profile.steps)) : new Set();
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
      const completed = step + 1;
      if (checkpointSteps.has(completed)) {
        checkpoints.push({ step: completed, latent: latent.slice() });
        processEvent(
          "denoise",
          "active",
          `Completed step ${completed} of ${profile.steps}; captured a real latent checkpoint.`,
        );
      }
    }
    times.denoise = performance.now() - tick;
    observedStepMs =
      steps.slice(1).reduce((s, x) => s + x, 0) / Math.max(1, steps.length - 1);
    await unload();
    processEvent("denoise", "done", "Denoising completed; the large denoiser session was released.");
    processEvent("decode", "active", "Decoding the final latent locally with TAESD.");
    stage("Decoding the image locally", 83);
    tick = performance.now();
    active = await session(manifest.components.decoder);
    loads.decoder = performance.now() - tick;
    if (checkpoints.length) {
      const previewTick = performance.now();
      processEvent("preview", "active", "Decoding saved checkpoints after releasing the denoiser.");
      for (const checkpoint of checkpoints) {
        const previewDecoded = await run(active, {
            latent: new ort.Tensor("float16", toHalf(checkpoint.latent), shape),
          }),
          preview = await png(
            decodedPixels(previewDecoded, profile.size),
            profile.size,
            profile.size,
            192,
          );
        postMessage({
          type: "process-preview",
          step: checkpoint.step,
          steps: profile.steps,
          blob: preview,
        });
      }
      times.process_preview_decode = performance.now() - previewTick;
      processEvent("preview", "done", `${checkpoints.length} actual latent checkpoints decoded.`);
    }
    tick = performance.now();
    const decoded = await run(active, {
      latent: new ort.Tensor("float16", toHalf(latent), shape),
    });
    times.decode = performance.now() - tick;
    await unload();
    processEvent("decode", "done", "Final latent decoded to native pixels; decoder session was released.");
    if (!decoded.every(Number.isFinite))
      throw Error(
        "The local decoder produced invalid pixels. Try Low memory mode.",
      );
    const pixels = decodedPixels(decoded, profile.size);
    processEvent("cv", "active", "Running local artifact, colour, edge, and pixel-quality checks.");
    stage("Local colour, edge and artifact checks", 94);
    tick = performance.now();
    const color_evidence = colorEvidence(pixels, profile.size, profile.size, semantic),
      cv = enhance(pixels, profile.size, profile.size, data.omit || []),
      enhancementDifference = differenceMap(pixels, cv.pixels);
    const raw = await png(pixels, profile.size, profile.size),
      enhanced = await png(cv.pixels, profile.size, profile.size),
      difference = await png(
        enhancementDifference.pixels,
        profile.size,
        profile.size,
        profile.size,
      );
    times.cv_resize_encode = performance.now() - tick;
    processEvent("cv", "done", "Classical vision checks and output encoding completed.");
    let verifier = null;
    if (
      data.verify &&
      semantic.needs_semantic_verification &&
      (!data.verifyProfiles || data.verifyProfiles.includes(profile.key))
    ) {
      processEvent("verify", "active", "Checking requested objects, counts, colours and layout on this device.");
      stage("Checking the prompt on this device", 97);
      verifier = await verifyLocally(semantic, cv.pixels, profile.size);
      times.verify = verifier.ms;
      processEvent(
        "verify",
        "done",
        verifier.unavailable
          ? verifier.reason
          : `${verifier.summary.passed} passed, ${verifier.summary.failed} failed, ${verifier.summary.not_verifiable} not verifiable.`,
      );
    }
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
      conditioning_prompt: conditioning,
      conditioning_prompt_fallback: conditioningFallback,
      quality_hints: {
        enabled: Boolean(data.qualityHints),
        negative_prompt: unconditionalText || null,
        positive_suffix_applied: qualitySuffixApplied,
      },
      attempt: data.attempt || 1,
      fallback_from: data.fallbackFrom || null,
      native_resolution: profile.size,
      display_resolution: 512,
      steps: profile.steps,
      unet_calls: profile.steps * 2,
      guidance: manifest.guidance,
      profile_reason: profile.reason,
      profile: {
        key: profile.key,
        requested_mode: profile.requested_mode,
        fallback_chain: profile.fallback_chain,
        predicted_ms: profile.predicted_ms,
      },
      device_budget: {
        memory_class_gib: navigator.deviceMemory || null,
        logical_cores: navigator.hardwareConcurrency || null,
        target_ms: profile.targetMs,
      },
      semantic_plan: {
        complexity: semantic.complexity,
        requested_objects: semantic.objects,
        requested_relations: semantic.relations,
        color_evidence,
        verification: verifier && !verifier.unavailable
          ? "Checked on this device by the compact verifier. It steers retries only; final benchmark accuracy must come from a separate offline evaluator."
          : "No on-device verification ran for this image. Colour evidence is global only, so this plan is not proof of semantic correctness.",
      },
      verifier,
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
      process_visualizer: {
        enabled: visualizeProcess,
        checkpoints: checkpoints.map((checkpoint) => checkpoint.step),
        note: visualizeProcess
          ? "Real latent checkpoints were captured during diffusion and decoded only after the denoiser was released."
          : "Disabled for the normal low-overhead generation path.",
      },
      cv: {
        usable: cv.usable,
        applied: cv.applied,
        before: cv.before,
        after: cv.after,
        artifact_fraction: cv.artifact_fraction,
        quality: cv.quality,
        stages: cv.stages,
        artifact_checks: cv.artifacts,
        total_ms: cv.total_ms,
        reverted_to_raw: Boolean(cv.reverted),
        enhancement_difference: {
          changed_pixels: enhancementDifference.changed_pixels,
          changed_fraction: enhancementDifference.changed_fraction,
          mean_absolute_delta: enhancementDifference.mean_absolute_delta,
          max_absolute_delta: enhancementDifference.max_absolute_delta,
          caveat: "Amplified diagnostic map of real raw-to-enhanced changes; it is not an image-generation output.",
        },
      },
      memory_note:
        "Browser deviceMemory is a coarse device class, not available GPU memory. Process memory is measured separately through ADB; this page cannot measure total GPU allocations.",
    };
    const variants = {};
    if (data.cvStudy) {
      metadata.cv_study = {};
      for (const component of ["repair", "exposure", "edges", "color"]) {
        const begin = performance.now(),
          variant = enhance(pixels, profile.size, profile.size, [component]),
          difference = differenceMap(cv.pixels, variant.pixels);
        variants["omit-" + component] = await png(
          variant.pixels,
          profile.size,
          profile.size,
        );
        variants["omit-" + component + "-difference"] = await png(
          difference.pixels,
          profile.size,
          profile.size,
          profile.size,
        );
        metadata.cv_study["omit-" + component] = {
          ms: performance.now() - begin,
          applied: variant.applied,
          before: variant.before,
          after: variant.after,
          usable: variant.usable,
          difference: {
            changed_pixels: difference.changed_pixels,
            changed_fraction: difference.changed_fraction,
            mean_absolute_delta: difference.mean_absolute_delta,
            max_absolute_delta: difference.max_absolute_delta,
          },
        };
      }
      const stress = cvStressStudy(pixels, profile.size, profile.size);
      metadata.cv_stress = {
        scope: "Controlled diagnostic copies only; these defects are not added to the generated output.",
        speckle: {
          injected_pixels: stress.speckle.injected_pixels,
          corrected_pixels: Math.round(stress.speckle.corrected_pixels),
        },
        color: { cast: stress.color.cast },
      };
      for (const name of ["speckle", "color"]) {
        const item = stress[name];
        variants[`stress-${name}-input`] = await png(
          item.input,
          profile.size,
          profile.size,
        );
        if (item.output)
          variants[`stress-${name}-output`] = await png(
            item.output,
            profile.size,
            profile.size,
          );
      }
    }
    postMessage({ type: "result", raw, enhanced, difference, metadata, variants });
  } catch (error) {
    const message = error.message || String(error),
      resourceFailure = /memory|non-finite|invalid pixels|exceeded|worker stopped|webgpu.*lost/i.test(message);
    postMessage({
      type: "error",
      message,
      code: resourceFailure ? "resource" : "generation",
      profile: profile
        ? { key: profile.key, fallback_chain: profile.fallback_chain }
        : null,
      elapsed_ms: performance.now() - started,
    });
  } finally {
    if (active) await active.release().catch(() => {});
  }
};
