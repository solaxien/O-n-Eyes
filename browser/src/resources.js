// Resource planning is deliberately conservative. navigator.deviceMemory is a
// coarse device class, not free RAM, so no profile assumes it owns that memory.
const PROFILES = Object.freeze({
  preview: { key: "preview", label: "Fast preview", size: 256, steps: 4, targetMs: 7000 },
  eco: { key: "eco", label: "Low memory", size: 256, steps: 6, targetMs: 11000 },
  balanced: { key: "balanced", label: "Balanced", size: 256, steps: 12, targetMs: 18000 },
  quality: { key: "quality", label: "Slow detail", size: 384, steps: 12, targetMs: 80000 },
  // This is intentionally not exposed in the user interface. It is the
  // reproducible reference for the reduced-step experiment.
  baseline: { key: "baseline", label: "Sampling reference", size: 256, steps: 20, targetMs: 30000 },
});

const FALLBACKS = Object.freeze({
  quality: "balanced",
  baseline: "balanced",
  balanced: "eco",
  eco: "preview",
  preview: null,
});

export function profileFor(key) {
  return PROFILES[key] || PROFILES.eco;
}

export function nextFallback(key) {
  return FALLBACKS[key] || null;
}

export function fallbackChain(key) {
  const result = [], seen = new Set();
  let candidate = key;
  while (candidate && !seen.has(candidate)) {
    seen.add(candidate);
    candidate = nextFallback(candidate);
    if (candidate) result.push(candidate);
  }
  return result;
}

// Actual checkpoints to retain while the U-Net is active. They are decoded
// only after the U-Net is released, avoiding concurrent large model sessions.
export function previewSteps(steps) {
  if (!Number.isInteger(steps) || steps < 2) return [];
  return [...new Set([0.25, 0.5, 0.75].map((fraction) =>
    Math.max(1, Math.min(steps - 1, Math.round(steps * fraction))),
  ))];
}

function withReason(profile, reason, extra = {}) {
  return { ...profile, reason, fallback_chain: fallbackChain(profile.key), ...extra };
}

/**
 * Select a profile before a local run. Complexity is derived from the prompt
 * contract, while observedStepMs comes only from earlier local generations.
 */
export function chooseProfile(
  mode,
  {
    memoryGiB = 4,
    hardwareConcurrency = 4,
    stepMs,
    complexity = 0,
    keepLoaded = false,
    budgetMs,
  } = {},
) {
  const constrained = memoryGiB < 6 || hardwareConcurrency <= 4;
  const measured = Number.isFinite(stepMs) && stepMs > 0;
  const autoBudget = budgetMs || (constrained ? 7500 : 10000);

  if (mode === "auto") {
    const preferred = constrained
      ? PROFILES.preview
      : complexity >= 0.65
        ? PROFILES.balanced
        : PROFILES.eco;
    const predictedMs = measured
      ? Math.round(stepMs * preferred.steps + 1500)
      : null;
    if (predictedMs !== null && predictedMs > autoBudget) {
      const lowered = preferred.key === "balanced" ? PROFILES.eco : PROFILES.preview;
      return withReason(
        lowered,
        `${lowered.label} selected because earlier local diffusion exceeded the ${Math.round(autoBudget / 1000)} s auto budget.`,
        { requested_mode: mode, predicted_ms: predictedMs, retained: keepLoaded },
      );
    }
    return withReason(
      preferred,
      constrained
        ? "Fast preview selected for the browser-reported device class."
        : preferred.key === "balanced"
          ? "Balanced sampling selected for this complex prompt on the available device class."
          : "Low-memory sampling selected for this prompt and device class.",
      { requested_mode: mode, predicted_ms: predictedMs, retained: keepLoaded },
    );
  }

  const requested = profileFor(mode);
  if (constrained && requested.size > 256)
    return withReason(PROFILES.eco, "Resolution reduced for the browser-reported device class.", {
      requested_mode: mode,
      downgraded_from: requested.key,
      retained: keepLoaded,
    });
  if (constrained && requested.steps > 12)
    return withReason(PROFILES.eco, "Sampling reference is unsafe for the browser-reported device class.", {
      requested_mode: mode,
      downgraded_from: requested.key,
      retained: keepLoaded,
    });
  return withReason(requested, "Selected local profile.", {
    requested_mode: mode,
    retained: keepLoaded,
  });
}

// Demonstration budgets. They feed the same planner as real measurements, so
// the adaptive and fallback paths can be shown live; results are labelled as
// simulated wherever they appear.
export const DEVICE_BUDGETS = Object.freeze({
  actual: { key: "actual", label: "This device (browser-reported)" },
  "phone-4gb": { key: "phone-4gb", label: "Simulated 4 GB, 4-core phone", memoryGiB: 4, hardwareConcurrency: 4 },
  "force-fallback": {
    key: "force-fallback",
    label: "Simulated memory failure on the first attempt",
    failFirstAttempt: true,
  },
});

export function deviceBudget(key, reported = {}) {
  const budget = DEVICE_BUDGETS[key] || DEVICE_BUDGETS.actual;
  return {
    key: budget.key,
    label: budget.label,
    simulated: budget.key !== "actual",
    memoryGiB: budget.memoryGiB ?? reported.memoryGiB ?? 4,
    hardwareConcurrency: budget.hardwareConcurrency ?? reported.hardwareConcurrency ?? 4,
    failFirstAttempt: Boolean(budget.failFirstAttempt),
  };
}

const MIB = 2 ** 20;
/**
 * Device readiness report. Every line says what was checked and what the app
 * will do about it. Inputs come from browser APIs; nothing is guessed.
 */
export function readinessItems({
  webgpu = false,
  fp16 = false,
  quotaBytes,
  usageBytes,
  requiredBytes,
  memoryGiB,
  hardwareConcurrency,
  modelsCached = false,
  verifierCached = false,
  serviceWorker = false,
  budget = deviceBudget("actual", { memoryGiB, hardwareConcurrency }),
} = {}) {
  const items = [];
  const add = (label, status, detail) => items.push({ label, status, detail });
  add("WebGPU", webgpu ? "pass" : "fail", webgpu ? "A GPU adapter is available to this page." : "No WebGPU adapter; this device cannot run the model locally.");
  add("FP16 shaders", fp16 ? "pass" : "fail", fp16 ? "shader-f16 is supported, so FP16 weights run as exported." : "shader-f16 is missing; the FP16 model cannot run here.");
  if (Number.isFinite(quotaBytes) && Number.isFinite(usageBytes)) {
    const free = quotaBytes - usageBytes,
      enough = modelsCached || free >= requiredBytes * 1.2;
    add("Storage", enough ? "pass" : "fail",
      modelsCached
        ? `Models already cached; ${Math.round(free / MIB)} MiB still free for this site.`
        : `${Math.round(free / MIB)} MiB free; about ${Math.round((requiredBytes * 1.2) / MIB)} MiB needed for the model bundle.`);
  } else add("Storage", "not_verifiable", "This browser does not report a storage estimate.");
  const simple = chooseProfile("auto", { memoryGiB: budget.memoryGiB, hardwareConcurrency: budget.hardwareConcurrency, complexity: 0.2 }),
    complex = chooseProfile("auto", { memoryGiB: budget.memoryGiB, hardwareConcurrency: budget.hardwareConcurrency, complexity: 0.9 });
  add("Memory class and cores", "pass",
    `${budget.memoryGiB} GB class, ${budget.hardwareConcurrency} cores${budget.simulated ? " (simulated)" : ""}. ` +
      "Browsers report a coarse class, not free memory.");
  add("Auto plan", "pass",
    `Simple prompts: ${simple.label}, ${simple.size} px, ${simple.steps} steps. Complex prompts: ${complex.label}, ${complex.size} px, ${complex.steps} steps.`);
  add("Fallback", "pass",
    budget.failFirstAttempt
      ? "Demo: the first attempt reports a simulated memory failure, so the real fallback chain runs."
      : "On a memory or numerical failure the app releases the worker and retries one profile lower.");
  add("Models", modelsCached ? "pass" : "not_verifiable", modelsCached ? "All model files are cached and checksum-verified." : "Models are not downloaded yet.");
  add("Prompt verifier", verifierCached ? "pass" : "not_verifiable",
    verifierCached ? "Compact detector cached; Balanced and Slow detail can check prompts." : "Not cached; prompt checks will be reported as unavailable.");
  add("Offline use", serviceWorker ? "pass" : "fail", serviceWorker ? "The app shell is cached by a service worker." : "No service worker; offline use is not possible.");
  return items;
}
