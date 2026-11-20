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
