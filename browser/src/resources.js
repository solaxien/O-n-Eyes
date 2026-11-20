export function chooseProfile(
  mode,
  { memoryGiB = 4, stepMs, keepLoaded = false } = {},
) {
  const profiles = {
    preview: { size: 256, steps: 4 },
    eco: { size: 256, steps: 6 },
    balanced: { size: 256, steps: 12 },
    quality: { size: 384, steps: 12 },
  };
  if (mode === "auto") {
    const constrained = memoryGiB < 8;
    const measuredSlow = Number.isFinite(stepMs) && stepMs * 6 + 1500 > 10000;
    const key = constrained || measuredSlow ? "preview" : "eco";
    return {
      ...profiles[key],
      reason: constrained
        ? "Four steps selected for the browser-reported memory class."
        : measuredSlow
          ? "Four steps selected from observed diffusion time."
          : "Six steps selected for the warm ten-second budget; initial model loading is additional.",
      retained: keepLoaded,
    };
  }
  const profile = profiles[mode] || profiles.eco;
  if (memoryGiB < 8 && profile.size > 256)
    return {
      ...profiles.eco,
      reason: "Resolution reduced for the browser-reported memory class.",
    };
  return { ...profile, reason: "Selected local profile." };
}
