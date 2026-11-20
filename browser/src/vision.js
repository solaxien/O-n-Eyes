// Classical image operations, implemented locally without a learned enhancer.
const clamp = (x) => Math.max(0, Math.min(255, Math.round(x)));

function qualityAssessment(metrics, artifactFraction) {
  // A local no-reference heuristic for decode/post-processing diagnostics. It
  // intentionally measures pixel health only; it cannot rate aesthetics or
  // prompt alignment.
  const contrast = Math.min(1, Math.max(0, (metrics.contrast - 8) / 62));
  const sharpness = 1 - Math.exp(-Math.max(0, metrics.laplacian_variance) / 180);
  const clipping = 1 - Math.min(1, metrics.clipped_fraction / 0.12);
  const artifacts = 1 - Math.min(1, artifactFraction / 0.035);
  return {
    kind: "no-reference-pixel-health-v1",
    score: Math.round(100 * (0.3 * contrast + 0.3 * sharpness + 0.2 * clipping + 0.2 * artifacts)),
    components: {
      contrast: Number(contrast.toFixed(4)),
      sharpness: Number(sharpness.toFixed(4)),
      clipping: Number(clipping.toFixed(4)),
      artifact_cleanliness: Number(artifacts.toFixed(4)),
    },
    caveat: "Pixel-health heuristic only; it does not measure realism, aesthetics, or prompt consistency.",
  };
}

export function hsv(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const high = Math.max(r, g, b),
    low = Math.min(r, g, b),
    delta = high - low;
  let hue = 0;
  if (delta) {
    if (high === r) hue = 60 * (((g - b) / delta) % 6);
    else if (high === g) hue = 60 * ((b - r) / delta + 2);
    else hue = 60 * ((r - g) / delta + 4);
  }
  return { h: hue < 0 ? hue + 360 : hue, s: high ? delta / high : 0, v: high };
}

export function isColor(color, value) {
  const { h, s, v } = value;
  switch (color) {
    case "red": return s >= 0.27 && v >= 0.15 && (h <= 18 || h >= 345);
    case "orange": return s >= 0.27 && v >= 0.2 && h > 18 && h < 45;
    case "yellow": return s >= 0.25 && v >= 0.25 && h >= 45 && h <= 70;
    case "green": return s >= 0.2 && v >= 0.15 && h > 70 && h < 170;
    case "blue": return s >= 0.2 && v >= 0.15 && h >= 195 && h <= 255;
    case "purple": return s >= 0.2 && v >= 0.15 && h > 255 && h < 300;
    case "pink": return s >= 0.16 && v >= 0.3 && h >= 300 && h < 345;
    case "brown": return s >= 0.25 && v >= 0.12 && v <= 0.72 && h >= 15 && h <= 45;
    case "golden": return s >= 0.25 && v >= 0.42 && h >= 38 && h <= 55;
    case "black": return v <= 0.2;
    case "white": return s <= 0.15 && v >= 0.8;
    case "gray": return s <= 0.15 && v > 0.2 && v < 0.8;
    case "silver": return s <= 0.15 && v >= 0.45 && v <= 0.92;
    default: return false;
  }
}

// This deliberately measures only global pixel evidence. It is useful as a
// cheap diagnostic for requested colours, but cannot associate a colour with
// an object or establish a spatial relationship.
export function colorEvidence(pixels, width, height, analysis, stride = 4) {
  const colors = [
    ...new Set(analysis.objects.flatMap((object) => object.color ? [object.color] : [])),
  ];
  if (!colors.length)
    return { kind: "global-color-pixels", sample_stride: stride, samples: 0, requested: [] };
  const counts = Object.fromEntries(colors.map((color) => [color, 0]));
  let samples = 0;
  for (let y = 0; y < height; y += stride)
    for (let x = 0; x < width; x += stride) {
      const i = (y * width + x) * 4,
        value = hsv(pixels[i], pixels[i + 1], pixels[i + 2]);
      samples++;
      for (const color of colors)
        if (isColor(color, value)) counts[color]++;
    }
  return {
    kind: "global-color-pixels",
    sample_stride: stride,
    samples,
    requested: colors.map((color) => {
      const fraction = counts[color] / samples;
      return {
        color,
        fraction: Number(fraction.toFixed(5)),
        detected: fraction >= 0.002,
      };
    }),
    caveat: "Global colour evidence only; it does not verify object identity, count, or layout.",
  };
}

// Produce an intentionally amplified diagnostic view of a real pixel delta.
// It is never used as a generated result or an enhancement; it only makes a
// subtle ablation difference inspectable at normal display size.
export function differenceMap(reference, candidate) {
  const count = reference.length / 4,
    delta = new Float32Array(count);
  let changed = 0,
    total = 0,
    maximum = 0;
  for (let pixel = 0; pixel < count; pixel++) {
    const i = pixel * 4,
      value =
        (Math.abs(reference[i] - candidate[i]) +
          Math.abs(reference[i + 1] - candidate[i + 1]) +
          Math.abs(reference[i + 2] - candidate[i + 2])) /
        3;
    delta[pixel] = value;
    total += value;
    if (value > 0) changed++;
    if (value > maximum) maximum = value;
  }
  const pixels = new Uint8ClampedArray(reference.length);
  for (let pixel = 0; pixel < count; pixel++) {
    const i = pixel * 4,
      // Square root makes small but genuine deltas visible without inventing
      // pixels that were not changed by the selected CV stage.
      intensity = maximum
        ? Math.round(255 * Math.sqrt(delta[pixel] / maximum))
        : 0;
    pixels[i] = intensity;
    pixels[i + 1] = Math.round(intensity * 0.16);
    pixels[i + 2] = Math.round(intensity * 0.04);
    pixels[i + 3] = 255;
  }
  return {
    pixels,
    changed_pixels: changed,
    changed_fraction: changed / count,
    mean_absolute_delta: total / count,
    max_absolute_delta: maximum,
  };
}

function inspect(rgb, width, height) {
  const count = width * height,
    gray = new Float32Array(count),
    hist = new Uint32Array(256);
  let clipped = 0,
    sum = 0,
    square = 0;
  for (let i = 0; i < count; i++) {
    const v =
      0.299 * rgb[i * 4] + 0.587 * rgb[i * 4 + 1] + 0.114 * rgb[i * 4 + 2];
    gray[i] = v;
    hist[clamp(v)]++;
    if (v < 3 || v > 252) clipped++;
  }
  const percentile = (p) => {
    let c = 0;
    for (let i = 0; i < 256; i++) {
      c += hist[i];
      if (c >= p * count) return i;
    }
    return 255;
  };
  let edges = 0;
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x,
        lap =
          gray[i - 1] +
          gray[i + 1] +
          gray[i - width] +
          gray[i + width] -
          4 * gray[i];
      sum += lap;
      square += lap * lap;
      if (
        Math.abs(gray[i + 1] - gray[i - 1]) +
          Math.abs(gray[i + width] - gray[i - width]) >
        28
      )
        edges++;
    }
  const interior = (width - 2) * (height - 2);
  return {
    gray,
    metrics: {
      luminance_span: percentile(0.99) - percentile(0.01),
      contrast: percentile(0.9) - percentile(0.1),
      clipped_fraction: clipped / count,
      laplacian_variance: square / interior - (sum / interior) ** 2,
      edges,
    },
  };
}
const now = () => (globalThis.performance ? performance.now() : Date.now());

function changedPixels(before, after) {
  let changed = 0;
  for (let i = 0; i < before.length; i += 4)
    if (before[i] !== after[i] || before[i + 1] !== after[i + 1] || before[i + 2] !== after[i + 2]) changed++;
  return changed;
}

/**
 * Detect common decode defects on the raw image. Each check states its
 * threshold and what the pipeline does about it, including "nothing" when a
 * defect cannot be fixed with classical processing.
 */
export function artifactReport(raw, width, height, inspected = inspect(raw, width, height)) {
  const { gray, metrics } = inspected,
    count = width * height;
  let highlights = 0,
    shadows = 0,
    luminance = 0,
    castN = 0,
    castRG = 0,
    castBG = 0;
  for (let i = 0; i < count; i++) {
    const v = gray[i];
    luminance += v;
    if (v > 250) highlights++;
    if (v < 5) shadows++;
    const r = raw[i * 4],
      g = raw[i * 4 + 1],
      b = raw[i * 4 + 2];
    if (v > 45 && v < 215 && Math.max(r, g, b) - Math.min(r, g, b) < 40) {
      castN++;
      castRG += r - g;
      castBG += b - g;
    }
  }
  // Decoder grid: TAESD/VAE seams show as stronger steps every 8 pixels.
  let onGrid = 0,
    offGrid = 0,
    onN = 0,
    offN = 0;
  for (let y = 1; y < height; y++)
    for (let x = 1; x < width; x++) {
      const i = y * width + x,
        step = Math.abs(gray[i] - gray[i - 1]) + Math.abs(gray[i] - gray[i - width]);
      if (x % 8 === 0 || y % 8 === 0) {
        onGrid += step;
        onN++;
      } else {
        offGrid += step;
        offN++;
      }
    }
  const gridRatio = offGrid > 0 ? onGrid / onN / (offGrid / offN) : 1,
    castMagnitude = castN > count * 0.05 ? Math.hypot(castRG / castN, castBG / castN) : 0,
    mean = luminance / count;
  const check = (name, value, threshold, flagged, action) => ({
    name,
    value: Number(value.toFixed(4)),
    threshold,
    flagged,
    action,
  });
  return {
    checks: [
      check("blur (Laplacian variance)", metrics.laplacian_variance, "< 60 is soft", metrics.laplacian_variance < 60,
        "Not fixable by classical filtering; Balanced or more steps is the remedy."),
      check("exposure (mean luminance)", mean, "outside 70 to 185", mean < 70 || mean > 185,
        "Bounded gamma correction in the exposure stage."),
      check("clipped highlights", highlights / count, "> 0.05", highlights / count > 0.05,
        "Reported; later stages are blocked from increasing clipping."),
      check("crushed shadows", shadows / count, "> 0.05", shadows / count > 0.05,
        "Reported; later stages are blocked from increasing clipping."),
      check("colour cast (RGB offset on near-neutral pixels)", castMagnitude, "> 6 levels", castMagnitude > 6,
        "Bounded neutral colour balance."),
      check("decoder grid (8 px seam ratio)", gridRatio, "> 1.3", gridRatio > 1.3,
        "Reported only; no dedicated correction."),
    ],
    flagged: 0,
  };
}

function exposureGamma(mean) {
  if (mean < 70) return Math.max(0.75, Math.log(100 / 255) / Math.log(Math.max(1, mean) / 255));
  if (mean > 185) return Math.min(1.3, Math.log(160 / 255) / Math.log(mean / 255));
  return 1;
}

export function enhance(raw, width, height, omit = []) {
  const started = now(),
    before = inspect(raw, width, height),
    pixels = new Uint8ClampedArray(raw),
    mask = new Uint8Array(width * height),
    stages = [],
    record = (stage, status, reason, ms = 0, snapshot = null) =>
      stages.push({
        stage,
        status,
        reason,
        ms: Number(ms.toFixed(2)),
        changed_pixels: snapshot ? changedPixels(snapshot, pixels) : 0,
      });
  const artifacts_report = artifactReport(raw, width, height, before);
  artifacts_report.flagged = artifacts_report.checks.filter((c) => c.flagged).length;
  let artifacts = 0,
    tick = now();
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x,
        values = [];
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++)
          values.push(before.gray[i + dy * width + dx]);
      values.sort((a, b) => a - b);
      const median = values[4];
      // Isolated outliers only; preserve structured edges and clustered defects.
      const span = values[7] - values[1];
      if (Math.abs(before.gray[i] - median) >= 48 && span < 24) {
        mask[i] = 1;
        artifacts++;
      }
    }
  const fraction = artifacts / (width * height),
    usable = before.metrics.luminance_span >= 8 && fraction <= 0.035,
    detectMs = now() - tick;
  const result = (out, applied, after, extra = {}) => ({
    pixels: out,
    applied,
    before: before.metrics,
    after: after.metrics,
    artifact_fraction: fraction,
    quality: {
      before: qualityAssessment(before.metrics, fraction),
      after: qualityAssessment(after.metrics, fraction),
    },
    usable,
    stages,
    artifacts: artifacts_report,
    total_ms: Number((now() - started).toFixed(2)),
    ...extra,
  });
  if (!usable && !omit.includes("gate")) {
    record("gate", "applied", "Decode failed the pixel gate (flat image or too many outliers); raw pixels kept.", detectMs);
    return result(pixels, ["Pixel quality gate flagged decode"], before);
  }
  const applied = [];
  // Stage 1: isolated speckle repair.
  if (omit.includes("repair")) record("repair", "omitted", "Omitted for ablation.");
  else if (!artifacts) record("repair", "skipped", "No isolated speckles were found.", detectMs);
  else if (fraction > 0.004)
    record("repair", "skipped", "Too many outliers to treat as speckles; they are likely texture.", detectMs);
  else {
    const snapshot = new Uint8ClampedArray(pixels);
    tick = now();
    for (let i = 0; i < mask.length; i++)
      if (mask[i])
        for (let c = 0; c < 3; c++) {
          const a = [];
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++)
              if (dx || dy) a.push(raw[(i + dy * width + dx) * 4 + c]);
          a.sort((a, b) => a - b);
          pixels[i * 4 + c] = Math.round((a[3] + a[4]) / 2);
        }
    applied.push("Isolated speckle correction");
    record("repair", "applied", `${artifacts} isolated speckles replaced by their neighbourhood median.`, detectMs + now() - tick, snapshot);
  }
  // Stage 2: exposure (brightness) correction with a bounded gamma.
  let gateBase = before;
  const mean = before.gray.reduce((s, v) => s + v, 0) / before.gray.length,
    gamma = exposureGamma(mean);
  if (omit.includes("exposure")) record("exposure", "omitted", "Omitted for ablation.");
  else if (Math.abs(gamma - 1) < 0.02 || before.metrics.luminance_span < 40)
    record("exposure", "skipped", `Mean luminance ${mean.toFixed(0)} is within 70 to 185, so brightness was left unchanged.`);
  else {
    const snapshot = new Uint8ClampedArray(pixels),
      lut = new Uint8ClampedArray(256);
    tick = now();
    for (let v = 0; v < 256; v++) lut[v] = clamp(255 * (v / 255) ** gamma);
    for (let i = 0; i < pixels.length; i += 4)
      for (let c = 0; c < 3; c++) pixels[i + c] = lut[pixels[i + c]];
    applied.push("Exposure correction");
    // Brightening legitimately raises gradient sizes, so later edge-safety
    // checks compare against the exposure-corrected image.
    gateBase = inspect(pixels, width, height);
    record("exposure", "applied", `Mean luminance ${mean.toFixed(0)}; gamma ${gamma.toFixed(2)} applied equally to all channels so hues are kept.`, now() - tick, snapshot);
  }
  // Stage 3: edge-aware bilateral detail recovery.
  if (omit.includes("edges")) record("edges", "omitted", "Omitted for ablation.");
  else {
    const source = new Uint8ClampedArray(pixels);
    tick = now();
    for (let y = 1; y < height - 1; y++)
      for (let x = 1; x < width - 1; x++) {
        const i = y * width + x;
        let total = 0,
          luma = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const j = i + dy * width + dx,
              d = before.gray[j] - before.gray[i],
              // This remains edge-aware but admits enough neighbouring
              // luminance to estimate a local base layer for detail recovery.
              w = Math.exp(-0.5 * (dx * dx + dy * dy) - (d * d) / 1152);
            total += w;
            luma += w * before.gray[j];
          }
        // Recover bounded detail relative to the bilateral base. The former
        // smoothing-only direction made most valid images look unchanged.
        const change = Math.max(
          -10,
          Math.min(10, 0.8 * (before.gray[i] - luma / total)),
        );
        for (let c = 0; c < 3; c++)
          pixels[i * 4 + c] = clamp(source[i * 4 + c] + change);
      }
    applied.push("Edge-aware bilateral detail recovery");
    record("edges", "applied", "Bilateral base layer separates edges from flat areas; bounded detail is added back.", now() - tick, source);
  }
  // Stage 4: neutral colour balance and bounded contrast.
  if (omit.includes("color")) record("color", "omitted", "Omitted for ablation.");
  else {
    const snapshot = new Uint8ClampedArray(pixels);
    tick = now();
    let neutral = 0,
      means = [0, 0, 0];
    for (let i = 0; i < pixels.length; i += 4) {
      const rgb = [pixels[i], pixels[i + 1], pixels[i + 2]],
        l = (rgb[0] + rgb[1] + rgb[2]) / 3;
      if (l > 45 && l < 215 && Math.max(...rgb) - Math.min(...rgb) < 15) {
        neutral++;
        for (let c = 0; c < 3; c++) means[c] += rgb[c] - l;
      }
    }
    const reasons = [];
    if (neutral / (width * height) >= 0.35) {
      means = means.map((x) => Math.max(-3, Math.min(3, (x / neutral) * 0.35)));
      if (Math.max(...means.map(Math.abs)) >= 1) {
        for (let i = 0; i < pixels.length; i += 4)
          for (let c = 0; c < 3; c++)
            pixels[i + c] = clamp(pixels[i + c] - means[c]);
        applied.push("Conservative neutral colour balance");
        reasons.push("neutral colour cast reduced");
      } else reasons.push("no measurable colour cast");
    } else reasons.push("too few neutral pixels to estimate a cast safely");
    if (before.metrics.contrast >= 12 && before.metrics.contrast < 55) {
      for (let i = 0; i < pixels.length; i += 4)
        for (let c = 0; c < 3; c++)
          pixels[i + c] = clamp(128 + (pixels[i + c] - 128) * 1.035);
      applied.push("Bounded contrast correction");
      reasons.push("low contrast stretched by 3.5%");
    } else
      reasons.push(before.metrics.contrast < 12 ? "image too flat to stretch safely" : "contrast already adequate");
    const changed = changedPixels(snapshot, pixels) > 0;
    record("color", changed ? "applied" : "skipped", reasons.join("; ") + ".", now() - tick, snapshot);
  }
  const after = inspect(pixels, width, height);
  if (
    !omit.includes("gate") &&
    (after.metrics.clipped_fraction > before.metrics.clipped_fraction + 0.025 ||
      (gateBase.metrics.edges > 100 &&
        (after.metrics.edges < gateBase.metrics.edges * 0.72 ||
          // Detail recovery can legitimately cross the older 1.65× edge
          // ceiling on textured images; retain a safety ceiling for clearly
          // over-sharpened output without discarding useful enhancement.
          after.metrics.edges > gateBase.metrics.edges * 2.0)))
  ) {
    record("gate", "applied", "Enhancement increased clipping or changed edges too much, so the raw pixels were kept.");
    return result(new Uint8ClampedArray(raw), ["Quality gate retained raw pixels"], before, { reverted: true });
  }
  record("gate", omit.includes("gate") ? "omitted" : "skipped", omit.includes("gate") ? "Omitted for ablation." : "Enhanced image passed the accept-or-revert check.");
  return result(pixels, applied, after);
}
export async function png(pixels, width, height, display = 512) {
  const source = new OffscreenCanvas(width, height);
  source
    .getContext("2d")
    .putImageData(new ImageData(pixels, width, height), 0, 0);
  const output = new OffscreenCanvas(
      display,
      Math.round((height * display) / width),
    ),
    ctx = output.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, output.width, output.height);
  return output.convertToBlob({ type: "image/png" });
}

function knownColorCast(raw) {
  const cast = new Uint8ClampedArray(raw);
  for (let i = 0; i < cast.length; i += 4) {
    cast[i] = clamp(cast[i] * 0.72 + 6);
    cast[i + 1] = clamp(cast[i + 1] * 0.92);
    cast[i + 2] = clamp(cast[i + 2] * 1.16 + 4);
  }
  return cast;
}

function correctKnownColorCast(cast) {
  const corrected = new Uint8ClampedArray(cast);
  for (let i = 0; i < corrected.length; i += 4) {
    corrected[i] = clamp((corrected[i] - 6) / 0.72);
    corrected[i + 1] = clamp(corrected[i + 1] / 0.92);
    corrected[i + 2] = clamp((corrected[i + 2] - 4) / 1.16);
  }
  return corrected;
}

function injectSpeckles(raw, width, height) {
  const corrupted = new Uint8ClampedArray(raw),
    occupied = new Set(),
    // Stay below the production repair safety cap (0.4%); this is a sparse
    // isolated-speckle diagnostic, not a dense-noise denoiser demonstration.
    count = Math.max(8, Math.floor(width * height * 0.002));
  let state = 0x9e3779b9;
  while (occupied.size < count) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const x = 1 + (state % Math.max(1, width - 2));
    state = (state * 1664525 + 1013904223) >>> 0;
    const y = 1 + (state % Math.max(1, height - 2));
    occupied.add(y * width + x);
  }
  for (const pixel of occupied) {
    const i = pixel * 4,
      value =
        0.299 * corrupted[i] +
          0.587 * corrupted[i + 1] +
          0.114 * corrupted[i + 2] >
        128
          ? 0
          : 255;
    corrupted[i] = corrupted[i + 1] = corrupted[i + 2] = value;
  }
  return { pixels: corrupted, injected_pixels: occupied.size };
}

// This is a controlled visual demonstration, not a second enhancement pass on
// the output. It makes conditional CV stages inspectable by applying known
// defects to disposable copies of the decoded image.
export function cvStressStudy(raw, width, height) {
  const speckle = injectSpeckles(raw, width, height),
    repaired = enhance(speckle.pixels, width, height, ["edges", "color", "gate"]),
    cast = knownColorCast(raw),
    invalid = new Uint8ClampedArray(raw.length);
  for (let i = 3; i < invalid.length; i += 4) invalid[i] = 255;
  const rejected = enhance(invalid, width, height);
  return {
    speckle: {
      input: speckle.pixels,
      output: repaired.pixels,
      injected_pixels: speckle.injected_pixels,
      corrected_pixels: repaired.artifact_fraction * width * height,
    },
    color: {
      input: cast,
      output: correctKnownColorCast(cast),
      cast: "known cool colour cast",
    },
    quality_gate: {
      input: invalid,
      usable: rejected.usable,
      rejection: rejected.applied[0],
    },
  };
}
