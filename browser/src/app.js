import { getAsset, prepareModels, MODEL_CACHE } from "./cache.js";
import { chooseProfile, nextFallback } from "./resources.js";
import { analyzePrompt, semanticSummary } from "./prompt.js";
import { candidateScore, candidateSeed, pickBest, selectionPolicy, shouldContinue } from "./selection.js";
// Profiles where the compact verifier runs; it then steers retries or picks
// the best candidate. Preview and Eco keep the single-image fast path.
const VERIFY_PROFILES = ["balanced", "quality"];
const $ = (id) => document.getElementById(id),
  state = {
    ready: false,
    busy: false,
    comparing: false,
    preparing: false,
    worker: null,
    urls: {},
    studyUrls: [],
    processUrls: [],
    comparisonUrls: [],
    result: null,
  };
// The one-reload guard is only for the current shell upgrade. A later page
// load may safely upgrade again when a newer service worker is deployed.
try {
  sessionStorage.removeItem("p3-shell-refreshed");
} catch {}
const fail = (message) => {
  $("error").textContent = message;
  $("error").hidden = false;
};
const formatMs = (value) =>
  Number.isFinite(value) ? `${(value / 1000).toFixed(2)} s` : "not recorded";
function syncControls() {
  const unavailable = !state.ready || state.busy || state.comparing;
  $("generate").disabled = unavailable;
  $("compare").disabled = unavailable;
  $("prepare").disabled = state.preparing || state.busy || state.comparing;
}
function setComparisonStatus(message) {
  $("comparison-status").textContent = message;
}
function markModelsReady({ scrollToControls = false } = {}) {
  state.ready = true;
  $("setup-panel").classList.add("ready");
  $("prepare").textContent = "Recheck cached models";
  syncControls();
  if (scrollToControls)
    requestAnimationFrame(() =>
      $("form").scrollIntoView({ behavior: "smooth", block: "start" }),
    );
}
function updateProfilePreview() {
  const plan = analyzePrompt($("prompt").value),
    profile = chooseProfile($("mode").value, {
      memoryGiB: navigator.deviceMemory || 4,
      hardwareConcurrency: navigator.hardwareConcurrency || 4,
      complexity: plan.complexity,
      keepLoaded: $("keep-loaded").checked,
    });
  $("profile-preview").textContent =
    `Local plan: ${profile.label} · ${profile.size} px · ${profile.steps} steps. ${profile.reason}`;
}
function revoke(urls) {
  for (const url of urls) URL.revokeObjectURL(url);
}
function metric(label, value) {
  const card = document.createElement("div"),
    title = document.createElement("span"),
    result = document.createElement("strong");
  card.className = "metric";
  title.className = "metric-label";
  title.textContent = label;
  result.textContent = value;
  card.append(title, result);
  return card;
}
function setMetrics(id, values) {
  $(id).replaceChildren(...values.map(([label, value]) => metric(label, value)));
}
function appendText(parent, text, tag = "p") {
  const node = document.createElement(tag);
  node.textContent = text;
  parent.append(node);
  return node;
}
function clearCvStudy() {
  revoke(state.studyUrls);
  state.studyUrls = [];
  $("cv-variants").replaceChildren();
  $("cv-stress-cards").replaceChildren();
  $("cv-stress").hidden = true;
  $("cv-study-results").hidden = true;
}
function clearComparison() {
  revoke(state.comparisonUrls);
  state.comparisonUrls = [];
  $("comparison-grid").replaceChildren();
  $("comparison").hidden = true;
}
const PROCESS_STAGES = [
  ["prompt", "Prompt plan"],
  ["text", "Text encoding"],
  ["denoise", "Diffusion"],
  ["preview", "Checkpoint decode"],
  ["decode", "Final decode"],
  ["cv", "Classical vision"],
];
function clearProcessVisualizer() {
  revoke(state.processUrls);
  state.processUrls = [];
  state.processStates = {};
  state.processVisualizing = false;
  $("process-timeline").replaceChildren();
  $("process-previews").replaceChildren();
  $("process-visualizer").hidden = true;
}
function renderProcessTimeline() {
  $("process-timeline").replaceChildren(
    ...PROCESS_STAGES.map(([id, label]) => {
      const entry = document.createElement("li"),
        title = document.createElement("strong"),
        detail = document.createElement("small"),
        status = state.processStates[id];
      entry.className = status?.status || "pending";
      title.textContent = label;
      detail.textContent = status?.detail || "Waiting";
      entry.append(title, detail);
      return entry;
    }),
  );
}
function startProcessVisualizer(enabled) {
  clearProcessVisualizer();
  if (!enabled) return;
  state.processVisualizing = true;
  $("process-visualizer").hidden = false;
  $("process-note").textContent =
    "Captures three real latent checkpoints during diffusion, then decodes them only after releasing the denoiser. This adds local decoder work.";
  renderProcessTimeline();
}
function updateProcessVisualizer(event) {
  if (!state.processVisualizing) return;
  state.processStates[event.stage] = {
    status: event.status,
    detail: event.detail,
  };
  renderProcessTimeline();
}
function addProcessPreview({ blob, step, steps }) {
  if (!state.processVisualizing) return;
  const figure = document.createElement("figure"),
    image = document.createElement("img"),
    caption = document.createElement("figcaption"),
    url = URL.createObjectURL(blob);
  state.processUrls.push(url);
  image.src = url;
  image.alt = `Actual latent checkpoint after diffusion step ${step} of ${steps}`;
  caption.textContent = `Actual latent checkpoint · step ${step}/${steps}`;
  figure.append(image, caption);
  $("process-previews").append(figure);
}
let adapter,
  shellUpdateUnavailable = false;
try {
  if (!navigator.gpu)
    throw Error(
      "WebGPU is unavailable in this browser. No server fallback is used.",
    );
  adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter || !adapter.features.has("shader-f16"))
    throw Error(
      "This browser needs a WebGPU adapter with FP16 support. No server fallback is used.",
    );
  $("capability").textContent =
    `${adapter.info.vendor || "Device"} GPU available · ${navigator.deviceMemory || "unknown"} GB memory class. Models are checked before use.`;
  $("download-status").textContent =
    "Preparing the offline browser runtime (about 29 MB on first use)…";
  if ("serviceWorker" in navigator) {
    if (navigator.serviceWorker.controller)
      navigator.serviceWorker.addEventListener(
        "controllerchange",
        () => {
          // A rebuilt shell has arrived. Reload once so this tab picks up its
          // new HTML, CSS and JS; model CacheStorage is intentionally retained.
          if (!sessionStorage.getItem("p3-shell-refreshed")) {
            sessionStorage.setItem("p3-shell-refreshed", "1");
            location.reload();
          }
        },
        { once: true },
      );
    try {
      // register() already performs the browser's normal update check. Calling
      // update() again made a transient local-server restart fatal even when a
      // valid cached shell was already controlling this page.
      await navigator.serviceWorker.register("/sw.js");
      await navigator.serviceWorker.ready;
    } catch (error) {
      if (!navigator.serviceWorker.controller)
        throw Error(
          "Could not start offline storage. Confirm the local server is running, then reload this page.",
        );
      shellUpdateUnavailable = true;
    }
  } else
    throw Error(
      "Offline storage requires a browser with service-worker support.",
    );
  $("prepare").disabled = false;
  $("download-status").textContent =
    shellUpdateUnavailable
      ? "Using the cached application shell. Restart the local server before downloading or checking for a UI update."
      : "Ready to download about 896 MB of model files. Completed files are reused after interruption.";
  try {
    const index = await fetch("/model-index.json").then((r) => r.json()),
      manifest = await (
        await getAsset("manifest.json", { sha256: index.manifest_sha256 })
      ).json(),
      cache = await caches.open(MODEL_CACHE);
    let complete = true;
    for (const file of Object.values(manifest.components).flatMap(
      (c) => c.files,
    )) {
      const r = await cache.match("/models/" + file.path);
      if (!r || r.headers.get("x-model-sha256") !== file.sha256) {
        complete = false;
        break;
      }
    }
    if (complete && Object.keys(manifest.components).length === 3) {
      markModelsReady();
      $("stage").textContent = "Models cached. Ready for local generation.";
      $("download-status").textContent =
        "Model files are cached on this device. Generation can run offline.";
      setComparisonStatus(
        "Ready for a matched 20-step reference and optimized local comparison.",
      );
    }
  } catch {}
} catch (error) {
  $("capability").textContent = error.message;
  fail(error.message);
}

$("prepare").onclick = async () => {
  clearTimeout(state.idleTimer);
  state.worker?.terminate();
  state.worker = null;
  state.ready = false;
  $("setup-panel").classList.remove("ready");
  $("prepare").textContent = "Download models for offline use";
  state.preparing = true;
  $("error").hidden = true;
  syncControls();
  $("download-progress").hidden = false;
  $("cancel").hidden = false;
  const controller = new AbortController();
  state.controller = controller;
  try {
    await prepareModels(({ file, bytes, total }) => {
      $("download-progress").value = (bytes / total) * 100;
      $("download-status").textContent =
        `${(bytes / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB · ${file}`;
    }, controller.signal);
    const persistent = await navigator.storage.persist().catch(() => false);
    markModelsReady({ scrollToControls: true });
    $("stage").textContent = "Ready. Generation runs entirely in this browser.";
    $("download-status").textContent =
      `Model files are cached and verified. ${persistent ? "Persistent storage granted." : "The browser may evict cached files when storage is low."}`;
    setComparisonStatus(
      "Ready for a matched 20-step reference and optimized local comparison.",
    );
  } catch (error) {
    fail(
      controller.signal.aborted
        ? "Download paused. Completed files are retained; press Download to continue."
        : error.message,
    );
  } finally {
    state.preparing = false;
    syncControls();
    $("cancel").hidden = true;
    state.controller = null;
  }
};
$("cancel").onclick = () => state.controller?.abort();
for (const id of ["prompt", "mode", "keep-loaded"])
  $(id).addEventListener("input", updateProfilePreview);
updateProfilePreview();

function renderCvStudy(result) {
  clearCvStudy();
  const variants = Object.entries(result.variants || {}).filter(
    ([key]) => key.startsWith("omit-") && !key.endsWith("-difference"),
  );
  if (!variants.length) return;
  const labels = {
    "omit-repair": "Without speckle repair",
    "omit-exposure": "Without exposure correction",
    "omit-edges": "Without edge-aware filtering",
    "omit-color": "Without colour correction",
  };
  for (const [key, blob] of variants) {
    const figure = document.createElement("figure"),
      image = document.createElement("img"),
      caption = document.createElement("figcaption"),
      title = document.createElement("strong"),
      detail = document.createElement("span"),
      study = result.metadata.cv_study?.[key],
      url = URL.createObjectURL(blob);
    state.studyUrls.push(url);
    figure.className = "variant-card";
    image.src = url;
    image.alt = labels[key] || key;
    title.textContent = labels[key] || key;
    const changed = study?.difference;
    detail.textContent = changed
      ? `${(changed.changed_fraction * 100).toFixed(3)}% pixels changed · mean RGB delta ${changed.mean_absolute_delta.toFixed(3)}.`
      : "Same decoded image; one CV stage omitted.";
    caption.append(title, detail);
    const differenceBlob = result.variants?.[key + "-difference"];
    if (differenceBlob) {
      const disclosure = document.createElement("details"),
        summary = document.createElement("summary"),
        differenceImage = document.createElement("img"),
        differenceUrl = URL.createObjectURL(differenceBlob);
      state.studyUrls.push(differenceUrl);
      summary.textContent = changed?.changed_pixels
        ? "Show amplified pixel-difference map"
        : "No pixels changed: this stage did not activate";
      differenceImage.src = differenceUrl;
      differenceImage.alt = `Amplified difference map for ${labels[key] || key}`;
      differenceImage.className = "difference-map";
      disclosure.append(summary, differenceImage);
      caption.append(disclosure);
    }
    figure.append(image, caption);
    $("cv-variants").append(figure);
  }
  renderStressStudy(result);
  $("cv-study-results").hidden = false;
}

function stressImage(blob, label) {
  const figure = document.createElement("figure"),
    image = document.createElement("img"),
    caption = document.createElement("figcaption"),
    url = URL.createObjectURL(blob);
  state.studyUrls.push(url);
  image.src = url;
  image.alt = label;
  caption.textContent = label;
  figure.append(image, caption);
  return figure;
}

function renderStressStudy(result) {
  const study = result.metadata.cv_stress;
  if (!study) return;
  const descriptions = {
    speckle: `Injected ${study.speckle.injected_pixels} isolated speckles, then ran only local speckle repair.`,
    color: "Injected a known cool colour cast, then applied the matching calibration correction.",
  };
  for (const name of ["speckle", "color"]) {
    const card = document.createElement("article"),
      title = document.createElement("h4"),
      detail = document.createElement("p"),
      input = result.variants?.[`stress-${name}-input`],
      output = result.variants?.[`stress-${name}-output`];
    card.className = "stress-card";
    title.textContent =
      name === "speckle"
        ? "Speckle correction"
        : name === "color"
          ? "Colour correction"
        : "Colour correction";
    detail.textContent = descriptions[name];
    card.append(title, detail);
    if (input && output) {
      const pair = document.createElement("div");
      pair.className = "stress-pair";
      pair.append(
        stressImage(input, "Injected defect"),
        stressImage(output, "Corrected copy"),
      );
      card.append(pair);
    }
    $("cv-stress-cards").append(card);
  }
  $("cv-stress").hidden = false;
}

function renderInsights(result) {
  const { metadata } = result,
    cv = metadata.cv,
    plan = metadata.semantic_plan,
    colorEvidence = plan.color_evidence?.requested || [];
  $("run-insights").hidden = false;
  const qualityState = $("quality-state");
  qualityState.textContent = cv.usable ? "Pixel checks passed" : "Pixel checks flagged";
  qualityState.classList.toggle("warn", !cv.usable);
  setMetrics("run-summary", [
    ["Selected profile", `${metadata.profile.key} · ${metadata.steps} steps`],
    ["Native generation", `${metadata.native_resolution} × ${metadata.native_resolution} px`],
    ["Worker runtime", formatMs(metadata.cached_generation_ms)],
    ["U-Net calls", String(metadata.unet_calls)],
    [
      "Process previews",
      metadata.process_visualizer.enabled
        ? `${metadata.process_visualizer.checkpoints.length} checkpoints · ${formatMs(metadata.stages_ms.process_preview_decode)}`
        : "Disabled for fast path",
    ],
  ]);
  const stack = [
    "BK-SDM-Tiny latent diffusion: a compact denoiser rather than a full Stable Diffusion deployment.",
    "FP16 neural weights and neural inputs: lower model/storage precision than FP32.",
    "DPM-Solver++ sampling: the active profile uses fewer denoising iterations than the 20-step reference.",
    "TAESD lightweight decoder: converts the latent into pixels locally.",
    metadata.models_retained
      ? "Sessions retained for repeat latency; this trades memory for faster subsequent runs."
      : "Text encoder, denoiser, and decoder run in stages and are released between stages.",
    metadata.conditioning_prompt !== metadata.prompt
      ? "Local structured prompt conditioning repeated recognised attributes and relations for the text encoder."
      : "Original prompt was encoded directly; no structured rewrite was applied.",
  ];
  $("optimization-stack").replaceChildren();
  for (const item of stack) appendText($("optimization-stack"), item, "li");
  setMetrics("quality-summary", [
    ["Contrast", `${cv.before.contrast.toFixed(1)} → ${cv.after.contrast.toFixed(1)}`],
    ["Luminance span", `${cv.before.luminance_span.toFixed(1)} → ${cv.after.luminance_span.toFixed(1)}`],
    ["Edge variance", `${cv.before.laplacian_variance.toFixed(1)} → ${cv.after.laplacian_variance.toFixed(1)}`],
    ["Artifact pixels", `${(cv.artifact_fraction * 100).toFixed(3)}%`],
    ["Enhanced pixels", `${(cv.enhancement_difference.changed_fraction * 100).toFixed(3)}%`],
    ["Mean RGB delta", cv.enhancement_difference.mean_absolute_delta.toFixed(3)],
    ["Pixel-health score", `${cv.quality.before.score} → ${cv.quality.after.score} / 100`],
    ["CV actions", cv.applied.length ? cv.applied.join(" · ") : "No correction needed"],
    ["Colour evidence", colorEvidence.length ? "Recorded below" : "No requested colour"],
  ]);
  const evidence = $("semantic-evidence");
  evidence.replaceChildren();
  const summary = semanticSummary({
    objects: plan.requested_objects,
    relations: plan.requested_relations,
  });
  appendText(evidence, `Requested objects: ${summary.objects}.`);
  appendText(evidence, `Requested relations: ${summary.relations}.`);
  if (colorEvidence.length)
    appendText(
      evidence,
      `Global colour samples: ${colorEvidence.map((item) => `${item.color} ${item.detected ? "detected" : "not detected"} (${(item.fraction * 100).toFixed(2)}%)`).join("; ")}.`,
    );
  const caveat = appendText(evidence, plan.verification);
  caveat.className = "muted";
  renderCvStages(cv);
  renderVerifier(metadata);
  renderCandidates(metadata);
  renderCvStudy(result);
}

const STAGE_NAMES = {
  repair: "Speckle repair",
  exposure: "Exposure",
  edges: "Edge-aware detail",
  color: "Colour and contrast",
  gate: "Accept-or-revert gate",
};
function renderCvStages(cv) {
  const stagesList = $("cv-stages"),
    artifactList = $("cv-artifacts");
  stagesList.replaceChildren();
  artifactList.replaceChildren();
  for (const stage of cv.stages || []) {
    const item = appendText(
      stagesList,
      `${STAGE_NAMES[stage.stage] || stage.stage}: ${stage.status}. ${stage.reason}${stage.changed_pixels ? ` ${stage.changed_pixels} pixels changed in ${stage.ms} ms.` : ""}`,
      "li",
    );
    item.className = "check " + (stage.status === "applied" ? "pass" : "not_verifiable");
  }
  for (const check of cv.artifact_checks?.checks || []) {
    const item = appendText(
      artifactList,
      `${check.name}: ${check.value} (flag if ${check.threshold}). ${check.flagged ? check.action : "Not flagged."}`,
      "li",
    );
    item.className = "check " + (check.flagged ? "fail" : "pass");
  }
}
const STATUS_TEXT = { pass: "Passed", fail: "Failed", not_verifiable: "Not verifiable here" };
function renderVerifier(metadata) {
  const list = $("verifier-checks"),
    verifier = metadata.verifier;
  list.replaceChildren();
  $("verifier-section").hidden = false;
  if (!verifier) {
    appendText(
      list,
      VERIFY_PROFILES.includes(metadata.profile.key)
        ? "Nothing in this prompt could be checked on this device."
        : "Prompt checks run in Balanced and Slow detail modes. This mode keeps the single-image fast path.",
      "li",
    ).className = "muted";
    return;
  }
  if (verifier.unavailable) {
    appendText(list, verifier.reason, "li").className = "muted";
    return;
  }
  const groups = [
    ["Object", verifier.object_checks],
    ["Count", verifier.count_checks],
    ["Colour", verifier.color_checks],
    ["Layout", verifier.relation_checks],
  ];
  for (const [kind, checks] of groups)
    for (const check of checks) {
      const detail =
        check.found !== undefined
          ? ` (found ${check.found})`
          : check.reason
            ? ` (${check.reason})`
            : "";
      const item = appendText(list, `${kind}: ${check.label}. ${STATUS_TEXT[check.status]}${detail}`, "li");
      item.className = "check " + check.status;
    }
  appendText(
    list,
    `Checked in ${formatMs(verifier.ms)} by ${verifier.verifier_scope}. These checks steer retries; they are not the final benchmark score.`,
    "li",
  ).className = "muted";
}
function renderCandidates(metadata) {
  const list = $("candidate-timeline"),
    selection = metadata.selection;
  list.replaceChildren();
  if (!selection || selection.candidates.length < 2) {
    $("candidate-section").hidden = true;
    return;
  }
  $("candidate-section").hidden = false;
  for (const candidate of selection.candidates) {
    const item = document.createElement("li");
    item.className = candidate.selected ? "done" : "";
    item.textContent = `Seed ${candidate.seed}, ${candidate.profile}, ${formatMs(candidate.elapsed_ms)}: ${candidate.verifier}${candidate.selected ? ". Selected" : ""}`;
    list.append(item);
  }
  appendText(list, `${selection.stop_reason}${selection.best_effort ? " Best effort result." : ""}`, "li").className = "muted";
}

function showResult(result, visibleMs) {
  for (const url of Object.values(state.urls)) URL.revokeObjectURL(url);
  state.urls = {
    raw: URL.createObjectURL(result.raw),
    enhanced: URL.createObjectURL(result.enhanced),
    difference: URL.createObjectURL(result.difference),
  };
  state.result = result;
  $("image").src = state.urls.enhanced;
  $("image").alt = result.metadata.prompt;
  $("image").hidden = false;
  $("placeholder").hidden = true;
  $("toggle").hidden = false;
  for (const button of document.querySelectorAll("[data-image]"))
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.image === "enhanced"),
    );
  $("save").href = state.urls.enhanced;
  $("save").hidden = false;
  $("details").hidden = false;
  result.metadata.submit_to_result_ms = visibleMs;
  $("metadata").textContent = JSON.stringify(result.metadata, null, 2);
  $("timing").textContent =
    `${(visibleMs / 1000).toFixed(2)} s on this device · ${result.metadata.native_resolution} px native → 512 px display`;
  const recovery = result.metadata.recovery;
  $("stage").textContent = recovery?.used
    ? `Generated locally after fallback to ${result.metadata.profile.key}. ${recovery.failures.length} resource-limited attempt${recovery.failures.length === 1 ? "" : "s"} recovered.`
    : result.metadata.selection?.candidates.length > 1
      ? `Generated locally. Best of ${result.metadata.selection.candidates.length} on-device candidates shown.`
      : result.metadata.cv.usable
      ? "Generated locally. Pixel quality checks passed."
      : "Generated locally, but pixel checks flagged this image. Raw output is retained.";
  const plan = result.metadata.semantic_plan,
    summary = semanticSummary({
      objects: plan.requested_objects,
      relations: plan.requested_relations,
    });
  $("prompt-plan").hidden = false;
  $("prompt-plan").textContent =
    `Plan · ${result.metadata.profile.key} · ${plan.complexity.toFixed(2)} complexity · objects: ${summary.objects} · relations: ${summary.relations}. ${plan.verification}`;
  renderInsights(result);
}

function comparisonCard(title, result) {
  const metadata = result.metadata,
    card = document.createElement("article"),
    image = document.createElement("img"),
    content = document.createElement("div"),
    heading = document.createElement("h3"),
    detail = document.createElement("p"),
    url = URL.createObjectURL(result.enhanced);
  state.comparisonUrls.push(url);
  card.className = "comparison-card";
  image.src = url;
  image.alt = `${title}: ${metadata.prompt}`;
  heading.textContent = title;
  detail.textContent = `${metadata.profile.key} · ${metadata.steps} steps · ${metadata.native_resolution} px · ${formatMs(metadata.cached_generation_ms)} worker runtime · ${metadata.unet_calls} U-Net calls`;
  content.append(heading, detail);
  card.append(image, content);
  return card;
}

function showComparison(reference, optimized) {
  clearComparison();
  const referenceMetadata = reference.metadata,
    optimizedMetadata = optimized.metadata,
    baselineRan = referenceMetadata.profile.key === "baseline",
    ratio = referenceMetadata.cached_generation_ms / optimizedMetadata.cached_generation_ms,
    relative = ratio >= 1
      ? `${ratio.toFixed(2)}× faster`
      : `${(1 / ratio).toFixed(2)}× slower`;
  $("comparison-grid").append(
    comparisonCard("20-step local reference", reference),
    comparisonCard("Selected optimized run", optimized),
  );
  $("comparison-note").textContent = baselineRan
    ? `Matched local comparison. The optimized worker runtime was ${relative} than the reference (${formatMs(referenceMetadata.cached_generation_ms)} → ${formatMs(optimizedMetadata.cached_generation_ms)}). Compare semantic details visually; this page does not claim they are equivalent.`
    : `The requested 20-step reference was reduced to ${referenceMetadata.profile.key} by the local resource policy, so this is not a valid 20-step baseline comparison.`;
  $("comparison").hidden = false;
}

async function revealResult(result) {
  const renderStarted = performance.now();
  showResult(result, result.metadata.submit_to_result_ms);
  await $("image").decode();
  if (innerWidth < 760)
    $("image").scrollIntoView({ behavior: "instant", block: "center" });
  await new Promise((painted) =>
    requestAnimationFrame(() => requestAnimationFrame(painted)),
  );
  result.metadata.submit_to_visible_ms =
    result.metadata.submit_to_result_ms + performance.now() - renderStarted;
  $("metadata").textContent = JSON.stringify(result.metadata, null, 2);
}

function runAttempt(worker, payload) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      worker.terminate();
      reject({
        message: "Local generation exceeded five minutes.",
        code: "resource",
        profile: { key: payload.mode },
      });
    }, 300000);
    const finish = (callback, value) => {
      clearTimeout(timeout);
      callback(value);
    };
    worker.onerror = (event) =>
      finish(reject, {
        message:
          event.message ||
          "The local worker stopped, possibly due to memory pressure.",
        code: "resource",
        profile: { key: payload.mode },
      });
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        $("stage").textContent = data.message;
        $("generation-progress").value = data.progress;
        return;
      }
      if (data.type === "process") {
        updateProcessVisualizer(data);
        return;
      }
      if (data.type === "process-preview") {
        addProcessPreview(data);
        return;
      }
      if (data.type === "error") return finish(reject, data);
      if (data.type === "result") finish(resolve, data);
    };
    worker.postMessage(payload);
  });
}
async function generate({
  prompt,
  seed,
  mode,
  omit = [],
  keepLoaded = $("keep-loaded").checked,
  structuredPrompt = $("structured-prompt").checked,
  cvStudy = false,
  processPreview = $("process-preview").checked,
  present = true,
  verify = $("verify-prompt").checked,
  qualityHints = $("quality-hints").checked,
  budgetMs,
}) {
  if (!state.ready || state.busy)
    throw Error(
      state.busy
        ? "A local generation is already running."
        : "Prepare the models first.",
    );
  if (!prompt.trim()) throw Error("Enter a description for your image.");
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
    throw Error("Use a whole-number seed from 0 to 4294967295.");
  state.busy = true;
  $("error").hidden = true;
  startProcessVisualizer(processPreview);
  syncControls();
  $("generation-progress").hidden = false;
  clearTimeout(state.idleTimer);
  if (!keepLoaded && state.worker) {
    state.worker.terminate();
    state.worker = null;
  }
  const started = performance.now(),
    semantic = analyzePrompt(prompt),
    failures = [];
  const initialSummary = semanticSummary(semantic);
  $("prompt-plan").hidden = false;
  $("prompt-plan").textContent =
    `Plan · analysing ${semantic.complexity.toFixed(2)} complexity · objects: ${initialSummary.objects} · relations: ${initialSummary.relations}.`;
  let attempt = 1,
    attemptMode = mode,
    fallbackFrom = null,
    activeKeepLoaded = keepLoaded,
    worker = state.worker;
  let success = false;
  // One locally generated candidate, with the existing resource fallback.
  async function candidateRun(candidateSeedValue, index) {
    for (;;) {
      worker ||= new Worker("/worker.js", { type: "module" });
      state.worker = worker;
      try {
        const data = await runAttempt(worker, {
          prompt,
          seed: candidateSeedValue,
          mode: attemptMode,
          omit,
          keepLoaded: activeKeepLoaded,
          structuredPrompt,
          cvStudy: cvStudy && index === 0,
          processPreview: processPreview && index === 0,
          attempt,
          fallbackFrom,
          verify,
          verifyProfiles: VERIFY_PROFILES,
          qualityHints,
        });
        data.metadata.recovery = { used: failures.length > 0, failures: [...failures] };
        return data;
      } catch (error) {
        const failed = error.profile?.key || attemptMode,
          fallback = error.code === "resource" ? nextFallback(failed) : null;
        worker.terminate();
        if (state.worker === worker) state.worker = null;
        worker = null;
        if (!fallback) throw Error(error.message || String(error));
        failures.push({ attempt, profile: failed, message: error.message || String(error) });
        attempt += 1;
        attemptMode = fallback;
        fallbackFrom = failed;
        activeKeepLoaded = false;
        $("stage").textContent = `Resource limit detected. Retrying locally with ${fallback} mode…`;
        $("generation-progress").value = 0;
      }
    }
  }
  try {
    const candidates = [],
      timeline = [];
    let policy = null,
      decision = null;
    for (let index = 0; ; index++) {
      const candidateStarted = performance.now(),
        value = candidateSeed(seed, index);
      if (index > 0)
        $("stage").textContent = `${decision.reason} Candidate ${index + 1}, seed ${value}…`;
      const data = await candidateRun(value, index);
      // Later candidates reuse the profile that actually ran, so Auto cannot
      // drift between candidates.
      attemptMode = data.metadata.profile.key;
      policy ||= selectionPolicy(data.metadata.profile.key, { verifyRequested: verify, budgetMs });
      candidates.push(data);
      const summary = data.metadata.verifier?.summary;
      timeline.push({
        index,
        seed: value,
        profile: data.metadata.profile.key,
        unet_calls: data.metadata.unet_calls,
        elapsed_ms: Math.round(performance.now() - candidateStarted),
        verifier: data.metadata.verifier
          ? data.metadata.verifier.unavailable
            ? "unavailable"
            : `${summary.passed} passed, ${summary.failed} failed, ${summary.not_verifiable} not verifiable`
          : "not run in this mode",
        score: Number(candidateScore(data.metadata).toFixed(3)),
      });
      decision = shouldContinue(
        policy,
        candidates.map((c) => c.metadata),
        performance.now() - started,
      );
      if (!decision.next) break;
    }
    const chosen = pickBest(candidates.map((c) => c.metadata)),
      data = candidates[chosen];
    timeline[chosen].selected = true;
    data.metadata.selection = {
      policy,
      candidates: timeline,
      selected_index: chosen,
      stop_reason: decision.reason,
      best_effort: /budget/i.test(decision.reason),
      note: "Candidate selection used the on-device verifier. Report final accuracy from a separate offline evaluator.",
    };
    data.metadata.submit_to_result_ms = performance.now() - started;
    if (present) await revealResult(data);
    success = true;
    return data;
  } catch (error) {
    fail(error.message);
    throw error;
  } finally {
    if (!activeKeepLoaded || !success) {
      worker?.terminate();
      if (state.worker === worker) state.worker = null;
    } else
      state.idleTimer = setTimeout(() => {
        worker.terminate();
        if (state.worker === worker) state.worker = null;
      }, 120000);
    state.busy = false;
    syncControls();
  }
}
addEventListener("pagehide", () => {
  state.worker?.terminate();
  revoke(Object.values(state.urls));
  revoke(state.studyUrls);
  revoke(state.processUrls);
  revoke(state.comparisonUrls);
});
$("form").onsubmit = (event) => {
  event.preventDefault();
  clearComparison();
  generate({
    prompt: $("prompt").value.trim(),
    seed: Number($("seed").value),
    mode: $("mode").value,
    cvStudy: $("cv-study").checked,
    processPreview: $("process-preview").checked,
  }).catch(() => {});
};
$("compare").onclick = async () => {
  const prompt = $("prompt").value.trim(),
    seed = Number($("seed").value),
    optimizedMode = $("mode").value,
    structuredPrompt = $("structured-prompt").checked;
  clearComparison();
  state.comparing = true;
  syncControls();
  $("error").hidden = true;
  setComparisonStatus("Running the local 20-step reference with this prompt and seed…");
  try {
    const reference = await generate({
      prompt,
      seed,
      mode: "baseline",
      keepLoaded: false,
      structuredPrompt,
      cvStudy: false,
      processPreview: false,
      present: false,
      verify: false,
    });
    setComparisonStatus("Reference complete. Running the selected optimized profile…");
    const optimized = await generate({
      prompt,
      seed,
      mode: optimizedMode,
      keepLoaded: false,
      structuredPrompt,
      cvStudy: false,
      processPreview: false,
      present: false,
      verify: false,
    });
    await revealResult(optimized);
    showComparison(reference, optimized);
    setComparisonStatus("Comparison complete. Both images were generated locally from the same prompt and seed.");
  } catch (error) {
    fail(error.message || String(error));
    setComparisonStatus("Comparison stopped before both local runs completed.");
  } finally {
    state.comparing = false;
    syncControls();
  }
};
for (const button of document.querySelectorAll("[data-image]"))
  button.onclick = () => {
    const kind = button.dataset.image;
    $("image").src = state.urls[kind];
    $("save").href = state.urls[kind];
    for (const other of document.querySelectorAll("[data-image]"))
      other.setAttribute("aria-pressed", String(other === button));
  };
// Reproducible diagnostics operate on this app only; no remote inference path exists.
window.__p3 = {
  generate,
  prepare: () =>
    new Promise((resolve, reject) => {
      $("prepare").click();
      const timer = setInterval(() => {
        if (!state.controller) {
          clearInterval(timer);
          state.ready ? resolve(true) : reject(Error($("error").textContent));
        }
      }, 200);
    }),
  state,
};
