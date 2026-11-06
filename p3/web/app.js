"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("generation-form");
const generate = byId("generate");
const output = byId("result-image");
const stage = byId("stage");
const meter = byId("progress");
const error = byId("error");
let ready = false;
let busy = false;
let result = null;
let evidenceLoaded = false;

const seconds = (ms) => Number.isFinite(ms) ? (ms / 1000).toFixed(2) + " s" : "—";
const milliseconds = (ms) => Number.isFinite(ms) ? ms.toFixed(1) + " ms" : "—";
const memory = (bytes) => Number.isFinite(bytes) ? (bytes / 2 ** 20).toFixed(0) + " MiB" : "—";
const fraction = (value) => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";
const score = (value) => Number.isFinite(value) ? value.toFixed(4) : "—";
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function json(url, options) {
  const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(12000), ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Request failed (" + response.status + ").");
  }
  return response.json();
}

async function checkWorker() {
  const health = await json("/api/health").catch(() => ({ status: "failed" }));
  ready = health.status === "ready";
  byId("worker").dataset.state = health.status;
  byId("worker-label").textContent = ready
    ? (health.backend === "cuda" ? "GPU ready" : "CPU fallback")
    : (health.status === "loading" ? "Loading model" : "Worker unavailable");
  byId("worker-note").textContent = ready
    ? "Inference worker · " + health.hardware + (health.backend === "cpu" ? " · longer runs expected" : "")
    : (health.status === "loading" ? "The worker is loading the checkpoint." : "Ask the host to check the worker logs.");
  generate.disabled = busy || !ready;
  setTimeout(checkWorker, ready ? 8000 : 1500);
}

for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => {
    byId("prompt").value = button.dataset.example;
    byId("prompt").focus();
  });
}

async function showImage(kind) {
  output.src = "data:image/webp;base64," + result[kind];
  output.alt = kind === "raw" ? "Raw generated image" : "Generated image after quality checks";
  await output.decode();
  for (const button of document.querySelectorAll("[data-image]")) {
    button.setAttribute("aria-pressed", String(button.dataset.image === kind));
  }
  byId("download").href = output.src;
  byId("download").download = "p3-" + result.metadata.seed + "-" + kind + ".webp";
}

for (const button of document.querySelectorAll("[data-image]")) {
  button.addEventListener("click", () => showImage(button.dataset.image).catch(reportError));
}

function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content !== undefined) element.textContent = content;
  if (className) element.className = className;
  return element;
}

function renderDetails(meta) {
  const timings = meta.stages_ms;
  const changes = meta.cv.applied.join(", ") || "No corrections triggered";
  const details = {
    Model: meta.model,
    Decoder: meta.decoder === "taesd" ? "TAESD" : "Checkpoint VAE",
    Adapter: meta.adapter || "No LoRA loaded",
    Scheduler: meta.scheduler + " · " + meta.requested_steps + " requested steps",
    Backend: meta.hardware + " · " + meta.precision.toUpperCase(),
    Profile: meta.profile + " · " + meta.selection_reason,
    Worker: seconds(meta.server_ms) + " including " + seconds(meta.queue_ms) + " queue time",
    Denoising: seconds(timings.denoise),
    "Decode and safety": seconds(timings.decode_and_safety),
    "Vision and resizing": seconds(timings.cv_and_resize),
    "CV stages": changes,
    "Suspected speckles": fraction(meta.cv.artifact_fraction),
    "Peak GPU allocations": memory(meta.peak_vram_bytes),
  };
  const list = byId("run-details");
  list.replaceChildren();
  for (const [key, value] of Object.entries(details)) {
    list.append(node("dt", key), node("dd", value));
  }
}

function reportError(err) {
  error.textContent = err.message || "Generation failed. Try again.";
  error.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || !ready) return;
  const prompt = byId("prompt").value.trim();
  if (prompt.length < 3) {
    reportError(new Error("Enter a prompt with at least three characters."));
    return;
  }
  const seed = byId("seed").value;
  const started = performance.now();
  busy = generate.disabled = true;
  byId("generate-label").textContent = "Generating image…";
  error.hidden = output.hidden = byId("result-details").hidden = byId("image-toggle").hidden = true;
  byId("placeholder").hidden = meter.hidden = false;
  stage.textContent = "Submitting prompt";
  meter.value = 4;
  try {
    const ticket = await json("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode: byId("mode").value, seed: seed === "" ? null : Number(seed) }),
    });
    while (true) {
      const job = await json("/api/jobs/" + encodeURIComponent(ticket.id));
      if (job.status === "failed") throw new Error(job.error);
      if (job.status === "complete") {
        result = job.result;
        break;
      }
      const label = job.stage === "Denoising" ? "Denoising " + job.step + " / " + job.total : job.stage;
      if (stage.textContent !== label) stage.textContent = label;
      meter.value = job.total ? 12 + 75 * job.step / job.total : 6;
      if (performance.now() - started > 610000) throw new Error("This run timed out.");
      await pause(240);
    }
    await showImage("enhanced");
    output.hidden = false;
    byId("placeholder").hidden = true;
    await new Promise((resolve) => requestAnimationFrame(resolve));
    byId("elapsed").textContent = seconds(performance.now() - started);
    byId("steps").textContent = result.metadata.steps + " calls";
    byId("size").textContent = result.metadata.native_size + " × " + result.metadata.native_size;
    byId("seed-result").textContent = "Seed " + result.metadata.seed;
    renderDetails(result.metadata);
    byId("result-details").hidden = byId("image-toggle").hidden = false;
    if (matchMedia("(max-width: 720px)").matches) output.scrollIntoView({ block: "center" });
  } catch (err) {
    reportError(err);
    stage.textContent = "Ready for another prompt";
    meter.hidden = true;
  } finally {
    busy = false;
    generate.disabled = !ready;
    byId("generate-label").textContent = "Generate image";
  }
});

function resultTable(group) {
  const section = node("section", undefined, "panel");
  section.append(node("h2", group.title), node("p", group.description));
  const scroll = node("div", undefined, "scroll");
  scroll.tabIndex = 0;
  scroll.setAttribute("aria-label", group.title + " table");
  const table = node("table");
  const columns = [
    ["Variant", "label"], ["Runs", "samples"], ["Denoiser calls", "denoiser_calls"],
    ["Steps", "steps"], ["Warm p50", "latency_p50_ms", seconds], ["Warm p95", "latency_p95_ms", seconds],
    ["CV p50", "cv_p50_ms", milliseconds], ["Peak RAM", "peak_rss_bytes", memory],
    ["Peak GPU", "peak_vram_bytes", memory], ["Bundle", "model_bytes", memory],
    ["CLIP cosine", "clip_mean", score], ["Composition", "composition_pass", fraction],
    ["FID (N)", "fid", (value, run) => value.toFixed(2) + " (" + run.fid_n + ")"],
  ].filter(([, key]) => group.rows.some((run) => run[key] != null));
  const head = table.createTHead().insertRow();
  for (const [title] of columns) {
    const cell = node("th", title);
    cell.scope = "col";
    head.append(cell);
  }
  const tbody = table.createTBody();
  for (const run of group.rows) {
    const row = tbody.insertRow();
    for (const [, key, format] of columns) {
      const value = run[key];
      row.insertCell().textContent = value == null ? "—" : format ? format(value, run) : value;
    }
  }
  scroll.append(table);
  section.append(scroll);
  return section;
}

function comparison(example) {
  const section = node("section", undefined, "panel");
  section.append(node("h2", example.title), node("p", example.prompt + " · Seed " + example.seed));
  const pairs = node("div", undefined, "pairs");
  for (const frame of example.images) {
    const figure = node("figure");
    const img = node("img");
    img.loading = "lazy";
    img.src = frame.src;
    img.alt = frame.label;
    figure.append(img, node("figcaption", frame.label));
    pairs.append(figure);
  }
  section.append(pairs);
  if (example.note) section.append(node("p", example.note));
  return section;
}

async function loadEvidence() {
  if (evidenceLoaded) return;
  const status = byId("evidence-status");
  try {
    const report = await json("/evidence.json");
    const view = byId("evidence");
    view.replaceChildren();
    if (report.highlights?.length) {
      const summary = node("dl", undefined, "panel readouts benchmark-summary");
      for (const { label, value } of report.highlights) {
        const readout = node("div");
        readout.append(node("dt", label), node("dd", value));
        summary.append(readout);
      }
      view.append(summary);
    }
    for (const group of report.groups) view.append(resultTable(group));
    for (const example of report.comparisons) view.append(comparison(example));
    status.textContent = report.note || "";
    status.hidden = !report.note;
    evidenceLoaded = true;
  } catch {
    status.textContent = "Benchmark results could not load. Refresh the page to try again.";
  }
}

for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => {
    const benchmarks = button.dataset.view === "benchmarks";
    for (const tab of document.querySelectorAll("[data-view]")) {
      tab.setAttribute("aria-pressed", String(tab === button));
    }
    byId("generate-view").hidden = benchmarks;
    byId("benchmarks-view").hidden = !benchmarks;
    if (benchmarks) loadEvidence();
  });
}

checkWorker();
