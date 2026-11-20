import * as ort from "./vendor/ort.webgpu.min.mjs?v=1.31.1";
import { getAsset } from "./cache.js?v=1";
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.logLevel = "warning";
export async function validateComponent(
  name,
  progress = () => {},
  optimization = "all",
  backend = "webgpu",
  noSubgroups = false,
) {
  if (noSubgroups) {
    const adapter = await navigator.gpu.requestAdapter();
    ort.env.webgpu.device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize:
          adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
  }
  const proof = await fetch("/models/" + name + ".json").then((r) => r.json());
  for (const file of proof.files) {
    progress({ stage: "download", file: file.path });
    await getAsset(file.path, {
      ...file,
      download: true,
      progress: (n, total) =>
        progress({ stage: "download", file: file.path, bytes: n, total }),
    });
  }
  const graph = new Uint8Array(
    await (await getAsset(proof.graph)).arrayBuffer(),
  );
  const externalData = [];
  for (const file of proof.files.filter((f) => f.path !== proof.graph))
    externalData.push({
      path: file.path.split("/").at(-1),
      data: await (await getAsset(file.path)).blob(),
    });
  progress({ stage: "create", component: name });
  const started = performance.now();
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: [backend],
    externalData,
    graphOptimizationLevel: optimization,
    enableCpuMemArena: false,
    enableMemPattern: false,
  });
  const loadMs = performance.now() - started,
    feeds = {};
  for (const [key, input] of Object.entries(proof.reference.inputs)) {
    const buffer = await fetch("/models/" + input.path).then((r) =>
      r.arrayBuffer(),
    );
    const type = { float16: "float16", float32: "float32", int32: "int32" }[
      input.dtype
    ];
    const data =
      input.dtype === "float16"
        ? new Uint16Array(buffer)
        : input.dtype === "int32"
          ? new Int32Array(buffer)
          : new Float32Array(buffer);
    feeds[key] = new ort.Tensor(type, data, input.shape);
  }
  progress({ stage: "run", component: name });
  const run = performance.now();
  const outputs = await session.run(feeds);
  const runMs = performance.now() - run;
  const actual = outputs.output.data;
  const expected = new Float32Array(
    await fetch("/models/" + proof.reference.output.path).then((r) =>
      r.arrayBuffer(),
    ),
  );
  let sum = 0,
    ref = 0,
    max = 0,
    bad = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i] - expected[i];
    sum += d * d;
    ref += expected[i] * expected[i];
    max = Math.max(max, Math.abs(d));
    if (!Number.isFinite(actual[i])) bad++;
  }
  const result = {
    name,
    optimization,
    backend,
    noSubgroups,
    userAgent: navigator.userAgent,
    load_ms: loadMs,
    run_ms: runMs,
    shape: outputs.output.dims,
    finite: bad === 0,
    rmse: Math.sqrt(sum / actual.length),
    relative_rmse: Math.sqrt(sum / Math.max(ref, 1e-20)),
    max_absolute_error: max,
    expected_elements: expected.length,
    actual_elements: actual.length,
  };
  if (name === "text") {
    result.token_errors = [0, 1, 2, 10, 76].map((i) => {
      let n = 0;
      for (let j = i * 768; j < (i + 1) * 768; j++)
        n += (actual[j] - expected[j]) ** 2;
      return {
        token: i,
        rmse: Math.sqrt(n / 768),
        expected: Array.from(expected.slice(i * 768, i * 768 + 6)),
        actual: Array.from(actual.slice(i * 768, i * 768 + 6)),
      };
    });
  }
  if (name === "text")
    result.all_token_rmse = Array.from({ length: 77 }, (_, i) => {
      let n = 0;
      for (let j = i * 768; j < (i + 1) * 768; j++)
        n += (actual[j] - expected[j]) ** 2;
      return Math.sqrt(n / 768);
    });
  for (const tensor of Object.values(outputs)) tensor.dispose();
  for (const tensor of Object.values(feeds)) tensor.dispose();
  await session.release();
  return result;
}
