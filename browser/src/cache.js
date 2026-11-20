export const MODEL_CACHE = "p3-model-fp16-v1";
export async function getAsset(
  path,
  { download = false, sha256, bytes, progress = () => {}, signal } = {},
) {
  const url = new URL("/models/" + path, globalThis.location.href).href;
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(url);
  if (cached && (!sha256 || cached.headers.get("x-model-sha256") === sha256))
    return cached;
  if (!download)
    throw Error(
      "A required model file is missing. Use Prepare to download or repair the offline cache.",
    );
  const response = await fetch(url, { signal, cache: "no-store" });
  if (!response.ok)
    throw Error(
      `Model download failed (${response.status}). Retry to continue from completed files.`,
    );
  const reader = response.body.getReader(),
    chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    progress(received, bytes || Number(response.headers.get("content-length")));
  }
  if (bytes !== undefined && received !== bytes)
    throw Error("Incomplete model file. Retry to download this file again.");
  const data = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.length;
  }
  if (sha256) {
    const actual = [
      ...new Uint8Array(await crypto.subtle.digest("SHA-256", data)),
    ]
      .map((x) => x.toString(16).padStart(2, "0"))
      .join("");
    if (actual !== sha256)
      throw Error("Model checksum mismatch. Retry to repair the download.");
  }
  const result = new Response(data, {
    headers: {
      "Content-Type": path.endsWith(".json")
        ? "application/json"
        : "application/octet-stream",
      "x-model-sha256": sha256 || "metadata",
    },
  });
  try {
    await cache.put(url, result.clone());
  } catch (error) {
    if (error.name === "QuotaExceededError")
      throw Error(
        "Browser storage is full. Use a regular browser tab, free at least 1.2 GB, and retry. Completed model files are retained.",
      );
    throw error;
  }
  return result;
}
export async function prepareModels(progress, signal) {
  const index = await fetch("/model-index.json").then((r) => r.json());
  const manifest = await (
    await getAsset("manifest.json", {
      download: true,
      signal,
      sha256: index.manifest_sha256,
      bytes: index.manifest_bytes,
    })
  ).json();
  if (Object.keys(manifest.components).length !== 3)
    throw Error("The browser export is not yet complete.");
  const files = Object.values(manifest.components).flatMap((x) => x.files);
  const total = files.reduce((s, x) => s + x.bytes, 0);
  let finished = 0;
  const estimate = await navigator.storage.estimate();
  if (estimate.quota - estimate.usage < total * 1.2)
    throw Error(
      "Not enough browser storage for the model. Free space and retry.",
    );
  for (const file of files) {
    let failure;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await getAsset(file.path, {
          ...file,
          download: true,
          signal,
          progress: (n) =>
            progress({ file: file.path, bytes: finished + n, total }),
        });
        failure = null;
        break;
      } catch (error) {
        failure = error;
        if (signal?.aborted) throw error;
      }
    }
    if (failure) throw failure;
    finished += file.bytes;
    progress({ file: file.path, bytes: finished, total });
  }
  for (const path of [
    "vocab.json",
    "merges.txt",
    "schedules.json",
    "tokenizer-tests.json",
  ])
    await getAsset(path, { download: true, signal });
  return manifest;
}
