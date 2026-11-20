import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { getAsset, isVerifierCached, prepareModels } from "../src/cache.js";

const digest = (data) => createHash("sha256").update(data).digest("hex");

test("Prepare repairs the verifier beside cached models and checks it before offline reuse", async () => {
  const base = "https://example.test";
  const cache = new Map();
  const url = (path) => new URL(path, base).href;
  globalThis.location = { href: base };
  globalThis.caches = { open: async () => ({
    match: async (path) => cache.get(url(path))?.clone(),
    put: async (path, response) => cache.set(url(path), response),
  }) };

  const model = process.env.P3_VERIFIER_MODEL;
  const bytes = model ? await readFile(model) : Buffer.from("verifier fixture");
  const verifier = model
    ? JSON.parse(await readFile(new URL("../verifier-manifest.json", import.meta.url)))
    : { path: "verifier.onnx", sha256: digest(bytes), bytes: bytes.length };
  const large = { path: "weights.bin", sha256: "cached-weight-digest", bytes: 900_000_000 };
  const manifest = JSON.stringify({ components: { text: { files: [large] }, unet: { files: [] }, decoder: { files: [] } } });
  cache.set(url("/models/weights.bin"), new Response("cached", { headers: { "x-model-sha256": large.sha256 } }));
  Object.defineProperty(navigator, "storage", { configurable: true, value: {
    estimate: async () => ({ quota: 1_000_000_000, usage: 990_000_000 }),
  } });
  let detector = new Response(Buffer.alloc(bytes.length));
  let downloads = 0;
  globalThis.fetch = async (request) => {
    const path = new URL(request, base).pathname;
    if (path === "/model-index.json") return Response.json({ manifest_bytes: Buffer.byteLength(manifest), manifest_sha256: digest(manifest) });
    if (path === "/verifier-index.json") return Response.json(verifier);
    if (path === "/models/manifest.json") return new Response(manifest);
    if (path === "/models/" + verifier.path) {
      downloads++;
      return detector.clone();
    }
    assert.notEqual(path, "/models/weights.bin", "Existing generation weights must be reused");
    return new Response("tokenizer fixture");
  };
  const prepare = () => prepareModels(() => {});
  let result = await prepare();
  assert.equal(result.verifier_ready, false);
  assert.match(result.verifier_error, /checksum mismatch/);
  assert.equal(await isVerifierCached(), false);
  detector = new Response(bytes);
  result = await prepare();
  assert.equal(result.verifier_ready, true);
  assert.equal(await isVerifierCached(), true);
  assert.equal((await prepare()).verifier_ready, true);
  assert.equal(downloads, 2);
  globalThis.fetch = () => { throw Error("offline"); };
  const cached = await getAsset(verifier.path, { sha256: verifier.sha256 });
  assert.equal(digest(Buffer.from(await cached.arrayBuffer())), verifier.sha256);
});
