# Android resource strategy and evaluation protocol

P3 Image Lab performs prompt encoding, diffusion, decoding, resize, and classical image processing in Chrome on the Android device. It has no generation server or cloud fallback. The application now makes each resource decision explicit and records it in the result metadata.

## Runtime policy

The browser cannot read free RAM or exact GPU allocations. It receives a coarse memory class (`navigator.deviceMemory`), logical-core count, WebGPU capabilities, and timing from a previous local generation. The planner therefore uses those only as conservative signals:

| Situation | Initial profile | Automatic fallback after a resource failure |
| --- | --- | --- |
| Memory class below 6 GiB or four or fewer logical cores | Preview: 256 px, 4 steps | No additional local profile is safer; return the local error. |
| Normal simple prompt | Eco: 256 px, 6 steps | Preview: 256 px, 4 steps |
| Complex prompt on a capable device | Balanced: 256 px, 12 steps | Eco, then Preview |
| Explicit quality request | Quality: 384 px, 12 steps | Balanced, Eco, then Preview |

The fallback is triggered only by a local resource or numerical failure. It releases the failed worker and starts a new worker with the lower profile. The image prompt, seed, and all failure reasons are retained in the resulting metadata. A missing WebGPU adapter or an unsupported FP16 feature is not treated as a successful fallback; the app clearly reports that the device cannot run this model locally.

`Keep models loaded` remains opt-in because it reduces repeat latency while increasing retained memory. A fallback always disables session retention for its retry.

## Prompt contract

Before generation, the local prompt parser extracts recognized objects, counts, colours, and spatial relations. The complexity score helps choose an initial profile. The optional local conditioning rewrite repeats those clauses in a compact form for the CLIP text encoder, and automatically falls back to the user’s exact wording if the rewritten prompt would exceed the 75-token model limit.

After decode, the app also records sampled global evidence for each requested colour. This is an inexpensive diagnostic only: it cannot associate colour with an object, establish a count, or validate a spatial relation. It must not be reported as semantic correctness.

The web interface exposes an **Optimization Lab** for an interactive, matched demonstration: it generates a 20-step reference and the selected optimized profile from the same prompt and seed, then shows their images, local worker runtimes, steps, resolution, and U-Net call counts. It also has an optional CV-ablation control that renders variants with one post-processing stage omitted. The displayed `pixel-health` score is a no-reference heuristic based on contrast, edge variance, clipping, and detected isolated artifacts; it is explicitly not a realism or semantic score.

This parser is not an image verifier. The metadata names every requested clause and explicitly states that the current model release does not include an on-device object/relation detector. Pixel quality gates and classical CV cannot prove that an apple is green, that a cup exists, or that one object is left of another.

## Required Android measurement runs

Run every profile on the same physical Android device, browser version, prompt manifest, seed, and thermal state. Close other Chrome tabs where possible and report that condition. Preserve a fresh directory for each run.

```powershell
cd browser
npm ci
npm test
npm run build
cd ..

# Reference before reduced sampling: same browser model, 20 steps.
node scripts/benchmark_browser.cjs --origin <https-origin> --tab <tab-id> --out .runs/browser/android-baseline-20 --manifest data/android-composition-v2.jsonl --adb <adb-path> --mode baseline --warmup --python python

# Optimized profiles on the same manifest.
node scripts/benchmark_browser.cjs --origin <https-origin> --tab <tab-id> --out .runs/browser/android-eco-6 --manifest data/android-composition-v2.jsonl --adb <adb-path> --mode eco --warmup --python python
node scripts/benchmark_browser.cjs --origin <https-origin> --tab <tab-id> --out .runs/browser/android-preview-4 --manifest data/android-composition-v2.jsonl --adb <adb-path> --mode preview --warmup --python python
```

The harness records whether local structured prompt conditioning was enabled (the default). Use `--structured-prompt false` for a matched exact-prompt ablation.

After the baseline and optimized runs have completed, create a cost-only comparison artifact. The script rejects mismatched prompts, cached model manifests, and conditioning settings rather than silently comparing unlike runs.

```powershell
python scripts/compare_browser_profiles.py --run .runs/browser/android-baseline-20 --run .runs/browser/android-eco-6 --run .runs/browser/android-preview-4 --out .runs/browser/android-cost-comparison.json
```

For each configuration, record model bytes, cached storage, cold preparation time, p50/p95 submit-to-visible latency, and sampled Chrome package PSS. PSS is an aggregate proxy, not an exact WebGPU allocation. Keep cold model initialization and the first model download separate from resident generation latency.

Score the captured images with a held-out semantic evaluator for object presence, count, colour binding, and relations. Use a different evaluator from any selector that might influence which image is shown. Report CLIP on the same prompt/image pairs and FID or KID only with a sufficiently large matched real-image reference set. The app does not yet bundle that evaluator; its current historical 16-prompt results remain insufficient for a semantic-success claim.

## Acceptance criteria

An Android release may claim the P3 objective only after it shows all of the following on the tested phone:

1. A 20-step browser reference versus 12, 6, and 4 steps, with matched seeds and a stated warm-up protocol.
2. At least two optimizations with measured effects. The current implementation already has FP16 neural layers, TAESD decoding, staged component release, and reduced sampling; a quantized model export needs separate browser validation before it can be claimed.
3. A semantic result table that includes complex prompts and relation accuracy, not CLIP alone.
4. Resource-failure evidence showing the automatic fallback profile actually generated locally.
5. A lower-memory model variant for Android devices unable to cache the current FP16 bundle. The existing bundle is about 855 MiB of weights and has only been demonstrated on an 8 GiB Poco F6; it is not an appropriate claim for entry-level Android phones.
