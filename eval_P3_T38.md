1RUA24CSE0039,1RUA24CSE0045,1RUA24CSE0060,1RUA24CSE0062

# P3 Evaluation T38

> **Historical v3 evaluation:** this table predates the current prompt-aware planner and automatic local fallback. It must be regenerated on Android before it is used to evaluate the current source.

**O(n) Eyes**

## On-device browser pipeline

[Live hosted app](https://on-eyes-device-lab-t38.brawnymoss1024.chatgpt.site) · [Source](https://github.com/solaxien/O-n-Eyes) · [Full browser evaluation](docs/BROWSER_ON_DEVICE.md)

On the physical Poco F6, all neural inference and classical CV ran in Chrome on the phone. Each row below uses the same 16 prompts/seeds at native 256 px, displayed at 512 px. Timings are submit-to-visible after model initialization, with resident sessions. The metric order follows the supplied problem statement: CLIP, FID, latency, peak memory, model size, then quality versus computational cost.

| Configuration | CLIP similarity ↑ | FID | Image generation latency p50 / p95 (s) | Observed peak Chrome PSS (MiB) | Model size (MiB) |
| --- | ---: | --- | ---: | ---: | ---: |
| Browser FP16 · 4 steps / 8 denoiser calls | 0.2853 | N/A, N=16 | 5.877 / 6.237 | 1,553 | 854.857 |
| Browser FP16 · 6 steps / 12 calls | 0.3001 | N/A, N=16 | 7.744 / 7.848 | 2,372 | 854.857 |
| Browser FP16 · 12 steps / 24 calls | 0.3053 | N/A, N=16 | 14.648 / 14.900 | 1,518 | 854.857 |

## Quality versus computational cost

Six steps reduced median time by about 47% versus twelve, with CLIP lower by 0.00515. Exact composition was 0/16 in every row. Full CV slightly reduced CLIP versus raw output and did not repair missing objects or relations. FID is not validly estimated from this small composition diagnostic. A separate browser distribution study and unoptimized browser-model control are still needed.

The six-step run includes initial session creation before its warm-up; the later step studies reused those sessions. Peak PSS covers all Chrome processes, including other tabs, and is not exact GPU allocation. The memory columns therefore do not establish a step-count memory advantage. Phone temperature and sequential run order are recorded. The initial six-step generation took 15.82 s. A separate offline reload/fresh-prompt test took 16.54 s with networks off, no ADB reverse path, zero server inference requests and 8,844 GPU dispatches. The initial public model-download network envelope was 771.79 s, separate from generation. These limits preclude an unconditional ten-second claim.

The browser architecture, CV component ablation, failure images, source/image hashes, numerical probes and network evidence are in [the browser report](docs/BROWSER_ON_DEVICE.md) and [machine-readable results](docs/browser/evidence.json). The site is served by static CPU hosting; no T4 or external GPU is involved.

## Additional validation

The visual audit identifies both genuine generation failures and a crop-color scoring disagreement. CLIP is not evidence that all composition clauses are satisfied.

[Full browser evaluation](docs/BROWSER_ON_DEVICE.md)
