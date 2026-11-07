1RUA24CSE0039,1RUA24CSE0045,1RUA24CSE0060,1RUA24CSE0062

# P3 Evaluation T38

**O(n) Eyes**

## Current visitor-device browser pipeline

[Live hosted app](https://on-eyes-device-lab-t38.brawnymoss1024.chatgpt.site) · [Source](https://github.com/solaxien/O-n-Eyes) · [Full browser evaluation](docs/BROWSER_ON_DEVICE.md)

On the physical Poco F6, all neural inference and classical CV ran in Chrome on the phone. Each row below uses the same 16 prompts/seeds at native 256 px, displayed at 512 px. Timings are submit-to-visible after model initialization, with resident sessions. The metric order follows the supplied problem statement: CLIP, FID, latency, peak memory, model size, then quality versus computational cost.

| Configuration | CLIP similarity ↑ | FID | Image generation latency p50 / p95 (s) | Observed peak Chrome PSS (MiB) | Model size (MiB) |
| --- | ---: | --- | ---: | ---: | ---: |
| Browser FP16 · 4 steps / 8 denoiser calls | 0.2853 | N/A, N=16 | 5.877 / 6.237 | 1,553 | 854.857 |
| Browser FP16 · 6 steps / 12 calls | 0.3001 | N/A, N=16 | 7.744 / 7.848 | 2,372 | 854.857 |
| Browser FP16 · 12 steps / 24 calls | 0.3053 | N/A, N=16 | 14.648 / 14.900 | 1,518 | 854.857 |

**Quality versus computational cost:** six steps reduced median time by about 47% versus twelve, with CLIP lower by 0.00515. Exact composition was 0/16 in every row. Full CV slightly reduced CLIP versus raw output and did not repair missing objects or relations. FID is not validly estimated from this small composition diagnostic. A separate browser distribution study and unoptimized browser-model control are still needed.

The six-step run includes initial session creation before its warm-up; the later step studies reused those sessions. Peak PSS covers all Chrome processes, including other tabs, and is not exact GPU allocation. The memory columns therefore do not establish a step-count memory advantage. Phone temperature and sequential run order are recorded. The initial six-step generation took 15.82 s. A separate offline reload/fresh-prompt test took 16.54 s with networks off, no ADB reverse path, zero server inference requests and 8,844 GPU dispatches. The initial public model-download network envelope was 771.79 s, separate from generation. These limits preclude an unconditional ten-second claim.

The browser architecture, CV component ablation, failure images, source/image hashes, numerical probes and network evidence are in [the browser report](docs/BROWSER_ON_DEVICE.md) and [machine-readable results](docs/browser/evidence.json). The site is served by static CPU hosting; no T4 or external GPU is involved. The earlier laptop inference endpoint is not the current app.

## Historical host evaluation

All six rows use the same 300 COCO val2017 captions and seeds at native 512 × 512 on an NVIDIA RTX 4050 Laptop GPU. Latency is warm inference through raw decode. Peak memory lists PyTorch GPU allocations and host process RSS, both in MiB. Model size counts the offline weights and configuration used in each row. The public worker also runs the safety checker.

| Configuration | CLIP similarity ↑ | FID ↓ (N = 300) | Image generation latency p50 / p95 (s) | Peak memory GPU / RSS (MiB) | Model size (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 · SD v1.4 FP32 · PNDM 50 requested / 51 calls | 0.3148 | 130.00 | 19.957 / 24.421 | 4,872 / 1,437 | 4,069 |
| 2 · SD v1.4 FP16 · PNDM 50 / 51 | 0.3148 | 129.62 | 5.993 / 6.152 | 2,677 / 1,814 | 2,035 |
| 3 · Tiny FP16 · PNDM 25 / 26 | 0.3023 | 130.89 | 1.865 / 1.927 | 1,653 / 1,795 | 1,013 |
| 4 · Tiny FP16 · DPM-Solver++ 20 / 20 | 0.3033 | 130.63 | 1.464 / 1.511 | 1,653 / 1,805 | 1,013 |
| 5 · Row 4 with TAESD | 0.3023 | 132.16 | 1.279 / 1.298 | 1,042 / 1,820 | 862 |
| 6 · Row 5 with COCO LoRA | 0.3028 | 129.16 | 1.276 / 1.291 | 1,042 / 1,840 | 864 |

## Quality versus computational cost

The tuned model was **15.6× faster** at the median than the FP32 reference. Peak GPU allocations fell **79%** and the active model bundle shrank **79%**. Host process RSS rose from 1,437 to 1,840 MiB. CLIP cosine decreased by 0.0119 and FID decreased by 0.84 within this cohort. The paired OpenCV pass took 29 ms at the median, with CLIP 0.3026 and FID 129.13 across the same 300 decodes.

On the separate 16-prompt set, Balanced rendered in 0.658 seconds at the median with CLIP 0.3387. Low resource took 0.323 seconds with CLIP 0.3234. Both profiles scored 0/16 on exact composition.

The public demo displayed a 512 px image in **2.47 seconds** in Chrome on a physical Poco F6. The laptop GPU performed inference for that request. In the final six browser runs, five displayed in 1.00–2.49 seconds and one took 11.33 seconds with 2.29 seconds inside the worker.

The separate ARM CPU comparison ran 12 native 256 px renders on the Poco F6. SD v1.4 FP16 at 25 steps took a median 218.6 seconds and 2,564 MiB peak process RSS. Tiny Q8_0 at six steps took 37.5 seconds and 761 MiB. Its bundle was 824 MiB including TAESD. The two-image prompt pair exposes substantial errors in object count, color, and position. The tested Adreno paths produced unusable outputs.

[Full ablation and methods](docs/ABLATION.md) · [Failure cases](docs/FAILURES.md) · [Fixed validation manifest](data/coco-val2017.jsonl)

## Implementation review and scope correction

The tables above remain the original RTX 4050 and two-prompt phone experiments. Their legacy runner did not fingerprint every runtime input, so they cannot be resumed under the revised protocol or treated as newly verified provenance. They are preserved with that limitation.

The original native exports contained the **base model without the trained LoRA**. Native phone timings exclude laptop-side resizing, QA, optional CV and scoring. The revised exporter now fuses and verifies the adapter, with separate device validation. New host and expanded phone measurements, scorer diagnostics, recovery tests and deployment limitations are reported in [review results](docs/REVIEW_RESULTS.md). They do not replace the original 300-image COCO scores.

The visual audit identifies both genuine generation failures and a crop-color scoring disagreement. CLIP is not evidence that all composition clauses are satisfied. [Sixteen-frame audit](docs/SEMANTIC_AUDIT.md)

No NVIDIA T4 or external GPU was used for this review, as requested by the user. Permanent public hosting and Docker runtime validation remain pending. The earlier temporary URL is not a guarantee of current availability.

The earlier temporary URL passed six public desktop/phone-width requests in 0.76–2.28 seconds on 7 November. A separate Chrome check on the physical Poco F6 displayed a 512 px image in 2.42 seconds with no page overflow. Inference ran on the laptop GPU. Those historical results are preserved; the temporary GPU-backed service has been replaced by the visitor-device browser app linked above.
