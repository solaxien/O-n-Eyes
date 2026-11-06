# P3 Ablation Study

## Laptop GPU study

All six rows generated the same 300 COCO val2017 captions at native 512 × 512 on an RTX 4050 Laptop GPU. The same seed gives the same FP32 initial noise before casting to each inference dtype. Each row had a separate warm-up. Timings end at the raw decode and omit the public safety checker. CLIP is the mean paired ViT-B/32 cosine. FID uses canonical Inception-v3 features and 300 matched real references center-cropped to 512 px. The small, fixed cohort limits comparisons to these runs. [Validation manifest](../data/coco-val2017.jsonl) · [CLIP model](https://huggingface.co/openai/clip-vit-base-patch32) · [FID implementation](https://github.com/toshas/torch-fidelity)

| Row | Configuration | Actual denoiser calls | CLIP ↑ | FID ↓ (300) | Warm p50 / p95 (s) | Peak GPU (MiB) | Model (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | SD v1.4 FP32, PNDM 50 requested | 51 | 0.3148 | 130.00 | 19.957 / 24.421 | 4,872 | 4,069 |
| 2 | SD v1.4 FP16, PNDM 50 requested | 51 | 0.3148 | 129.62 | 5.993 / 6.152 | 2,677 | 2,035 |
| 3 | BK-SDM-Tiny-2M FP16, PNDM 25 requested | 26 | 0.3023 | 130.89 | 1.865 / 1.927 | 1,653 | 1,013 |
| 4 | Tiny FP16, DPM-Solver++ 20 | 20 | 0.3033 | 130.63 | 1.464 / 1.511 | 1,653 | 1,013 |
| 5 | Row 4 with TAESD | 20 | 0.3023 | 132.16 | 1.279 / 1.298 | 1,042 | 862 |
| 6 | Row 5 with COCO LoRA | 20 | 0.3028 | 129.16 | 1.276 / 1.291 | 1,042 | 864 |

- **FP16** reduced the SD v1.4 median by 70% and roughly halved the bundle. Mean CLIP changed by less than 0.0001.
- **Tiny plus fewer calls** brought the median down from 5.99 to 1.86 seconds and reduced CLIP by 0.0125. This transition changes architecture and sampling together. The same-step phone comparison below isolates the architecture change on that backend.
- **DPM-Solver++ at 20 steps** reduced the Tiny median by a further 21%. Mean CLIP moved from 0.3023 to 0.3033, and FID moved from 130.89 to 130.63. This row changes both the scheduler and step count.
- **TAESD** cut median decode time from 182.8 to 22.1 ms and reduced peak GPU allocations from 1,653 to 1,042 MiB. FID rose by 1.53 in this cohort.
- **COCO LoRA** lowered FID from 132.16 to 129.16 with a CLIP increase of 0.0005 and similar warm latency. The rank-4 adapter used 1,000 train2017 images and 250 updates. [Training record](training.json)

The Windows FP32 reference completed in two timed segments after a CUDA interruption. The first 207 renders had a median of 19.897 seconds and the last 93 had a median of 23.857 seconds. The restarted process loaded one tensor at a time onto CUDA and reproduced one saved image pixel for pixel. The reported p50 and p95 combine both segments. The tuned configuration uses 79% less peak GPU memory and 79% less model storage than the FP32 reference. Its process RSS is 1,840 MiB versus 1,437 MiB for FP32. [System design](DESIGN.md) · [Evaluation](../eval_P3_T38.md)

## Paired classical vision

Each row starts from the same 300 raw row 6 decodes. The CV timing is additional CPU processing. The images already have the 512 px delivery size, so the resize comparison uses the lower-resolution profile in the composition study.

| CV variant | Added p50 (ms) | CLIP ↑ | FID ↓ (300) |
| --- | ---: | ---: | ---: |
| Raw decode | 0.09 | 0.30281 | 129.16 |
| Full pass | 28.97 | 0.30263 | 129.13 |
| Omit color and contrast | 22.91 | 0.30262 | 129.10 |
| Omit edge-aware filtering | 28.06 | 0.30281 | 129.15 |
| Omit quality gate | 22.83 | 0.30263 | 129.13 |
| Omit speckle repair | 24.18 | 0.30263 | 129.14 |

The full pass repaired speckles in 295 decodes, denoised 167, adjusted chroma in 52, and adjusted local contrast in 8. Median Laplacian variance retained 95.4% of its starting value in the full pass and 99.7% with edge filtering omitted. CLIP and FID changes were small. The quality gate kept the same pixels for all 300 valid decodes. In a synthetic 128 × 96 gradient with five injected bright pixels, repair lowered their mean RGB error from 138.4 to 1.0 levels. The gate flagged a flat test image, and resizing the gradient to a 512 px long edge preserved its 4:3 aspect ratio.

## Complex prompts and adaptive profiles

This separate 16-prompt cohort uses fixed objects, colors, relative positions, and seeds. The composition check uses YOLOS Tiny at 0.55 confidence for counts, CLIP on detected crops for colors, and a 5% margin for relative object centers. A detector miss can fail several dependent clauses. An exact pass requires all checked clauses for the image to pass. [Prompt manifest](../data/composition.jsonl) · [Qualitative checks](FAILURES.md)

| Configuration | Median runtime | CLIP ↑ | Count checks | Color checks | Position checks | Exact passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SD v1.4 FP16, PNDM 50 requested | 6.939 s | 0.3331 | 13/36 | 9/31 | 2/17 | 0/16 |
| Tiny 20, raw | 1.484 s | 0.3341 | 9/36 | 2/31 | 1/17 | 0/16 |
| Tiny 20, same decodes with CV | 15.35 ms added | 0.3345 | 8/36 | 1/31 | 1/17 | 0/16 |
| Balanced 8, 512 px with CV | 0.658 s | 0.3387 | 10/36 | 2/31 | 1/17 | 0/16 |
| Low resource 6, 384 px with CV and bicubic upscale | 0.323 s | 0.3234 | 6/36 | 3/31 | 1/17 | 0/16 |
| Same low-resource decodes with bilinear upscale | 9.83 ms for CV and resize | 0.3220 | 6/36 | 3/31 | 1/17 | 0/16 |

At 512 px, Balanced reduced denoising from 20 to eight calls and generated in under 0.7 seconds at the median, including OpenCV. CLIP increased on this set and the exact composition result stayed at zero. The six-step, lower-resolution mode halved the Balanced median while losing four count checks and 0.0153 CLIP cosine. The two resize filters changed CLIP by 0.0014 on identical decodes and returned the same composition counts.

## Poco F6 on-device study

All rows ran directly on Android 16 using four ARM CPU threads, TAESD, DPM++ 2M, the same two prompts and seeds, and 256 px native resolution. Bundle size includes the external TAESD weights. The runner measures inference through decode and reports process high-water RSS across warm-up and the timed renders. The laptop checks and resizes the returned pixels after timing. [Android reproduction steps](../android/README.md)

| Model and storage | Steps | Warm p50 (s) | Peak RSS (MiB) | Bundle (MiB) | CLIP ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SD v1.4 FP16 | 25 | 218.58 | 2,564 | 2,043 | 0.2575 |
| SD v1.4 FP16 | 8 | 74.36 | 2,570 | 2,043 | 0.2508 |
| Tiny FP16 | 25 | 134.36 | 965 | 1,021 | 0.3186 |
| Tiny FP16 | 8 | 43.12 | 965 | 1,021 | 0.3155 |
| Tiny FP16 | 6 | 36.47 | 964 | 1,021 | 0.2654 |
| Tiny Q8_0 | 6 | 37.54 | 761 | 824 | 0.2805 |

At 25 steps, Tiny reduced median latency by 39% and peak RSS by 62%. With Tiny held fixed, eight steps cut the 25-step median by 68%. Reducing to six steps further lowered latency and reduced mean CLIP from 0.3155 to 0.2654 in FP16. At six steps Q8_0 cut the bundle by 19% and peak RSS by 21%, with a 2.9% increase in the two-image median. The most compact configuration was 5.8× faster than the 25-step SD v1.4 reference. Every row scored 0/2 on exact composition. For the red-apple prompt, the 25-step Tiny example gives the clearest apples of the illustrated phone renders, and the requested cup is missing. The tested Adreno paths crashed or produced unusable frames. [Failure cases](FAILURES.md)

## Hosted timing and fallback

Chrome on a physical Poco F6 displayed a 512 px result in 2.47 seconds through the public laptop GPU worker. Browser timing starts at submit and ends after image decode and display. The raw laptop study times generation inside the process. The public path also runs the checkpoint safety checker, OpenCV, WebP packaging, and transport.

In the final six browser runs, five displayed in 1.00–2.49 seconds. One desktop run took 11.33 seconds end to end with 2.29 seconds inside the worker. An earlier retest lost its tunnel connection during navigation and recovered on retry. The public URL uses a temporary tunnel to the laptop.

With the GPU worker's budget set to 1.6 seconds, Auto selected the six-call, 384 px profile and returned both image versions in 0.66 seconds from job acceptance. With the worker set to CPU, Auto selected five calls at 320 px and returned in 8.93 seconds. These were single warmed requests using the same prompt and seed. [Machine-readable measurements](../p3/web/evidence.json)
