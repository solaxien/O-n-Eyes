# P3 implementation review results

Review completed on 7 November 2026. These are new diagnostic pilots on the local RTX 4050 and physical Poco F6. The original six-by-300 COCO results, two-prompt native study, and failed Adreno trials are preserved. No T4 or external GPU was used. [Machine-readable evidence with protocols, image hashes and per-clause diagnostics](review-evidence.json)

## Verified fixes

The supplementary review also fixes permanent Quality-mode disablement after a transient warm-up failure. Auto gets a 30-second cooldown with a finite estimate, explicit Quality can retry within the memory/time budget, and successful observations recalibrate it. Tests cover recovery, downgrades, insufficient time, CPU fallback, OOM, and the shared one-retry limit.

Training now binds caches to image contents, source weights/configuration, code and runtime, validates encoded tensors, and uses checksum-verified immutable checkpoint blobs with an atomic resume pointer. Existing adapters and legacy work directories are preserved. Corrupt COCO reference PNGs are repaired. No original training was repeated. The Android workflow now installs and verifies the tested NDK **30.0.16248370** and writes build provenance; CPU remains the default backend.

- Benchmark protocols now bind model/adapter bytes, configuration, source, manifest, hardware and software. Incompatible resumes fail safely. Compatible images require decoded RGB, dimensions, hashes and committed records. Warm-up metadata belongs to each session, and old timings remain intact.
- Atomic image/JSON writes, exclusive run locks, history copies, stale-score invalidation and score-input checks prevent silent reuse after corruption or regeneration. A Windows sharing-error retry was validated after it occurred in the device control.
- Scoring uses shared valid IDs for every comparison and reports coverage separately. Detection boxes, confidences, crop colors, and per-clause outcomes remain inspectable.
- The mobile exporter fuses the trained adapter, pins the converter, verifies all mapped tensors, and records artifact provenance. Native image checkpoints are saved as they arrive, including raw failures and thermal observations.
- The app loads its local adapter in offline mode and has a real readiness endpoint. Docker/Compose include non-root execution, weight caching, readiness, restart-on-exit and bounded logs.

Twenty-nine targeted regression checks passed, along with Python compilation and dependency consistency. The local browser test made six real requests at desktop and mobile widths, checked Raw/Enhanced, layout and 77-token errors, and displayed images in 0.53–1.52 s. These are localhost checks, not public-network or on-device inference timings. Docker is unavailable on this laptop, so container build/runtime checks remain pending.

## Controlled host pilot

All nine variants use the same 16 prompts and seeds at native 512 px, one warm-up per configuration, no retries, and offline safety disabled equally. [Comparison matrix](REPRODUCIBILITY.md)

| Configuration | CLIP ↑ | FID | Warm p50 / p95 (s) | Peak GPU / RSS (MiB) | Bundle (MiB) | Count / color / position | Exact |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| sd25 | 0.3336 | N/A, N=16 | 3.684 / 3.746 | 2677 / 1753 | 2035 | 14/36 / 8/31 / 1/17 | 0/16 |
| tiny25 | 0.3325 | N/A, N=16 | 2.172 / 2.213 | 1653 / 1731 | 1013 | 11/36 / 4/31 / 1/17 | 0/16 |
| tiny-pndm20 | 0.3352 | N/A, N=16 | 1.538 / 1.550 | 1653 / 1736 | 1013 | 9/36 / 2/31 / 1/17 | 0/16 |
| tiny20 | 0.3317 | N/A, N=16 | 1.444 / 1.451 | 1653 / 1743 | 1013 | 7/36 / 1/31 / 1/17 | 0/16 |
| taesd | 0.3323 | N/A, N=16 | 1.281 / 1.292 | 1042 / 1744 | 862 | 6/36 / 1/31 / 0/17 | 0/16 |
| tuned | 0.3341 | N/A, N=16 | 1.283 / 1.299 | 1042 / 1753 | 864 | 9/36 / 2/31 / 1/17 | 0/16 |
| tuned-guidance5 | 0.3325 | N/A, N=16 | 1.277 / 1.290 | 1042 / 1757 | 864 | 9/36 / 4/31 / 1/17 | 0/16 |
| tuned8 | 0.3394 | N/A, N=16 | 0.548 / 0.554 | 1042 / 1775 | 864 | 8/36 / 1/31 / 0/17 | 0/16 |
| tuned6 | 0.3410 | N/A, N=16 | 0.422 / 0.432 | 1042 / 1752 | 864 | 7/36 / 3/31 / 1/17 | 1/16 |

`sd25` versus `tiny25` holds FP16, original VAE, guidance and PNDM 25 fixed. `tiny25` versus `tiny-pndm20` changes only requested steps. Switching PNDM to DPM++ also changes 21 actual calls to 20. Decoder, adapter, guidance and step tests then have their own rows. Precision remains isolated in the historical FP32/FP16 study; it was not repeated.

Lower guidance doubled the very small color-clause count from 2 to 4 of 31 but did not improve exact success or relations and slightly lowered CLIP. The default stays 7.5. Six steps passed one exact prompt, while count and position coverage remained poor. That exploratory result does not establish a new quality default. The tuned 20-step outputs matched all 16 historical tuned frames pixel for pixel.

Sequential runs, small N and changing host conditions limit timing conclusions. The early host rows overlapped CPU-side export verification; native-phone timing ran on separate hardware. No population significance or general improvement is claimed. Global CLIP can rise while object-level results worsen. [Full visual audit](SEMANTIC_AUDIT.md)

## CV pilot

All rows reuse the same 16 native tuned decodes and output dimensions.

| CV variant | Additional median ms | CLIP | Count / color / position | Exact |
| --- | ---: | ---: | --- | ---: |
| tuned-raw | 0.06 | 0.3341 | 9/36 / 2/31 / 1/17 | 0/16 |
| tuned-full | 14.64 | 0.3345 | 8/36 / 1/31 / 1/17 | 0/16 |
| tuned-omit-repair | 14.60 | 0.3344 | 8/36 / 1/31 / 1/17 | 0/16 |
| tuned-omit-edges | 21.68 | 0.3341 | 9/36 / 2/31 / 1/17 | 0/16 |
| tuned-omit-color | 20.80 | 0.3345 | 8/36 / 1/31 / 1/17 | 0/16 |
| tuned-omit-gate | 17.63 | 0.3345 | 8/36 / 1/31 / 1/17 | 0/16 |

All 16 frames passed pixel sanity checks. Median Laplacian-variance retention was 91.8%; this is an edge/detail proxy, not proof of perceptual quality. Speckle repair ran on 16 images, edge filtering on 10, and chroma correction on one. Full CV lost one automatic count and color clause; omitting edge filtering preserved the raw clause totals. Raw output remains available. Filtering is not a semantic repair mechanism.

Known-corruption tests verify that isolated hot-pixel repair reduces error at the injected pixels by over 90%, a clean gradient stays within one pixel level, uniform failed frames retain their raw pixels and are rejected, and resizing preserves aspect ratio. These tests support specific pixel operations, not overall visual superiority. The small CV timing differences are noisy single-pass measurements and should not be interpreted as additive component speedups.

## Expanded native export validation

Both FP16 exports passed exact verification of 803 converted tensors. The adapted checkpoint changes 72 U-Net tensors using the trained LoRA. The new base export is byte-identical to the historical base export. Each configuration generated all 16 composition prompts at native 256 px, six DPM++2M steps, four ARM threads and external TAESD. The same compiled runner and phone were used. No CV, resize or safety checker is part of its inference timing.

| Configuration | CLIP ↑ | FID | Warm p50 / p95 (s) | Peak GPU / RSS (MiB) | Bundle (MiB) | Count / color / position | Exact |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| base-f16-6 | 0.2606 | N/A, N=16 | 27.646 / 29.779 | — / 967 | 1021 | 3/36 / 0/31 / 0/17 | 0/16 |
| adapted-f16-6 | 0.2717 | N/A, N=16 | 21.577 / 29.010 | — / 1514 | 1021 | 2/36 / 0/31 / 0/17 | 0/16 |

The adapted run started at 32.5 °C battery temperature and ended at 40.0 °C. The base control ran later; its recorded thermal state and sequential order confound latency differences. Do not attribute the timing difference to the adapter. RSS is the native process high-water mark, including load/warm-up. Native images were pulled, checked and resized on the host. All 32 frames passed the pixel gate; object binding and realism remain inadequate. Sixteen unique prompts are still short of the blueprint's 30 repeated warm runs per configuration.

The base control started at 39.5 °C and ended at 41.9 °C battery temperature. Adapted peak RSS was higher, 1,514 versus 967 MiB, despite equal stored bundle sizes. That memory cost also needs repeat investigation before selecting an adapted native default.

![Actual adapted native frames at 256 px](evidence/adapted-native.png)

Export SHA-256 values:

- Base `695ee51f65dba7da30574d7c9e0d87392eba4b1966a0cb05a4e96c11be7b3d28`
- Adapted `0b1c7e8a1b1c2a5381c2f25578f9f359e04d8d0ab520fd0f432fe5a135e161d6`
- Adapter `dcffb4476adcf795689bd710679eaaaf96d19bf5a7ec7f23bf0a46e2cd295c49`

The interrupted base-control attempt after a Windows sharing error remains in its original directory. A new control directory records the fix; no failed timing was merged into the completed run. [Native scope, budgets and GPU diagnostic plan](MOBILE_FEASIBILITY.md)

## Remaining limits

The refreshed [temporary public demo](https://holland-comedy-screensavers-rare.trycloudflare.com) was tested after the final adaptive fix on 7 November. Six actual browser requests at desktop and phone widths took **0.76–2.28 s**, including transport and display. These are browser-layout tests through the laptop GPU, not native-phone inference. The previous tunnel no longer resolved. The replacement is also temporary and needs this laptop to remain online. The site labels its original benchmark tables historical and links the new [review page](https://holland-comedy-screensavers-rare.trycloudflare.com/review.html).

The native CPU route does not meet ten-second inference or the proposed semantic-quality gates. Device-side CV, an adapted Q8 export, accelerator operator parity, and sustained thermal evaluation remain future work. Adreno outputs previously failed and are not relabeled as successful generation.

A stable public HTTPS deployment remains unresolved without continuously available hardware and routing. The old temporary tunnel is historical. The revised Python app is checked locally; the container path is reproducible configuration awaiting a Docker host. [Deployment instructions](DEPLOYMENT.md)

The native deliverable is a benchmark executable. There is no Activity, AndroidManifest, Gradle app project or APK. The supplied brief does not explicitly require an APK, but an interactive native Android product remains an unimplemented feature. [Requirements-to-evidence map and judging walkthrough](JUDGING.md)
