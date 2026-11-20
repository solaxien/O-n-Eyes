# P3 System Design

The current public app uses visitor-device WebGPU/WASM inference. Its complete architecture, measured resource strategy and offline acceptance are in [Browser on-device design](BROWSER_ON_DEVICE.md). The host/native design below is retained for reproducing historical comparisons.

## Historical host runtime layout

```mermaid
flowchart TD
    A["Browser · prompt, mode, seed"] --> B["FastAPI · bounded job queue"]
    B --> C["Adaptive profile · time and memory budget"]
    C --> D["Tokenizer and CLIP text encoder"]
    D --> E["Seeded noise and guided U-Net denoising"]
    E --> F["VAE decode and checkpoint safety check"]
    F --> G["OpenCV repair, filtering, color, and QA"]
    G --> H["Resize, WebP, and paired Raw / Enhanced views"]
    G -->|One retry if budget permits| C
    I["Paired prompt and seed manifest"] --> D
    I --> J["Poco F6 · native ARM64 diffusion runner"]
    J --> K["Device timings and memory"]
    J --> M["Pull raw native pixels to host"]
    M --> N["Host-only resize, QA, optional CV and scoring"]
    O["Verified base or fused-LoRA export"] --> J
    H --> L["Offline CLIP, composition, and COCO FID"]
```

The hosted path runs Python, PyTorch, Diffusers components, and OpenCV on the inference host. Browser requests carry only prompt and profile settings. The browser polls the job endpoint for queue, encoding, denoising, decoding, and packaging states. A single worker serializes model use and the queue caps outstanding work at two requests. Finished images expire after three minutes and remain in bounded process memory.

The mobile performance track uses the same model families on the physical Poco F6. A pinned stable-diffusion.cpp runner uses four ARM CPU threads, memory-mapped weights, and TAESD. The original two-prompt results used **base checkpoints without the trained LoRA**. The new exporter can fuse and verify the adapter before conversion; its separate device validation is identified by the exported artifact hash. The Android executable performs prompt encoding, denoising, and decoding. Resize, quality gating, optional OpenCV, and offline scoring run on the laptop after pulling native pixels. The native executable does not include the hosted safety checker. [Android build and run instructions](../android/README.md)

## Generation and budget

The worker tokenizes the whole prompt and rejects input above the checkpoint’s CLIP context length. It encodes positive and empty conditioning, starts from seed-derived noise, applies classifier-free guidance, runs DPM-Solver++ updates, and decodes once through the VAE. It runs the checkpoint safety check before returning an image. The generator stores model revision, precision, effective steps and size, stage timings, and applied CV operations in run metadata.

| Profile | Default configuration |
| --- | --- |
| Reference | SD v1.4, FP32, checkpoint scheduler, 50 requested steps, 512 px |
| Full quality | BK-SDM-Tiny-2M FP16, DPM-Solver++ at 20 steps, TAESD and COCO LoRA |
| Balanced | Tiny FP16, 8 steps, 512 px |
| Low resource | Tiny FP16, 6 steps, 384 px, bicubic upscale to 512 px |
| CPU fallback | Tiny FP32, 5 steps, 320 px, bicubic upscale to 512 px |

The compact checkpoint removes U-Net blocks and distills the teacher’s outputs. The six-row study adds reduced precision, fewer sampling steps, DPM-Solver++, TAESD, and a COCO LoRA. The adapter trains rank-4 attention updates with the original U-Net, text encoder, and VAE weights frozen. It sees one pass over 1,000 captioned train2017 images and uses a separate 300-image val2017 manifest for evaluation.

Auto mode starts with the full quality profile when its warm-up estimate, remaining time, and free GPU memory fit. It can choose Balanced or Low resource instead. A GPU out-of-memory error can trigger one smaller run if time remains. The same one-retry limit applies to unusable decodes. The browser shows the actual resolution, denoiser calls, decoder, and adapter used.

A failed Quality warm-up uses a finite conservative estimate and a 30-second Auto cooldown. An explicit Quality request can retry when the memory/time checks pass. After cooldown, Auto can recalibrate Quality through a successful request. Successful observations recover even a legacy infinite estimate. If no eligible profile fits the remaining budget, the worker fails before starting inference. Runtime OOM and pixel-quality handling share the single-retry limit.

## Classical vision

All measurements and corrections run at native resolution. The pipeline uses median residuals and a gradient mask to localize isolated speckles, and Telea inpainting on tiny masks. Bilateral filtering on Lab luminance reduces flat-region noise while retaining edges. Low-contrast scenes receive a small CLAHE blend, and near-neutral midtones can trigger a bounded chroma correction. The gate checks clipping, edge retention, luminance range, and suspected artifact fraction. If a correction exceeds the clipping or edge-loss guard, the raw decode remains available. Area interpolation downsamples previews and bicubic interpolation enlarges low-resolution outputs.

The pixel checks address localized defects. Missing objects, attribute swaps, and reversed positions are evaluated offline against the original prompt. The automatic composition score combines exact detector counts, CLIP crop color labels, and a 5% image-axis margin for relative centers. The failure analysis distinguishes detector misses from visible generation errors.

## Measurement scope

The host benchmark records the exact model, sampler, resolution, precision, decoder, and adapter for each run. Within a fixed shape, the same FP32 seed-derived noise is cast to each tested precision. The FP32 reference disables TF32. Each counted denoiser call runs a batch containing the conditional and unconditional branches. The first six comparisons use raw decoded images. CV ablations reuse row 6's pixels. FID uses common generated IDs and the matched COCO references. At 300 images it is a comparison within this study, with the sample size reported next to each value. CLIP cosine and composition checks measure complementary properties. [Fixed validation cohort](../data/coco-val2017.jsonl)

On Windows the FP32 benchmark can load each tensor onto CUDA individually. After a CUDA interruption, this path bypassed the PyTorch memory-mapping crash and reproduced a saved image pixel for pixel. The timed study resumed from the saved samples after a separate warm-up.

Worker latency covers synchronized encoding, denoising, decoding, safety, and CV. Browser latency additionally covers queue, transport, and image decode. The offline quality study scores raw model outputs before the public worker's safety check. Its latency and bundle size use the components present in those offline runs. PyTorch reports peak allocated GPU bytes and process RSS is sampled independently. Mobile reports the native artifact bytes and physical process memory measured by its runner.

## Experiment integrity and evaluation scope

New runs use immutable content-hashed protocols and atomic per-image checkpoints. Model, adapter, inference source, hardware, software, scheduler settings, and the ordered manifest all participate. Incompatible resumes fail before old evidence is changed. Completed images are decoded and hash-checked. Each resumed session retains separate warm-up metadata. Regeneration archives stale scores; reports verify image/record provenance before using new scores. Historical results without these records remain labeled legacy evidence. [Exact protocol and reproduction](REPRODUCIBILITY.md)

The review's diagnostic comparisons isolate architecture, steps, scheduler, decoder, adapter, guidance, and CV as far as their runtime definitions permit. CLIP measures global alignment. Per-clause counts, colors, and relations remain separate and include detector diagnostics and visual-review caveats. No distribution metric is computed on a 16-image composition set. [Visual audit](SEMANTIC_AUDIT.md)

## Deployment and operating limits

The UI's `/api/health` reports loading, failed, or ready. `/api/ready` returns HTTP 503 until the model can serve work, enabling container readiness checks. A non-root Docker service, persistent weight volume, restart policy, and bounded logs provide a reproducible service layout. Container execution still needs validation on a Docker-equipped host. A permanent HTTPS endpoint still needs continuously available hardware and DNS/hosting access. The temporary laptop tunnel does not guarantee uptime. [Deployment instructions](DEPLOYMENT.md)

Local adapter loading passes its directory and explicit weight filename with `local_files_only=True`. This avoids Diffusers' filename-guessing failure when Hugging Face offline mode is enabled. The exporter uses the same explicit local selection. The review reproduced the failure and checked the corrected Python worker in offline mode.

Native latency, memory, and quality acceptance budgets are separate from the hosted browser budget. Local CPU inference remains the validated phone backend. No T4 or external GPU is used in this review at the user's instruction. [Mobile feasibility and remaining work](MOBILE_FEASIBILITY.md)
