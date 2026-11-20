# P3 Image Lab

**O(n) Eyes · Team T38**

P3 Image Lab is a resource-efficient text-to-image generation system that runs entirely on the visitor's device. The browser app uses ONNX Runtime with WebGPU to perform diffusion inference locally, so no server-side GPU is needed. It works offline after the initial model download via a service worker.

[Open P3 Image Lab](https://on-eyes-device-lab-t38.brawnymoss1024.chatgpt.site) · [Browser evaluation](docs/BROWSER_ON_DEVICE.md) · [Deployment](docs/DEPLOYMENT.md)

## Run locally

Build and serve the browser app:

```sh
cd browser
npm ci --ignore-scripts
npm run build
```

Then serve the `browser/dist` directory with any static HTTP server, or use the included device server:

```sh
python scripts/serve_device.py --web browser/dist --port 8080
```

Open `http://localhost:8080` in a WebGPU-capable browser (Chrome 113+). The first load downloads the pinned ONNX model files (~855 MB). Subsequent loads use the browser cache and work offline.

`Dockerfile.device` packages a CPU-only static host and remote asset proxy:

```sh
docker build -f Dockerfile.device -t p3-device . && docker run --rm -p 8080:8080 p3-device
```

## Export ONNX models for the browser

To re-export the models from PyTorch/Diffusers to ONNX:

```sh
pip install -r requirements-web-export.txt
python -m p3.export_browser --out browser-export
```

This exports the text encoder, U-Net, TAESD decoder, and scheduler config as ONNX files with FP16 weights.

## Architecture

The browser app implements:

- **CLIP text encoding** via ONNX Runtime WebAssembly
- **DPM-Solver++ denoising** with configurable step counts (4/6/12)
- **TAESD decoding** for lightweight latent-to-image conversion
- **Classical CV post-processing** (speckle repair, bilateral filtering, color correction, quality gate, and a transparent no-reference pixel-health diagnostic) in JavaScript
- **Service worker** for offline operation after initial model download
- **Adaptive step profiles** for quality vs. latency tradeoff
- **Prompt-aware Android planner** using prompt complexity, device class, measured local time, and an automatic local fallback chain
- **Sampling reference profile** (20 steps) for a matched reduced-step benchmark
- **In-page Optimization Lab** to compare the same prompt/seed at 20 steps and an optimized profile, plus optional visual CV ablations
- **On-device prompt checks** with a 3.6 MB YOLOX-Nano detector: object presence, exact counts, colour inside each detected box, and 2D layout; depth and non-COCO objects are shown as not verifiable
- **Budgeted retry and selection**: Balanced retries once with a new seed when a checkable clause fails; Slow detail keeps the best of up to three candidates within a time budget; the 20-step reference is never retried or selected
- **Quality hints**: an optional negative prompt and quality suffix that add no denoiser calls
- **Device readiness report and simulated budgets** to show adaptive selection and the real fallback chain live
- **Installable app** (web manifest and icons) on top of the offline service worker

## On-device evaluation (Poco F6)

All inference and CV ran in Chrome on the physical Poco F6. Each row uses the same 16 prompts/seeds at native 256 px, displayed at 512 px.

| Configuration | CLIP ↑ | Latency p50 / p95 (s) | Peak Chrome PSS (MiB) | Model size (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Browser FP16 · 4 steps / 8 denoiser calls | 0.2853 | 5.877 / 6.237 | 1,553 | 854.857 |
| Browser FP16 · 6 steps / 12 calls | 0.3001 | 7.744 / 7.848 | 2,372 | 854.857 |
| Browser FP16 · 12 steps / 24 calls | 0.3053 | 14.648 / 14.900 | 1,518 | 854.857 |

Six steps reduced median time by ~47% versus twelve, with CLIP lower by 0.005. A separate offline test took 16.54 s with networks off, zero server requests, and 8,844 GPU dispatches.

The active source now includes prompt-aware profile selection and automatic retry after a local resource failure. Those changes require a new physical-phone benchmark before their performance or semantic benefit can be claimed. See [Android resource strategy and evaluation](docs/ANDROID_RESOURCE_STRATEGY.md).

## Project structure

```
browser/           In-browser ONNX WebGPU diffusion app
  src/             Application source (app, worker, vision, tokenizer, etc.)
  test/            Browser pipeline tests
p3/                Python utilities (offline export, training and evaluation only; no serving)
  export_browser.py   Export PyTorch models to ONNX for browser
  quantize_browser.py Weight-only INT8 U-Net variant with agreement report
  import_browser_run.py Import phone captures for scoring
  benchmark.py, score.py  CPU reference benchmark and independent offline scorer (CLIP, YOLOS, FID, KID)
  train_lora.py    COCO LoRA training (the adapter fused into the browser model)
  models.py        Model checkpoint definitions
  provenance.py    Experiment provenance and hashing
scripts/           Benchmarking, phone monitoring, dev server
data/              Browser validation logs
deploy/            Hosting configuration
docs/              Documentation and evidence
```

## Measure the current source

Everything below writes to a new folder and never overwrites evidence.

```powershell
# Phone: 20-step reference, Balanced, Eco and Preview, then offline scoring and the report table
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 -Origin <https-origin> -Tab <tab-id> -Adb <adb.exe> -SelectionStudy

# Optional INT8 experiment (keep FP16 unless the phone shows a real gain)
python -m p3.quantize_browser --export .models/browser/v3 --out .models/browser/v3-int8
```

Step-by-step instructions, including preparing the offline verifier, are in [HANDOFF.md](HANDOFF.md).

## Documentation

- [System design](docs/DESIGN.md)
- [Handoff and run book](HANDOFF.md)
- [Browser on-device design and evaluation](docs/BROWSER_ON_DEVICE.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Evaluation for Team T38](eval_P3_T38.md)
- [Android resource strategy and evaluation](docs/ANDROID_RESOURCE_STRATEGY.md)
