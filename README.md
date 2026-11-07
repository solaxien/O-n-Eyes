# P3 Image Lab

**O(n) Eyes · Team T38**

P3 Image Lab now runs the adapted diffusion model **in the visitor's browser**. Prompt encoding, WebGPU generation, decoding, resizing and classical CV all stay on that device. The website serves static assets and has no inference API.

[Open the on-device app](https://on-eyes-device-lab-t38.brawnymoss1024.chatgpt.site) · [Browser measurements, architecture and limitations](docs/BROWSER_ON_DEVICE.md) · [Model release](https://github.com/solaxien/O-n-Eyes/releases/tag/browser-models-v3)

The physical Poco F6 generated a fresh image after an offline page reload with Wi-Fi and mobile data disabled. Six-step repeat generation had a 7.744 s median with models kept loaded; its initial load/generation took 15.82 s. The 896 MB model download is additional. This is a working research prototype: exact multi-object composition remained 0/16, and the first download, cold initialization and slow detail mode do not meet ten seconds. Supported WebGPU/FP16 browsers and sufficient storage are required.

## Run the on-device web app

The host below only serves files; it needs no GPU. The visitor's device performs generation.

```sh
docker build -f Dockerfile.device -t p3-device . && docker run --rm -p 7860:7860 p3-device
```

Open `http://localhost:7860` on that computer, or use the public HTTPS app from a phone. Docker runtime validation is pending because Docker is unavailable on the development laptop. The plain Node build and Python static server have been exercised. [Setup, source export, device tests and scoring](docs/BROWSER_ON_DEVICE.md#reproduce)

The original host/native implementation and all historical measurements remain intact. They are separate experiments, not browser benchmark results. [Review](docs/REVIEW_RESULTS.md) · [Reproduction protocol](docs/REPRODUCIBILITY.md) · [Deployment](docs/DEPLOYMENT.md)

## Archived host-based demo

On a Linux machine with Docker, an NVIDIA driver, and the NVIDIA Container Toolkit, use this command.

```sh
docker build -t p3-image-lab . && docker run --rm --gpus all -p 7860:7860 -v p3-weights:/weights p3-image-lab
```

Open [localhost:7860](http://localhost:7860). The first launch downloads the pinned public weights into the volume. The UI shows worker status while the model loads.

For a persistent container service with weight caching, readiness checks and restart policy, use `docker compose -f deploy/compose.yaml up --build -d`. This needs a Docker-equipped NVIDIA host; container execution has not been tested on the current Windows laptop. `/api/ready` returns HTTP 503 until the model is ready. A permanent public endpoint still requires an available host and HTTPS routing.

For Windows or Linux with Python 3.12–3.14 and a CUDA-compatible NVIDIA driver, create a virtual environment and use these commands.

```sh
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m uvicorn p3.app:app --host 127.0.0.1 --port 7860
```

Use one worker so only one model is resident. Set `P3_MODELS` to change the weight directory and `P3_DEVICE=cpu` for a low-resolution CPU run. CPU generation uses a longer request budget. Set `P3_REQUEST_BUDGET` in seconds if the host needs a different limit. The default GPU budget is ten seconds from job acceptance to packaging.

## Prepare data and train the adapter

```sh
python -m pip install -r requirements-eval.txt
python -m p3.fetch_coco --split train --count 1000 --out .runs/data/train
mkdir -p .runs/data/val
cp data/coco-val2017.jsonl .runs/data/val/manifest.jsonl
python -m p3.fetch_coco --split validation --count 300 --out .runs/data/val
python -m p3.train_lora --manifest .runs/data/train/manifest.jsonl --out .models/training/t38-coco-new.safetensors --work .runs/training-v2
```

The adapter trains for one pass over train2017 using rank-4 attention LoRA. New training uses a fresh work directory and output filename. Existing adapters and legacy training caches are preserved. The app loads the bundled adapter if present. Set `P3_ADAPTER` to load an alternate local adapter. The fixed validation manifest selects one caption per image and adds an evaluation seed. Its captions are from the COCO Consortium under CC BY 4.0, indexed via the Hugging Face dataset mirror. [COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm), [Caption index](https://huggingface.co/datasets/phiyodr/coco2017)

## Evaluate the six configurations

Use a fresh output directory when running the revised evaluator. Protocol changes are rejected, and legacy experiments cannot be resumed under a new identity. The `.runs/coco` paths below are examples for a fresh checkout; preserve an existing historical `.runs/coco` and substitute a new name. [Recovery, controlled comparisons and tests](docs/REPRODUCIBILITY.md)

```sh
for variant in baseline baseline-half tiny25 tiny20 taesd tuned; do
  python -m p3.benchmark generate --manifest .runs/data/val/manifest.jsonl --variant "$variant" \
    --out .runs/coco --adapter p3/weights/t38-coco.safetensors || break
done
python -m p3.benchmark score --manifest .runs/data/val/manifest.jsonl \
  --run .runs/coco/{baseline,baseline-half,tiny25,tiny20,taesd,tuned} --fid
python -m p3.benchmark cv --out .runs/coco --from-variant tuned
python -m p3.benchmark score --manifest .runs/data/val/manifest.jsonl \
  --run .runs/coco/tuned-{raw,full,omit-repair,omit-edges,omit-color,omit-gate} --fid
python -m p3.benchmark report --run .runs/coco/{baseline,baseline-half,tiny25,tiny20,taesd,tuned} \
  --out .runs/coco/summary.json
```

| Variant | Denoiser and precision | Scheduler and requested steps | Decoder and adaptation |
| --- | --- | --- | --- |
| `baseline` | SD v1.4 FP32 | Checkpoint, 50 | Original VAE |
| `baseline-half` | SD v1.4 FP16 | Checkpoint, 50 | Original VAE |
| `tiny25` | BK-SDM-Tiny-2M FP16 | Checkpoint, 25 | Original VAE |
| `tiny20` | BK-SDM-Tiny-2M FP16 | DPM-Solver++, 20 | Original VAE |
| `taesd` | BK-SDM-Tiny-2M FP16 | DPM-Solver++, 20 | TAESD |
| `tuned` | BK-SDM-Tiny-2M FP16 | DPM-Solver++, 20 | TAESD and COCO LoRA |

Use the same validation manifest for all six 512 px configurations. The first image warms the model and is recorded separately. The checkpoint scheduler may issue more denoiser calls than requested steps. The run records both. The CV study reuses row 6's decoded pixels. FID includes the number of common generated and real images used in the comparison. [COCO dataset](https://cocodataset.org/#download)

The same generation and scoring commands accept `data/composition.jsonl` for a separate study of object count, color, and position. Omit `--fid` for that set. The `balanced` and `eco` variants measure the tuned demo's lower-budget profiles. [GenEval methodology](https://github.com/djghosh13/geneval)

## Design

- [System design](docs/DESIGN.md)
- [Ablation study](docs/ABLATION.md)
- [Failure case analysis](docs/FAILURES.md)
- [Evaluation for Team T38](eval_P3_T38.md)
- [Android benchmark setup](android/README.md)
- [Initial blueprint](P3_BLUEPRINT.md)
- [Visual composition audit](docs/SEMANTIC_AUDIT.md)
- [Mobile feasibility](docs/MOBILE_FEASIBILITY.md)

The weights remain under their upstream licenses. The archived host worker retains the checkpoints' safety checker; the browser export omits that classifier. Pixel QA is not a safety classifier. [Model use terms](browser/src/MODEL-LICENSE.txt) · [Attribution](browser/src/ATTRIBUTION.txt) · [SD v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) · [BK-SDM-Tiny-2M](https://huggingface.co/nota-ai/bk-sdm-tiny-2m)
