# Poco F6 on-device benchmark

The runner executes diffusion on the phone's ARM64 CPU and Adreno GPU through Vulkan. It holds one model context across the warm-up and timed prompts. It records inference wall time, peak process RSS, model load time, and the size of the exact weight file. CV can be applied to the pulled outputs for scoring. Its time is tracked separately on the host.

## Build

Run the repository's [Build Android benchmark](../.github/workflows/android.yml) workflow and download the `p3-native-android-arm64` artifact. For a local build, use CMake, Ninja, an Android NDK, and a host Vulkan shader compiler. The workflow shows the exact build flags and pins stable-diffusion.cpp and its GGML dependency.

## Prepare weights

Fetch the [pinned Diffusers converter](https://github.com/huggingface/diffusers/blob/v0.37.0/scripts/convert_diffusers_to_original_stable_diffusion.py). Use the same conversion process for the baseline and Tiny. The starting mobile comparison uses FP16 storage for both checkpoints to keep the reference within the device budget.

```sh
python -m p3.export_mobile --model baseline --converter convert_diffusers_to_original_stable_diffusion.py --out .models/mobile/sd14-f16.safetensors
python -m p3.export_mobile --model tiny --converter convert_diffusers_to_original_stable_diffusion.py --out .models/mobile/tiny-f16.safetensors
sd-cli -M convert -m .models/mobile/tiny-f16.safetensors --type q8_0 -o .models/mobile/tiny-q8.gguf
```

## Run

Connect the phone over USB, enable USB debugging, and accept its trust prompt. Select only one USB-connected phone. Run each configuration with the same prompt manifest, resolution, guidance, and backend.

```sh
python -m p3.android_benchmark --binary p3-native --manifest data/composition.jsonl \
  --out .runs/mobile --variant baseline --label "SD v1.4 FP16 · 50 steps" \
  --model .models/mobile/sd14-f16.safetensors --steps 50
python -m p3.android_benchmark --binary p3-native --manifest data/composition.jsonl \
  --out .runs/mobile --variant tiny25 --label "Tiny FP16 · 25 steps" \
  --model .models/mobile/tiny-f16.safetensors --steps 25
python -m p3.android_benchmark --binary p3-native --manifest data/composition.jsonl \
  --out .runs/mobile --variant tiny8 --label "Tiny FP16 · 8 steps" \
  --model .models/mobile/tiny-f16.safetensors --steps 8
python -m p3.android_benchmark --binary p3-native --manifest data/composition.jsonl \
  --out .runs/mobile --variant tiny8-q8 --label "Tiny Q8_0 · 8 steps" \
  --model .models/mobile/tiny-q8.gguf --steps 8
```

Use `--adb` to specify the Platform Tools binary if it is outside PATH. `--limit` supports short smoke runs. Review the recorded native log to verify runtime placement and capture any allocation failures. RSS measures the process high-water mark. Driver-managed GPU allocations can sit outside that figure.
