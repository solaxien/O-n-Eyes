# Poco F6 on-device benchmark

The benchmark runs diffusion directly on the ARM CPU of a Poco F6 with Android 16. The runner uses four threads and one model context across the warm-up and timed prompts. It records inference wall time, peak process RSS, model load time, and weight bundle size. The host checks the native pixels before counting a run. Optional OpenCV enhancements on the pulled images are timed separately.

## Build

Download `p3-native-android-arm64` from the Build Android benchmark workflow or build it locally with NDK r30, CMake, and Ninja. `ANDROID_NDK_HOME` should point to the NDK directory. [Build workflow](../.github/workflows/android.yml)

```sh
mkdir -p .tools
git clone --depth 1 --branch master-1052-f5df5f8 \
  https://github.com/leejet/stable-diffusion.cpp.git .tools/sd
git -C .tools/sd submodule update --init --depth 1 ggml
cmake -S android -B .runs/android-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSD_SOURCE="$PWD/.tools/sd" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
  -DSD_VULKAN=OFF -DSD_OPENCL=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod
cmake --build .runs/android-cpu --target p3-native
```

The Poco F6's vendor Vulkan compiler exited with SIGSEGV during the first denoiser pass. OpenCL with Flash Attention returned flat frames. The trial with standard attention, CPU text encoding, and CPU TAESD returned patterned noise. The study below uses the CPU renderer and records the GPU trials as failures. [Failure analysis](../docs/FAILURES.md)

## Prepare weights

Fetch the [pinned Diffusers converter](https://github.com/huggingface/diffusers/blob/v0.37.0/scripts/convert_diffusers_to_original_stable_diffusion.py). Use the same conversion process for the baseline and Tiny. The starting mobile comparison uses FP16 storage for both checkpoints to keep the reference within the device budget.

```sh
python -m p3.export_mobile --model baseline --converter convert_diffusers_to_original_stable_diffusion.py --out .models/mobile/sd14-f16.safetensors
python -m p3.export_mobile --model tiny --converter convert_diffusers_to_original_stable_diffusion.py --out .models/mobile/tiny-f16.safetensors
sd-cli -M convert -m .models/mobile/tiny-f16.safetensors --type q8_0 -o .models/mobile/tiny-q8.gguf
python -c 'from p3.models import resolve_taesd; resolve_taesd()'
```

## Run

Connect one phone over USB and authorize USB debugging. These commands run the first two prompts in the same order with fixed seeds, 256 px native resolution, DPM++ 2M, and TAESD. Each configuration runs a separate warm-up before recording the two timings. The study compares SD v1.4 and Tiny at 25 and eight steps in FP16, then tests six steps and Q8_0 storage on Tiny.

```sh
p3_mobile() {
  python -m p3.android_benchmark --binary .runs/android-cpu/p3-native \
    --manifest data/composition.jsonl --out .runs/mobile-cpu \
    --taesd .models/taesd/diffusion_pytorch_model.safetensors \
    --size 256 --backend cpu --limit 2 "$@"
}
p3_mobile --variant sd14-f16-8 --label "SD v1.4 FP16 · 8 steps" \
  --model .models/mobile/sd14-f16.safetensors --steps 8
p3_mobile --variant tiny-f16-8 --label "BK-SDM-Tiny FP16 · 8 steps" \
  --model .models/mobile/tiny-f16.safetensors --steps 8
p3_mobile --variant tiny-f16-6 --label "BK-SDM-Tiny FP16 · 6 steps" \
  --model .models/mobile/tiny-f16.safetensors --steps 6
p3_mobile --variant tiny-q8-6 --label "BK-SDM-Tiny Q8_0 · 6 steps" \
  --model .models/mobile/tiny-q8.gguf --steps 6
p3_mobile --variant sd14-f16-25 --label "SD v1.4 FP16 · 25 steps" \
  --model .models/mobile/sd14-f16.safetensors --steps 25
p3_mobile --variant tiny-f16-25 --label "BK-SDM-Tiny FP16 · 25 steps" \
  --model .models/mobile/tiny-f16.safetensors --steps 25
```

Use `--adb` if the Platform Tools binary is outside `PATH`. `--limit 1` shortens a smoke run. Peak RSS is the high-water mark for the runner process. The bundle size includes the exported checkpoint and external TAESD weights. Timing covers model inference through decoded pixels. The benchmark then pulls the image, runs the pixel quality check, and resizes it to 512 px on the laptop.
