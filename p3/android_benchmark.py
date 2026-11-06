"""Run a matched diffusion benchmark on an authorized USB-connected Android phone."""

import argparse
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from PIL import Image

from p3.benchmark import read_manifest, write_json
from p3.vision import enhance, resize


def benchmark(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.variant):
        raise ValueError("Use a variant ID consisting of letters, digits, dashes, or underscores.")
    rows = read_manifest(args.manifest)[:args.limit]
    root = args.out / args.variant
    root.mkdir(parents=True, exist_ok=True)
    adb = [str(args.adb), "-d"]

    def capture(*command):
        return subprocess.check_output([*adb, *command], text=True, encoding="utf-8").strip()

    def call(*command):
        subprocess.run([*adb, *map(str, command)], check=True)

    product = capture("shell", "getprop", "ro.product.device")
    model = capture("shell", "getprop", "ro.product.model")
    android = capture("shell", "getprop", "ro.build.version.release")
    if capture("shell", "getprop", "ro.product.cpu.abi") != "arm64-v8a":
        raise RuntimeError("Connect an ARM64 Android device.")

    remote = f"/data/local/tmp/p3-t38/{args.variant}"
    call("shell", "mkdir", "-p", f"{remote}/images")

    def stage_weights(path):
        if path.suffix not in {".gguf", ".safetensors"}:
            raise ValueError("Use GGUF or Safetensors mobile weights.")
        with path.open("rb") as file:
            checksum = hashlib.file_digest(file, "sha256").hexdigest()
        target = f"/data/local/tmp/p3-t38/{checksum[:16]}{path.suffix}"
        size = path.stat().st_size
        if capture("shell", f"stat -c %s {target} 2>/dev/null || echo missing") != str(size):
            call("push", path, target)
        return target, checksum, size

    model_path, checksum, size = stage_weights(args.model)
    decoder_args = []
    if args.taesd:
        decoder_path, decoder_checksum, decoder_size = stage_weights(args.taesd)
        decoder_args.append(decoder_path)
        size += decoder_size
    call("push", args.binary, f"{remote}/p3-native")
    call("shell", "chmod", "700", f"{remote}/p3-native")

    environment = {
        "label": args.label,
        "backend": args.backend,
        "hardware": f"Poco F6 ({model})" if product == "peridot" else model,
        "os": f"Android {android}",
        "model_bytes": size,
        "model_sha256": checksum,
        "decoder": "taesd" if args.taesd else "vae",
        "steps": args.steps,
        "native_size": args.size,
        "requested_samples": len(rows),
        "vendor_opencl": args.vendor_opencl,
    }
    if args.taesd:
        environment["decoder_sha256"] = decoder_checksum
    index = {sample["id"]: sample for sample in rows}
    with TemporaryDirectory(prefix="android-input-", dir=root) as scratch:
        manifest = Path(scratch) / "prompts.tsv"
        with manifest.open("w", encoding="utf-8", newline="\n") as file:
            for sample in rows:
                prompt = " ".join(sample["prompt"].split())
                file.write(f"{sample['id']}\t{sample['seed']}\t{prompt}\n")
        call("push", manifest, f"{remote}/prompts.tsv")
        runtime_environment = ["env", "LD_LIBRARY_PATH=/vendor/lib64"] if args.vendor_opencl else []
        invocation = shlex.join([
            *runtime_environment,
            f"{remote}/p3-native", model_path, f"{remote}/prompts.tsv",
            f"{remote}/images", str(args.size), str(args.steps), args.backend, *decoder_args,
        ])
        process = subprocess.Popen(
            [*adb, "shell", invocation], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        completed = {}
        assert process.stdout is not None
        with (root / "native.log").open("w", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                fields = line.strip().split("\t")
                if fields[0] == "P3_LOAD":
                    environment["load_ms"] = float(fields[1])
                    environment["model_type"] = fields[3]
                    environment["load_rss_bytes"] = int(fields[2])
                elif fields[0] == "P3_RUN" and fields[1] == "warmup":
                    environment["first_generation_ms"] = float(fields[2])
                elif fields[0] == "P3_RUN" and fields[1] in index:
                    completed[fields[1]] = (float(fields[2]), int(fields[3]))
                    print(args.variant, fields[1], f"{fields[2]} ms", flush=True)
        environment["exit_code"] = process.wait()
        write_json(root / "environment.json", environment)
        if completed:
            call("pull", f"{remote}/images", scratch)
            (root / "images").mkdir(exist_ok=True)
        valid = 0
        for sample in rows:
            key = sample["id"]
            record = {"sample": sample}
            if key in completed:
                with Image.open(Path(scratch) / "images" / f"{key}.ppm") as opened:
                    image = opened.convert("RGB")
                record.update(
                    generation_ms=completed[key][0], peak_rss_bytes=completed[key][1],
                )
                started = perf_counter()
                checked = enhance(image)
                record["quality"] = checked.before
                if not checked.usable:
                    record["error"] = "The quality check rejected the native decode."
                else:
                    output = resize(checked.image if args.cv else image, 512)
                    record["off_device_cv_ms"] = (perf_counter() - started) * 1000 if args.cv else None
                    output.save(root / "images" / f"{key}.png")
                    valid += 1
            else:
                record["error"] = "No image returned. See the native log and exit code."
            write_json(root / "runs" / f"{key}.json", record)
    environment["valid_samples"] = valid
    write_json(root / "environment.json", environment)
    return environment["exit_code"] or (0 if valid == len(rows) else 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--taesd", type=Path, help="Use an external tiny autoencoder for decoding.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--vendor-opencl", action="store_true", help="Load the phone's vendor OpenCL library.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cv", action="store_true", help="Apply OpenCV after pulling images. Excluded from phone timings.")
    raise SystemExit(benchmark(parser.parse_args()))
