"""Checkpointed native Android diffusion; quality checks and CV run on the host."""

import argparse
from pathlib import Path
import re
import shlex
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from PIL import Image

from p3.benchmark import read_manifest
from p3.provenance import (
    SCHEMA, archive_sample, establish_protocol, exclusive_run, file_hash, invalidate_scores,
    read_json, save_image, session_id, software_identity, source_hashes, valid_record, write_json,
)
from p3.vision import enhance, resize


def benchmark(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.variant):
        raise ValueError("Use an alphanumeric variant ID with dashes or underscores.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not 256 <= args.size <= 512 or args.size % 64 or not 1 <= args.steps <= 50:
        raise ValueError("Use 256–512 px in multiples of 64 and 1–50 steps")
    with exclusive_run(args.out / args.variant):
        return _benchmark(args)


def _benchmark(args):
    rows = read_manifest(args.manifest)[:args.limit]
    root = args.out / args.variant
    adb = [str(args.adb), "-d"]

    def capture(*command):
        return subprocess.check_output([*adb, *map(str, command)], text=True, encoding="utf-8", timeout=30).strip()

    def call(*command):
        subprocess.run([*adb, *map(str, command)], check=True, timeout=600)

    def prop(name):
        return capture("shell", "getprop", name)

    if prop("ro.product.cpu.abi") != "arm64-v8a":
        raise RuntimeError("Connect an ARM64 Android device.")
    product, model, android = prop("ro.product.device"), prop("ro.product.model"), prop("ro.build.version.release")
    for path in (args.model, args.taesd):
        if path and path.suffix not in {".gguf", ".safetensors"}:
            raise ValueError("Use GGUF or Safetensors mobile weights")
    checksum = file_hash(args.model)
    decoder_checksum = file_hash(args.taesd) if args.taesd else None
    sidecar = args.model.with_suffix(args.model.suffix + ".json")
    export = read_json(sidecar) if sidecar.exists() else None
    if export and export.get("artifact_sha256") != checksum:
        raise ValueError("Mobile checkpoint does not match its export provenance")
    protocol = {
        "schema": SCHEMA, "kind": "native-android", "manifest": rows,
        "model_sha256": checksum, "decoder_sha256": decoder_checksum,
        "export": export, "binary_sha256": file_hash(args.binary),
        "hardware": {"product": product, "model": model, "android": android,
                     "build": prop("ro.build.fingerprint"), "vendor_build": prop("ro.vendor.build.fingerprint")},
        "settings": {"steps": args.steps, "size": args.size, "backend": args.backend,
                     "vendor_opencl": args.vendor_opencl, "threads": 4, "guidance": 7.5,
                     "sampler": "DPM++2M", "scheduler": "discrete", "warmups_per_session": 1,
                     "off_device_cv": args.cv, "display_size": 512, "safety_checker": False},
        "host_software": software_identity(),
        "code": source_hashes("android_benchmark.py", "provenance.py", "vision.py"),
    }
    fingerprint = establish_protocol(root, protocol)
    pending = [row for row in rows if valid_record(root, row, fingerprint) is None]
    if not pending:
        print(f"{args.variant}: {len(rows)} native image checkpoints verified", flush=True)
        return 0
    session = session_id()
    invalidate_scores(root, session)
    remote = f"/data/local/tmp/p3-t38/{fingerprint[:16]}/{session}"
    call("shell", "mkdir", "-p", f"{remote}/images")

    def stage_weights(path, sha):
        target = f"/data/local/tmp/p3-t38/{sha}{path.suffix}"
        existing = capture("shell", f"sha256sum {target} 2>/dev/null || echo missing").split()[0]
        if existing != sha:
            call("push", path, target)
        if capture("shell", "sha256sum", target).split()[0] != sha:
            raise ValueError("Device weight transfer failed its SHA-256 check")
        return target

    model_path = stage_weights(args.model, checksum)
    decoder_args = [stage_weights(args.taesd, decoder_checksum)] if args.taesd else []
    call("push", args.binary, f"{remote}/p3-native")
    if capture("shell", "sha256sum", f"{remote}/p3-native").split()[0] != protocol["binary_sha256"]:
        raise ValueError("Device runner checksum mismatch")
    call("shell", "chmod", "700", f"{remote}/p3-native")

    def thermal():
        data = {}
        for name, command in (("battery", "dumpsys battery"), ("thermal", "dumpsys thermalservice")):
            try:
                text = capture("shell", command)
                data[name] = [line.strip() for line in text.splitlines()
                              if re.search(r"temperature|status|Temperature", line, re.I)][:30]
            except (OSError, subprocess.SubprocessError):
                data[name] = ["unavailable"]
        return data

    environment = {
        "label": args.label, "backend": args.backend,
        "hardware": f"Poco F6 ({model})" if product == "peridot" else model,
        "os": f"Android {android}", "model_bytes": args.model.stat().st_size + (args.taesd.stat().st_size if args.taesd else 0),
        "model_sha256": checksum, "decoder_sha256": decoder_checksum,
        "decoder": "taesd" if args.taesd else "vae", "steps": args.steps, "native_size": args.size,
        "requested_samples": len(rows), "vendor_opencl": args.vendor_opencl, "protocol_hash": fingerprint,
        "adapter": export.get("adapter") if export else "unknown; no export sidecar",
        "scope": "native diffusion only; resize, QA, optional CV and scoring on host",
        "session": session, "thermal_before": thermal(),
    }
    first_session = not (root / "environment.json").exists()
    if first_session:
        write_json(root / "environment.json", environment)
    session_path = root / "sessions" / f"{session}.json"
    write_json(session_path, environment | {"status": "running"})
    index = {row["id"]: row for row in pending}
    for row in pending:
        archive_sample(root, row["id"], session)
        write_json(root / "runs" / f"{row['id']}.json", {
            "sample": row, "protocol_hash": fingerprint, "session": session, "status": "pending"})
    with TemporaryDirectory(prefix="android-input-", dir=root) as scratch_dir:
        scratch = Path(scratch_dir)
        manifest = scratch / "prompts.tsv"
        manifest.write_text("".join(f"{row['id']}\t{row['seed']}\t{' '.join(row['prompt'].split())}\n"
                                    for row in pending), encoding="utf-8")
        call("push", manifest, f"{remote}/prompts.tsv")
        runtime_environment = ["env", "LD_LIBRARY_PATH=/vendor/lib64"] if args.vendor_opencl else []
        invocation = shlex.join([*runtime_environment, f"{remote}/p3-native", model_path,
                                 f"{remote}/prompts.tsv", f"{remote}/images", str(args.size), str(args.steps),
                                 args.backend, *decoder_args])
        process = subprocess.Popen([*adb, "shell", invocation], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        try:
            with (root / "sessions" / f"{session}.log").open("w", encoding="utf-8", buffering=1) as log:
                for line in process.stdout:
                    log.write(line)
                    fields = line.strip().split("\t")
                    if fields[0] == "P3_LOAD":
                        environment.update(load_ms=float(fields[1]), load_rss_bytes=int(fields[2]), model_type=fields[3])
                    elif fields[0] == "P3_RUN" and fields[1] == "warmup":
                        environment["first_generation_ms"] = float(fields[2])
                    elif fields[0] == "P3_RUN" and fields[1] in index:
                        key = fields[1]
                        stamp = {"sample": index[key], "protocol_hash": fingerprint, "session": session,
                                 "generation_ms": float(fields[2]), "peak_rss_bytes": int(fields[3])}
                        try:
                            ppm = scratch / f"{key}.ppm"
                            call("pull", f"{remote}/images/{key}.ppm", ppm)
                            with Image.open(ppm) as opened:
                                image = opened.convert("RGB")
                            if image.size != (args.size, args.size):
                                raise ValueError("Native output dimensions do not match the protocol")
                            raw_info = save_image(root / "raw" / f"{key}.png", image)
                            started = perf_counter()
                            checked = enhance(image)
                            if not checked.usable:
                                raise ValueError("Host pixel check rejected native decode; raw pixels preserved.")
                            output = resize(checked.image if args.cv else image, 512)
                            off_device_ms = (perf_counter() - started) * 1000
                            record = stamp | {"status": "complete", "quality": checked.before,
                                              "off_device_qa_cv_resize_ms": off_device_ms, "off_device_cv": args.cv,
                                              "artifacts": {"raw": raw_info, "images": save_image(root / "images" / f"{key}.png", output)},
                                              "thermal_after": thermal()}
                        except (OSError, ValueError, subprocess.SubprocessError) as exc:
                            record = stamp | {"status": "failed", "error": str(exc)}
                        write_json(root / "runs" / f"{key}.json", record)
                        print(args.variant, key, fields[2], record["status"], flush=True)
                    if fields[0] in {"P3_LOAD", "P3_RUN", "P3_ERROR"}:
                        write_json(session_path, environment | {"status": "running"})
            environment["exit_code"] = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    valid = sum(valid_record(root, row, fingerprint) is not None for row in rows)
    environment.update(valid_samples=valid, thermal_after=thermal())
    write_json(session_path, environment | {"status": "finished"})
    if first_session:
        write_json(root / "environment.json", environment)
    return environment["exit_code"] or (0 if valid == len(rows) else 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--taesd", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--vendor-opencl", action="store_true", help="Diagnostic use only; tested Adreno path failed.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cv", action="store_true", help="Apply CV on host after generation; excluded from phone timings.")
    raise SystemExit(benchmark(parser.parse_args()))
