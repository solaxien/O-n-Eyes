"""Generate, score, and summarize paired model and post-processing ablations."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import re
from threading import Event, Thread
from time import perf_counter

import numpy as np
from PIL import Image
import psutil

from p3.vision import CV_STAGES, enhance, resize
from p3.provenance import (
    SCHEMA, archive_sample, digest, establish_protocol, exclusive_run, file_hash,
    inputs_snapshot, invalidate_scores, load_protocol, read_json, runtime_identity,
    save_image, scores_current, session_id, source_hashes, valid_record, write_json,
)


@dataclass(frozen=True)
class Variant:
    label: str
    checkpoint: str
    precision: str
    steps: int
    size: int = 512
    cv: bool = False
    scheduler: str = "dpmpp"
    decoder: str = "vae"
    lora: bool = False
    guidance: float = 7.5


VARIANTS = {
    "baseline": Variant("1 · SD v1.4 FP32 · 50 steps", "baseline", "fp32", 50, scheduler="checkpoint"),
    "baseline-half": Variant("2 · SD v1.4 FP16 · 50 steps", "baseline", "fp16", 50, scheduler="checkpoint"),
    "tiny25": Variant("3 · Tiny FP16 · 25 steps", "tiny", "fp16", 25, scheduler="checkpoint"),
    "tiny20": Variant("4 · Tiny FP16 · DPM++ 20", "tiny", "fp16", 20),
    "taesd": Variant("5 · Tiny FP16 · DPM++ 20 · TAESD", "tiny", "fp16", 20, decoder="taesd"),
    "tuned": Variant("6 · Tiny FP16 · DPM++ 20 · TAESD · COCO LoRA", "tiny", "fp16", 20, decoder="taesd", lora=True),
    "balanced": Variant("Tiny FP16 · DPM++ 8 · TAESD · COCO LoRA + CV", "tiny", "fp16", 8, cv=True, decoder="taesd", lora=True),
    "eco": Variant("Tiny FP16 · DPM++ 6 · TAESD · COCO LoRA + CV · 384 px", "tiny", "fp16", 6, 384, True, decoder="taesd", lora=True),
    "sd25": Variant("SD v1.4 FP16 · PNDM 25", "baseline", "fp16", 25, scheduler="checkpoint"),
    "tiny-pndm20": Variant("Tiny FP16 · PNDM 20", "tiny", "fp16", 20, scheduler="checkpoint"),
    "tuned8": Variant("Tuned · 8 steps · raw", "tiny", "fp16", 8, decoder="taesd", lora=True),
    "tuned6": Variant("Tuned · 6 steps · raw", "tiny", "fp16", 6, decoder="taesd", lora=True),
    "tuned6-384": Variant("Tuned · 6 steps · 384 px · raw", "tiny", "fp16", 6, size=384, decoder="taesd", lora=True),
    "tuned-guidance5": Variant("Tuned · guidance 5 · raw", "tiny", "fp16", 20, decoder="taesd", lora=True, guidance=5.0),
}
OFFICIAL_VARIANTS = ("baseline", "baseline-half", "tiny25", "tiny20", "taesd", "tuned")
CV_VARIANTS = {"raw": "Raw decode", "full": "Full OpenCV pass", "linear": "Bilinear resize"}
CV_VARIANTS.update({f"omit-{stage}": f"CV without {stage}" for stage in CV_STAGES})


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", key) for key in ids):
        raise ValueError("Use a nonempty manifest with unique alphanumeric, dash, or underscore IDs.")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
           or type(row.get("seed")) is not int or not 0 <= row["seed"] < 2**32 for row in rows):
        raise ValueError("Every sample needs a nonempty prompt and an unsigned 32-bit seed.")
    return rows


def host_protocol(variant: Variant, rows: list[dict], device: str, adapter: Path | None) -> dict:
    from p3.models import CHECKPOINTS, TAESD, model_files, resolve_model, resolve_taesd
    directory, _ = resolve_model(variant.checkpoint, variant.precision, include_safety=False)
    weights = {name: file_hash(directory / name) for name in model_files(variant.precision, include_safety=False)}
    if variant.decoder == "taesd":
        decoder, _ = resolve_taesd()
        weights.update({f"taesd/{name}": file_hash(decoder / name)
                        for name in ("config.json", "diffusion_pytorch_model.safetensors")})
    return {"schema": SCHEMA, "kind": "host-generation", "settings": asdict(variant),
            "manifest": rows, "checkpoint": asdict(CHECKPOINTS[variant.checkpoint]),
            "decoder_revision": asdict(TAESD) if variant.decoder == "taesd" else None,
            "weights": weights, "adapter_sha256": file_hash(adapter) if variant.lora else None,
            "runtime": runtime_identity(device),
            "code": source_hashes("benchmark.py", "generate.py", "models.py", "vision.py", "provenance.py"),
            "measurement": {"batch": 1, "warmups_per_session": 1, "safety": False,
                            "noise": "CPU FP32 seeded then cast", "rss_poll_seconds": 0.015,
                            "output_size": 512, "retries": 0}}


@contextmanager
def sample_rss():
    process = psutil.Process()
    peak = [process.memory_info().rss]
    done = Event()

    def observe():
        while not done.wait(0.015):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = Thread(target=observe, daemon=True)
    thread.start()
    try:
        yield peak
    finally:
        done.set()
        thread.join()
        peak[0] = max(peak[0], process.memory_info().rss)


def run_variant(variant_id: str, rows: list[dict], destination: Path, device: str, adapter: Path | None):
    root = destination / variant_id
    with exclusive_run(root):
        _run_variant(variant_id, rows, root, device, adapter)


def _run_variant(variant_id: str, rows: list[dict], root: Path, device: str, adapter: Path | None):
    from p3.generate import GenerationError, Generator, Profile
    import torch

    variant = VARIANTS[variant_id]
    if variant.lora and adapter is None:
        raise ValueError("This variant requires --adapter pointing to the trained COCO LoRA.")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    protocol = host_protocol(variant, rows, device, adapter)
    fingerprint = establish_protocol(root, protocol)
    pending = [sample for sample in rows if valid_record(root, sample, fingerprint) is None]
    if not pending:
        print(f"{root.name}: all {len(rows)} image and record hashes verified; no inference needed.", flush=True)
        return
    session = session_id()
    invalidate_scores(root, session)
    write_json(root / "sessions" / f"{session}.json", {"status": "loading", "protocol_hash": fingerprint})
    with sample_rss() as cold_peak:
        generator = Generator(
            variant.checkpoint, variant.precision, device,
            decoder=variant.decoder, adapter=adapter if variant.lora else None,
            check_safety=False,
        )
    profile = Profile(variant.label, variant.steps, variant.size, variant.scheduler, variant.guidance)
    with sample_rss() as first_peak:
        first = generator.generate(
            "A photograph of a green pear on a plain kitchen counter.",
            0, profile, process_image=variant.cv
        )
    environment = {
        "label": variant.label,
        "hardware": generator.hardware,
        "backend": device,
        "os": platform.platform(),
        "torch": torch.__version__,
        "load_ms": generator.load_ms,
        "first_generation_ms": first.metadata["generation_ms"],
        "cold_peak_rss_bytes": max(cold_peak[0], first_peak[0]),
        "model_bytes": generator.model_bytes,
        "revision": first.metadata["revision"],
        "requested_steps": first.metadata["requested_steps"],
        "denoiser_calls": first.metadata["steps"],
        "precision": first.metadata["precision"],
        "scheduler": first.metadata["scheduler"],
        "decoder": first.metadata["decoder"],
        "adapter": first.metadata["adapter"],
        "native_size": first.metadata["native_size"],
        "safety_check": first.metadata["safety_check"],
        "protocol_hash": fingerprint,
        "session": session,
    }
    # Keep the first environment immutable. Every resumed warm-up has its own record.
    if not (root / "environment.json").exists():
        write_json(root / "environment.json", environment)
    write_json(root / "sessions" / f"{session}.json", environment | {"status": "running"})
    for directory in ("runs", "images", "raw"):
        (root / directory).mkdir(exist_ok=True)
    for sample in pending:
        key = sample["id"]
        record = root / "runs" / f"{key}.json"
        archive_sample(root, key, session)
        stamp = {"sample": sample, "session": session, "protocol_hash": fingerprint}
        write_json(record, stamp | {"status": "running"})
        try:
            with sample_rss() as peak:
                generated = generator.generate(sample["prompt"], sample["seed"], profile, process_image=variant.cv)
            artifacts = {"images": save_image(root / "images" / f"{key}.png", generated.image),
                         "raw": save_image(root / "raw" / f"{key}.png", generated.raw)}
            data = generated.metadata | stamp | {"peak_rss_bytes": peak[0], "status": "complete", "artifacts": artifacts}
            print(f"{variant_id} {key}: {data['generation_ms']:.0f} ms", flush=True)
        except (GenerationError, torch.OutOfMemoryError) as exc:
            data = stamp | {"status": "failed", "error": str(exc)}
            print(f"{variant_id} {key}: {type(exc).__name__}", flush=True)
        write_json(record, data)
    write_json(root / "sessions" / f"{session}.json", environment | {"status": "finished"})


def ablate_cv(destination: Path, from_variant: str):
    root = destination / from_variant
    with exclusive_run(root):
        _ablate_cv(destination, from_variant)


def _ablate_cv(destination: Path, from_variant: str):
    root = destination / from_variant
    source_protocol = load_protocol(root)
    if source_protocol is None:
        raise ValueError("CV ablations require a fingerprinted source. Preserve legacy studies and use a new experiment.")
    rows = source_protocol["protocol"]["manifest"]
    records = [valid_record(root, row, source_protocol["fingerprint"]) for row in rows]
    if not all(records):
        raise ValueError("Complete or repair the source run before making paired CV ablations.")
    environment = read_json(root / "environment.json")
    sources = {row["sample"]["id"]: {"raw": row["artifacts"]["raw"], "record": digest(row)} for row in records}
    for key, label in CV_VARIANTS.items():
        if key == "linear" and environment.get("native_size") == 512:
            # Rows at 512 px already have the requested delivery resolution.
            continue
        ablation = destination / f"{from_variant}-{key}"
        with exclusive_run(ablation):
            fingerprint = establish_protocol(ablation, {
                "schema": SCHEMA, "kind": "cv-ablation", "manifest": rows,
                "source_protocol": source_protocol["fingerprint"], "source_images": sources,
                "stage": key, "output_size": 512, "runtime": runtime_identity("cpu"),
                "code": source_hashes("benchmark.py", "vision.py", "provenance.py"),
            })
            session = session_id()
            if not (ablation / "environment.json").exists():
                write_json(ablation / "environment.json", environment | {"label": label, "protocol_hash": fingerprint})
            pending = [row for row in rows if valid_record(ablation, row, fingerprint) is None]
            if pending:
                invalidate_scores(ablation, session)
            for row in pending:
                name = row["id"]
                archive_sample(ablation, name, session)
                stamp = {"sample": row, "protocol_hash": fingerprint, "session": session}
                write_json(ablation / "runs" / f"{name}.json", stamp | {"status": "running"})
                with Image.open(root / "raw" / f"{name}.png") as opened:
                    raw = opened.convert("RGB")
                started = perf_counter()
                omitted = frozenset({key.removeprefix("omit-")}) if key.startswith("omit-") else frozenset()
                processed = None if key == "raw" else enhance(raw, omitted)
                image = resize(processed.image if processed else raw, 512, linear=key == "linear")
                cv_ms = (perf_counter() - started) * 1000
                data = stamp | {
                    "status": "complete", "cv_ms": cv_ms,
                    "artifacts": {"images": save_image(ablation / "images" / f"{name}.png", image)},
                    "stages": processed.applied if processed else [],
                    "before": processed.before if processed else None,
                    "after": processed.after if processed else None,
                    "usable": processed.usable if processed else None,
                }
                write_json(ablation / "runs" / f"{name}.json", data)


def summarize(root: Path) -> dict:
    with exclusive_run(root):
        return _summarize(root)


def _summarize(root: Path) -> dict:
    environment = read_json(root / "environment.json")
    protocol = load_protocol(root)
    if protocol:
        rows = protocol["protocol"]["manifest"]
        succeeded = [record for sample in rows
                     if (record := valid_record(root, sample, protocol["fingerprint"])) is not None]
    else:
        rows = [read_json(path) for path in (root / "runs").glob("*.json")]
        succeeded = [row for row in rows if valid_record(root, row["sample"], None, allow_legacy=True)]
    timings = [row["generation_ms"] for row in succeeded if "generation_ms" in row]
    cv_times = [
        row["stages_ms"]["cv_and_resize"] if "stages_ms" in row else row["cv_ms"]
        for row in succeeded if "stages_ms" in row or "cv_ms" in row
    ]
    quality_path = root / "scores.json"
    quality = read_json(quality_path) if quality_path.exists() else {}
    quality_status = "legacy-unverified" if not protocol else "missing"
    if quality and protocol:
        quality_status = "verified" if scores_current(root, quality) else "stale-excluded"
        if quality_status == "stale-excluded":
            quality = {}
    quality.pop("by_sample", None)
    quality.pop("provenance", None)
    return environment | {
        "id": root.name,
        "samples": len(succeeded),
        "provenance_status": "fingerprinted" if protocol else "legacy-unverified",
        "quality_status": quality_status,
        "failures": len(rows) - len(succeeded),
        "latency_p50_ms": float(np.percentile(timings, 50)) if timings else None,
        "latency_p95_ms": float(np.percentile(timings, 95)) if timings else None,
        "cv_p50_ms": float(np.percentile(cv_times, 50)) if cv_times else None,
        "peak_rss_bytes": max((row.get("peak_rss_bytes", 0) for row in succeeded), default=0) or None,
        "peak_vram_bytes": max((row.get("peak_vram_bytes") or 0 for row in succeeded), default=0) or None,
    } | quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--variant", choices=VARIANTS, required=True)
    generate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--adapter", type=Path)
    cv = commands.add_parser("cv")
    cv.add_argument("--out", type=Path, required=True)
    cv.add_argument("--from-variant", default="tuned")
    score = commands.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--run", type=Path, nargs="+", required=True)
    score.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    score.add_argument("--fid", action="store_true")
    score.add_argument("--allow-legacy", action="store_true", help="Explicitly score historical outputs without original protocol hashes.")
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, nargs="+", required=True)
    report.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    if options.command == "generate":
        if options.limit is not None and options.limit < 1:
            parser.error("--limit must be positive")
        rows = read_manifest(options.manifest)[:options.limit]
        run_variant(options.variant, rows, options.out, options.device, options.adapter)
    elif options.command == "cv":
        ablate_cv(options.out, options.from_variant)
    elif options.command == "score":
        from p3.score import score_runs
        score_runs(options.manifest, options.run, options.device, options.fid, allow_legacy=options.allow_legacy)
    elif options.command == "report":
        write_json(options.out, [summarize(path) for path in options.run])


if __name__ == "__main__":
    main()
