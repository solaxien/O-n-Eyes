"""Generate, score, and summarize paired model and post-processing ablations."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
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


VARIANTS = {
    "baseline": Variant("1 · SD v1.4 FP32 · 50 steps", "baseline", "fp32", 50, scheduler="checkpoint"),
    "baseline-half": Variant("2 · SD v1.4 FP16 · 50 steps", "baseline", "fp16", 50, scheduler="checkpoint"),
    "tiny25": Variant("3 · Tiny FP16 · 25 steps", "tiny", "fp16", 25, scheduler="checkpoint"),
    "tiny20": Variant("4 · Tiny FP16 · DPM++ 20", "tiny", "fp16", 20),
    "taesd": Variant("5 · Tiny FP16 · DPM++ 20 · TAESD", "tiny", "fp16", 20, decoder="taesd"),
    "tuned": Variant("6 · Tiny FP16 · DPM++ 20 · TAESD · COCO LoRA", "tiny", "fp16", 20, decoder="taesd", lora=True),
    "balanced": Variant("Tiny FP16 · DPM++ 8 · TAESD · COCO LoRA + CV", "tiny", "fp16", 8, cv=True, decoder="taesd", lora=True),
    "eco": Variant("Tiny FP16 · DPM++ 6 · TAESD · COCO LoRA + CV · 384 px", "tiny", "fp16", 6, 384, True, decoder="taesd", lora=True),
}
OFFICIAL_VARIANTS = ("baseline", "baseline-half", "tiny25", "tiny20", "taesd", "tuned")
CV_VARIANTS = {"raw": "Raw decode", "full": "Full OpenCV pass", "linear": "Bilinear resize"}
CV_VARIANTS.update({f"omit-{stage}": f"CV without {stage}" for stage in CV_STAGES})


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", key) for key in ids):
        raise ValueError("Use a nonempty manifest with unique alphanumeric, dash, or underscore IDs.")
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    from p3.generate import GenerationError, Generator, Profile
    import torch

    variant = VARIANTS[variant_id]
    if variant.lora and adapter is None:
        raise ValueError("This variant requires --adapter pointing to the trained COCO LoRA.")
    root = destination / variant_id
    root.mkdir(parents=True, exist_ok=True)
    with sample_rss() as cold_peak:
        generator = Generator(
            variant.checkpoint, variant.precision, device,
            decoder=variant.decoder, adapter=adapter if variant.lora else None,
        )
    profile = Profile(variant.label, variant.steps, variant.size, variant.scheduler)
    with sample_rss() as first_peak:
        first = generator.generate(
            "A photograph of a green pear on a plain kitchen counter.",
            0, profile, process_image=variant.cv
        )
    write_json(root / "environment.json", {
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
    })
    for directory in ("runs", "images", "raw"):
        (root / directory).mkdir(exist_ok=True)
    for sample in rows:
        key = sample["id"]
        record = root / "runs" / f"{key}.json"
        if record.is_file():
            saved = json.loads(record.read_text(encoding="utf-8"))
            if saved.get("sample") == sample and "error" not in saved and (root / "images" / f"{key}.png").is_file():
                continue
        try:
            with sample_rss() as peak:
                generated = generator.generate(sample["prompt"], sample["seed"], profile, process_image=variant.cv)
            generated.image.save(root / "images" / f"{key}.png")
            if variant.cv:
                generated.raw.save(root / "raw" / f"{key}.png")
            data = generated.metadata | {"peak_rss_bytes": peak[0], "sample": sample}
            print(f"{variant_id} {key}: {data['generation_ms']:.0f} ms", flush=True)
        except (GenerationError, torch.OutOfMemoryError) as exc:
            data = {"sample": sample, "error": str(exc)}
            print(f"{variant_id} {key}: {type(exc).__name__}", flush=True)
        write_json(record, data)


def ablate_cv(destination: Path, from_variant: str):
    root = destination / from_variant
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    for key, label in CV_VARIANTS.items():
        if key == "linear" and environment.get("native_size") == 512:
            # Rows at 512 px already have the requested delivery resolution.
            continue
        ablation = destination / f"{from_variant}-{key}"
        (ablation / "images").mkdir(parents=True, exist_ok=True)
        write_json(ablation / "environment.json", environment | {"label": label})
        for record_path in sorted((root / "runs").glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if "error" in record:
                write_json(ablation / "runs" / record_path.name, record)
                continue
            source = root / "raw" / f"{record_path.stem}.png"
            if not source.is_file():
                source = root / "images" / f"{record_path.stem}.png"
            with Image.open(source) as opened:
                raw = opened.convert("RGB")
            started = perf_counter()
            omitted = frozenset({key.removeprefix("omit-")}) if key.startswith("omit-") else frozenset()
            processed = None if key == "raw" else enhance(raw, omitted)
            image = resize(processed.image if processed else raw, 512, linear=key == "linear")
            cv_ms = (perf_counter() - started) * 1000
            image.save(ablation / "images" / f"{record_path.stem}.png")
            data = {
                "sample": record["sample"],
                "cv_ms": cv_ms,
                "stages": processed.applied if processed else [],
                "before": processed.before if processed else None,
                "after": processed.after if processed else None,
            }
            write_json(ablation / "runs" / record_path.name, data)


def summarize(root: Path) -> dict:
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "runs").glob("*.json")]
    succeeded = [row for row in rows if "error" not in row]
    timings = [row["generation_ms"] for row in succeeded if "generation_ms" in row]
    cv_times = [
        row["stages_ms"]["cv_and_resize"] if "stages_ms" in row else row["cv_ms"]
        for row in succeeded if "stages_ms" in row or "cv_ms" in row
    ]
    quality_path = root / "scores.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    quality.pop("by_sample", None)
    return environment | {
        "id": root.name,
        "samples": len(succeeded),
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
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, nargs="+", required=True)
    report.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    if options.command == "generate":
        rows = read_manifest(options.manifest)[:options.limit]
        run_variant(options.variant, rows, options.out, options.device, options.adapter)
    elif options.command == "cv":
        ablate_cv(options.out, options.from_variant)
    elif options.command == "score":
        from p3.score import score_runs
        score_runs(options.manifest, options.run, options.device, options.fid)
    elif options.command == "report":
        write_json(options.out, [summarize(path) for path in options.run])


if __name__ == "__main__":
    main()
