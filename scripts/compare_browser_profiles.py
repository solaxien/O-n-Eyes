"""Create a provenance-preserving cost comparison from completed phone runs.

This never generates or scores images. It only compares the records captured
by benchmark_browser.cjs, so a 20-step reference and reduced-step runs can be
reported without mixing devices, prompts, or model bundles by accident.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarise(directory):
    experiment_path = directory / "experiment.json"
    completion_path = directory / "completion.json"
    experiment, completion = read(experiment_path), read(completion_path)
    manifest = experiment["manifest"]
    if completion.get("complete") != len(manifest):
        raise ValueError(f"{directory}: incomplete capture")
    records, record_hashes = [], {}
    for sample in manifest:
        path = directory / f"{sample['id']}.json"
        record = read(path)
        if record.get("sample") != sample or record.get("warmup"):
            raise ValueError(f"{path}: sample identity mismatch")
        records.append(record)
        record_hashes[sample["id"]] = digest(path)
    latency = [record["metadata"]["submit_to_visible_ms"] for record in records]
    model_bytes = {record["metadata"].get("model_bytes") for record in records}
    if len(model_bytes) != 1:
        raise ValueError(f"{directory}: model size varies within one run")
    memory_path = directory / "memory.json"
    memory = read(memory_path) if memory_path.exists() else {}
    return {
        "directory": str(directory),
        "requested_mode": experiment.get("mode"),
        "structured_prompt": experiment.get("structured_prompt"),
        "samples": len(records),
        "profiles_used": dict(Counter(record["metadata"]["profile"]["key"] for record in records)),
        "steps_used": dict(Counter(record["metadata"]["steps"] for record in records)),
        "fallbacks": sum(bool(record["metadata"].get("recovery", {}).get("used")) for record in records),
        "latency_ms": {
            "min": min(latency),
            "p50": median(latency),
            "p95": percentile(latency, 0.95),
            "max": max(latency),
        },
        "peak_pss_mib": memory.get("peak_pss_mib"),
        "model_bytes": model_bytes.pop(),
        "provenance": {
            "experiment_sha256": digest(experiment_path),
            "completion_sha256": digest(completion_path),
            "records": record_hashes,
            "manifest_sha256": experiment.get("manifest_sha256"),
            "cached_model_manifest_sha256": experiment.get("identity", {}).get("modelIndex", {}).get("manifest_sha256"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True,
                        help="Completed benchmark_browser.cjs output directory; repeat for each profile.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing comparison artifact")
    runs = [summarise(path.resolve()) for path in args.run]
    manifests = {run["provenance"]["manifest_sha256"] for run in runs}
    bundles = {run["provenance"]["cached_model_manifest_sha256"] for run in runs}
    prompts = {run["structured_prompt"] for run in runs}
    if len(manifests) != 1 or len(bundles) != 1 or len(prompts) != 1:
        raise ValueError("Runs differ in prompt manifest, cached model bundle, or prompt-conditioning policy")
    baseline = next((run for run in runs if run["requested_mode"] == "baseline"), None)
    if baseline:
        for run in runs:
            run["vs_baseline"] = {
                "p50_latency_ratio": run["latency_ms"]["p50"] / baseline["latency_ms"]["p50"],
                "p95_latency_ratio": run["latency_ms"]["p95"] / baseline["latency_ms"]["p95"],
                "peak_pss_ratio": (
                    run["peak_pss_mib"] / baseline["peak_pss_mib"]
                    if run["peak_pss_mib"] and baseline["peak_pss_mib"] else None
                ),
            }
    result = {
        "schema": 1,
        "scope": "Measured browser cost only. Image quality and semantic correctness require an independent scorer.",
        "comparison_source_sha256": digest(Path(__file__).resolve()),
        "baseline_present": baseline is not None,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "baseline_present": baseline is not None,
        "p50_ms": {run["requested_mode"]: run["latency_ms"]["p50"] for run in runs},
    }, indent=2))


if __name__ == "__main__":
    main()
