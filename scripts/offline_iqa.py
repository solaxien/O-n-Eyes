"""Offline no-reference image quality for phone outputs (BRISQUE, lower is better).

Compares the raw decode with the enhanced image for every sample in a
benchmark_browser.cjs capture, so the classical CV stage is judged by a metric
it was not tuned on. BRISQUE uses OpenCV's published LIVE model files, which
are downloaded once and checked by SHA-256.

  pip install opencv-contrib-python-headless
  python scripts/offline_iqa.py --run .runs/phase1-.../eco --out .runs/phase1-.../iqa-eco.json
"""

import argparse
import hashlib
import json
import statistics
import urllib.request
from pathlib import Path

import cv2

MODEL_BASE = "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/modules/quality/samples/"
MODELS = {
    "brisque_model_live.yml": "4c44c7eec9e5139830c0bb0e4a044c9cd351ba5bbc9e3798beca3243e2782ff4",
    "brisque_range_live.yml": "a18427f4f7ad087524bd0d63389649a3899b51ca6d44a4d5f8d8171f1f914a9c",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models(cache: Path) -> tuple[str, str]:
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in MODELS.items():
        path = cache / name
        if not path.exists():
            urllib.request.urlretrieve(MODEL_BASE + name, path)
        if sha(path) != expected:
            path.unlink()
            raise SystemExit(f"{name} failed its checksum; delete the cache and retry.")
    return str(cache / "brisque_model_live.yml"), str(cache / "brisque_range_live.yml")


def brisque(path: Path, model: str, ranges: str) -> float:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return float(cv2.quality.QualityBRISQUE_compute(image, model, ranges)[0])


def evaluate(run: Path, cache: Path) -> dict:
    model, ranges = models(cache)
    experiment = json.loads((run / "experiment.json").read_text())
    rows = []
    for sample in experiment["manifest"]:
        record = json.loads((run / f"{sample['id']}.json").read_text())
        artifacts = record["artifacts"]
        raw, enhanced = run / artifacts["raw"]["file"], run / artifacts["enhanced"]["file"]
        row = {"id": sample["id"], "raw": brisque(raw, model, ranges), "enhanced": brisque(enhanced, model, ranges)}
        row["improved"] = row["enhanced"] < row["raw"]
        rows.append(row)
    return {
        "metric": "BRISQUE (OpenCV, LIVE model); lower is better",
        "run": str(run),
        "n": len(rows),
        "raw_mean": statistics.mean(r["raw"] for r in rows),
        "enhanced_mean": statistics.mean(r["enhanced"] for r in rows),
        "improved_fraction": sum(r["improved"] for r in rows) / len(rows),
        "models_sha256": MODELS,
        "rows": rows,
        "caveat": "BRISQUE was trained on natural-photo distortions; treat it as one signal, not a verdict on realism.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/brisque"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite an existing result.")
    result = evaluate(args.run, args.cache)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print({k: v for k, v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
