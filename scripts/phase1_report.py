"""Build the current-source evaluation table from measured runs.

Inputs are files the pipeline already writes:
  * a cost comparison from scripts/compare_browser_profiles.py (phone timings,
    Chrome PSS, model bytes, U-Net calls), and
  * scores.json files from `python -m p3.benchmark score` run on imported phone
    images (independent CLIP and YOLOS composition scorer).

Each measurement comes from the input files. Their SHA-256 hashes accompany the table.

Usage:
  python scripts/phase1_report.py --cost .runs/phase1/cost.json \
      --scored baseline=.runs/phase1/scored-baseline ... \
      --out .runs/phase1/report
"""

import argparse
import hashlib
import json
from pathlib import Path

ORDER = ["baseline", "balanced", "eco", "preview", "quality"]
LABELS = {
    "baseline": "20-step reference",
    "balanced": "Balanced, 12 steps",
    "eco": "Eco, 6 steps",
    "preview": "Preview, 4 steps",
    "quality": "Slow detail, 12 steps, 384 px",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(text: str) -> tuple[str, Path]:
    key, _, value = text.partition("=")
    if not key or not value:
        raise argparse.ArgumentTypeError("Use NAME=PATH")
    return key, Path(value)


def fmt(value, digits=3, scale=1.0, suffix=""):
    if value is None:
        return "N/A"
    return f"{value * scale:.{digits}f}{suffix}"


def clause(scores: dict, name: str) -> str:
    item = scores.get("clauses", {}).get(name)
    return f"{item['passed']}/{item['total']}" if item and item.get("total") else "N/A"


def run_label(run: dict) -> str:
    base = LABELS.get(run["requested_mode"], run["requested_mode"])
    extras = []
    if run.get("verify"):
        extras.append("verifier + retry")
    if run.get("quality_hints"):
        extras.append("quality hints")
    return base + (f" ({', '.join(extras)})" if extras else "")


def build(cost_path: Path, scored: dict[str, Path]) -> tuple[str, dict]:
    cost = json.loads(cost_path.read_text())
    runs = {Path(r["directory"]).name: r for r in cost["runs"]}
    proofs = {"cost": {"path": str(cost_path), "sha256": sha(cost_path)}, "scores": {}}
    lines = [
        "## On-device results (Poco F6, current source)",
        "",
        "Same prompts and seeds in every row. Timings are warm submit-to-visible on the phone. "
        "CLIP and composition come from the separate offline scorer (CLIP ViT-B/32, YOLOS-Tiny), "
        "not from the on-device verifier.",
        "",
        "| Configuration | CLIP ↑ | KID / FID | Latency p50 / p95 (s) | Peak Chrome PSS (MiB) | Model size (MiB) "
        "| U-Net calls (total) | Count / colour / position | Exact |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    ordered = sorted(runs.items(), key=lambda kv: (
        ORDER.index(kv[1]["requested_mode"]) if kv[1]["requested_mode"] in ORDER else 99,
        kv[1].get("verify", False), kv[1].get("quality_hints", False)))
    for name, run in ordered:
        score_dir = scored.get(name)
        scores = {}
        if score_dir:
            score_file = score_dir / "scores.json"
            scores = json.loads(score_file.read_text())
            proofs["scores"][name] = {"path": str(score_file), "sha256": sha(score_file)}
        exact = (f"{round(scores['composition_pass'] * scores['composition_n'])}/{scores['composition_n']}"
                 if scores.get("composition_pass") is not None else "N/A")
        lines.append(
            f"| {run_label(run)} | {fmt(scores.get('clip_mean'), 4)} | N/A, N={run['samples']} "
            f"| {fmt(run['latency_ms']['p50'], 2, 0.001)} / {fmt(run['latency_ms']['p95'], 2, 0.001)} "
            f"| {fmt(run.get('peak_pss_mib'), 0)} | {fmt(run['model_bytes'], 1, 1 / 2**20)} "
            f"| {run.get('unet_calls_total', 'N/A')} | {clause(scores, 'count')} / {clause(scores, 'color')} / "
            f"{clause(scores, 'position')} | {exact} |"
        )
    baseline = next((r for r in runs.values() if r["requested_mode"] == "baseline"), None)
    if baseline:
        lines += ["", "Speed against the 20-step reference (p50):", ""]
        for name, run in ordered:
            if run is baseline:
                continue
            ratio = baseline["latency_ms"]["p50"] / run["latency_ms"]["p50"]
            lines.append(f"- {run_label(run)}: {ratio:.2f}x {'faster' if ratio >= 1 else 'slower'}")
    not_verifiable = {name: json.loads((d / "scores.json").read_text()).get("not_verifiable_clauses")
                      for name, d in scored.items()}
    if any(not_verifiable.values()):
        lines += ["", f"Clauses the 2D scorer cannot judge (depth, scenery) are excluded and counted separately: "
                      f"{max(v or 0 for v in not_verifiable.values())} per run."]
    lines += ["", "Source files and hashes: `phase1-provenance.json`."]
    return "\n".join(lines) + "\n", proofs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--scored", type=pair, nargs="*", default=[], help="RUN_DIR_NAME=SCORED_DIR")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite an existing report directory.")
    table, proofs = build(args.cost, dict(args.scored))
    args.out.mkdir(parents=True)
    (args.out / "phase1-results.md").write_text(table, encoding="utf-8")
    (args.out / "phase1-provenance.json").write_text(json.dumps(proofs, indent=2) + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
