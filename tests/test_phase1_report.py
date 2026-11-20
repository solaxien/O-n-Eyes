import json
import tempfile
import unittest
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("phase1_report", Path("scripts/phase1_report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def run(mode, p50, calls, verify=False):
    return {"directory": f"/x/{mode}{'-verify' if verify else ''}", "requested_mode": mode, "samples": 20,
            "verify": verify, "quality_hints": False, "latency_ms": {"p50": p50, "p95": p50 * 1.1},
            "peak_pss_mib": 1500, "model_bytes": 896382552, "unet_calls_total": calls}


class Phase1ReportTests(unittest.TestCase):
    def test_table_uses_measured_files_and_baseline_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cost = root / "cost.json"
            cost.write_text(json.dumps({"runs": [run("eco", 7700, 240), run("baseline", 24000, 800),
                                                 run("balanced", 15000, 600, verify=True)]}))
            scored = root / "scored-eco"
            scored.mkdir()
            (scored / "scores.json").write_text(json.dumps({
                "clip_mean": 0.3001, "composition_pass": 0.1, "composition_n": 19, "not_verifiable_clauses": 19,
                "clauses": {"count": {"passed": 5, "total": 36}, "color": {"passed": 3, "total": 31},
                            "position": {"passed": 0, "total": 17}}}))
            table, proofs = report.build(cost, {"eco": scored})
            lines = table.splitlines()
            self.assertLess(lines.index(next(l for l in lines if l.startswith("| 20-step"))),
                            lines.index(next(l for l in lines if l.startswith("| Eco"))))
            self.assertIn("| 0.3001 |", table)
            self.assertIn("5/36 / 3/31 / 0/17 | 2/19 |", table)
            self.assertIn("Eco, 6 steps: 3.12x faster", table)
            self.assertIn("Balanced, 12 steps (verifier + retry)", table)
            self.assertIn("sha256", json.dumps(proofs))


if __name__ == "__main__":
    unittest.main()
