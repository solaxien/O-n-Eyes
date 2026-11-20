"""Regression checks for actual corruption, interruption, and experiment mixing risks."""

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from p3 import benchmark
from p3.generate import GeneratedImage
from p3.provenance import (digest, establish_protocol, exclusive_run, inputs_snapshot,
                           load_protocol, read_json, save_image, scores_current,
                           source_hashes, valid_record, write_json)
from p3.score import _generated_image, paired_samples


ROWS = [{"id": "one", "prompt": "A red apple", "seed": 38},
        {"id": "two", "prompt": "A blue cup", "seed": 39}]


class FakeGenerator:
    calls = []
    hardware, load_ms, model_bytes = "test CPU", 3, 100

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt, seed, profile, **kwargs):
        self.calls.append(prompt)
        return GeneratedImage(Image.new("RGB", (512, 512), "red"), Image.new("RGB", (512, 512), "blue"), {
            "generation_ms": 10 + seed, "revision": "fixed", "requested_steps": profile.steps,
            "steps": profile.steps, "precision": "fp16", "scheduler": "DPM", "decoder": "taesd",
            "adapter": "fixture", "native_size": 512, "safety_check": False})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".runs" / "tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.root = self.out / "taesd"
        self.protocol = {"schema": 2, "manifest": ROWS, "settings": asdict(benchmark.VARIANTS["taesd"]),
                         "weights": "base", "adapter": "original", "runtime": "hostA", "code": "v1"}
        FakeGenerator.calls = []
        self.addCleanup(patch.stopall)
        patch("p3.generate.Generator", FakeGenerator).start()
        patch("p3.benchmark.host_protocol", side_effect=lambda *args: copy.deepcopy(self.protocol)).start()

    def run_model(self):
        benchmark.run_variant("taesd", ROWS, self.out, "cpu", None)

    def test_compatible_resume_preserves_timings_and_environment_without_loading(self):
        self.run_model()
        original = (self.root / "runs/one.json").read_bytes()
        environment = (self.root / "environment.json").read_bytes()
        self.run_model()
        self.assertEqual(len(FakeGenerator.calls), 3)
        self.assertEqual(original, (self.root / "runs/one.json").read_bytes())
        self.assertEqual(environment, (self.root / "environment.json").read_bytes())

    def test_every_protocol_change_refuses_mixing(self):
        self.run_model()
        before = (self.root / "environment.json").read_bytes()
        original = copy.deepcopy(self.protocol)
        for key in ("weights", "adapter", "runtime", "code", "settings", "manifest"):
            with self.subTest(key=key):
                self.protocol = original | {key: "changed"}
                with self.assertRaisesRegex(ValueError, "Protocol changed"):
                    self.run_model()
                self.assertEqual(before, (self.root / "environment.json").read_bytes())
        self.assertEqual(len(FakeGenerator.calls), 3)

    def test_corrupt_image_regenerates_only_one_and_archives_scores(self):
        self.run_model()
        saved = (self.root / "runs/two.json").read_bytes()
        environment = (self.root / "environment.json").read_bytes()
        write_json(self.root / "scores.json", {"clip_mean": 0.9})
        (self.root / "images/one.png").write_bytes(b"broken PNG")
        self.assertIsNone(_generated_image(self.root, ROWS[0]))
        self.run_model()
        self.assertEqual(len(FakeGenerator.calls), 5)
        self.assertEqual(saved, (self.root / "runs/two.json").read_bytes())
        self.assertEqual(environment, (self.root / "environment.json").read_bytes())
        self.assertFalse((self.root / "scores.json").exists())
        self.assertEqual(len(list((self.root / "history").glob("*/scores.json"))), 1)
        self.assertEqual(len(list((self.root / "sessions").glob("*.json"))), 2)

    def test_same_size_replaced_image_and_broken_raw_are_detected(self):
        self.run_model()
        Image.new("RGB", (512, 512), "green").save(self.root / "images/one.png")
        (self.root / "raw/two.png").write_bytes(b"broken")
        fingerprint = load_protocol(self.root)["fingerprint"]
        self.assertIsNone(valid_record(self.root, ROWS[0], fingerprint))
        self.assertIsNone(valid_record(self.root, ROWS[1], fingerprint))

    def test_interrupted_record_is_not_reused(self):
        self.run_model()
        record = read_json(self.root / "runs/one.json")
        record["status"] = "running"
        write_json(self.root / "runs/one.json", record)
        self.run_model()
        self.assertEqual(len(FakeGenerator.calls), 5)

    def test_legacy_refuses_resume_and_corrupt_legacy_is_unscorable(self):
        self.root.mkdir()
        write_json(self.root / "environment.json", {"hardware": "historical"})
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.run_model()
        write_json(self.root / "runs/one.json", {"sample": ROWS[0]})
        (self.root / "images").mkdir()
        (self.root / "images/one.png").write_bytes(b"invalid")
        self.assertIsNone(_generated_image(self.root, ROWS[0], allow_legacy=True))

    def test_corrupt_json_recovers(self):
        self.run_model()
        (self.root / "runs/one.json").write_text("{")
        self.run_model()
        self.assertEqual(len(FakeGenerator.calls), 5)

    def test_scores_reject_changed_inputs(self):
        self.run_model()
        scores = {"provenance": {"manifest": ROWS, "inputs": inputs_snapshot(self.root, ROWS),
                                "legacy": False, "scorer_code": source_hashes("score.py", "provenance.py")}}
        self.assertTrue(scores_current(self.root, scores))
        Image.new("RGB", (512, 512), "black").save(self.root / "images/one.png")
        self.assertFalse(scores_current(self.root, scores))

    def test_exclusive_writer(self):
        with exclusive_run(self.root):
            with self.assertRaises(RuntimeError):
                with exclusive_run(self.root):
                    pass

    def test_pairing_excludes_missing_samples_for_all_metrics(self):
        paired = paired_samples(ROWS, {"A": {"one": 1, "two": 2}, "B": {"one": 1}})
        self.assertEqual(paired, ROWS[:1])

    def test_transient_windows_replace_error_retries_without_losing_commit(self):
        import os
        if os.name != "nt":
            self.skipTest("Windows sharing violation")
        path = self.out / "sharing.json"
        write_json(path, {"before": True})
        replace = os.replace
        count = [0]

        def transient(source, target):
            count[0] += 1
            if count[0] == 1:
                self.assertEqual(read_json(path), {"before": True})
                raise PermissionError("temporary sharing violation")
            return replace(source, target)

        with patch("p3.provenance.os.replace", side_effect=transient):
            write_json(path, {"after": True})
        self.assertEqual(read_json(path), {"after": True})
        self.assertEqual(count[0], 2)


if __name__ == "__main__":
    unittest.main()
