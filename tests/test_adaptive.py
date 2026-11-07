import math
from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from p3.app import ImageRequest, Job, Worker
from p3.generate import GeneratedImage, GenerationError, pick_profile


class AdaptiveTests(unittest.TestCase):
    def worker(self, outcomes=None, device="cuda"):
        worker = Worker.__new__(Worker)
        worker.lock = Lock()
        worker.budget = 120
        worker.estimates = {"quality": 7.5, "balanced": 5.5, "eco": 3.5, "cpu": 90.0}
        worker.quality_retry_after = 0
        calls = []
        outcomes = list(outcomes or [True])

        def generate(prompt, seed, profile, **kwargs):
            calls.append(profile.name)
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return GeneratedImage(Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16)),
                                  {"generation_ms": 1000, "steps": profile.steps, "cv": {"usable": result}})

        worker.generator = SimpleNamespace(device=device, free_bytes=lambda: 3_000_000_000, generate=generate)
        return worker, calls

    def run_job(self, worker, mode="auto", created=100):
        job = Job(created=created)
        with patch("p3.app.perf_counter", return_value=100), patch("p3.app._webp", return_value="fixture"), patch("torch.cuda.empty_cache"):
            worker._run(job, ImageRequest(prompt="A green pear", seed=38, mode=mode))
        return job

    def test_failed_warmup_recovers_after_cooldown_with_default_budget(self):
        worker, calls = self.worker()
        worker.budget = 10
        with patch("p3.app.perf_counter", return_value=60):
            worker._defer_quality()
        self.assertTrue(math.isfinite(worker.estimates["quality"]))
        job = self.run_job(worker)
        self.assertEqual(calls, ["Full quality"])
        self.assertEqual(job.status, "complete")
        self.assertEqual(worker.quality_retry_after, 0)

    def test_cooldown_downgrades_auto_but_explicit_quality_can_retry(self):
        worker, calls = self.worker([True, True])
        with patch("p3.app.perf_counter", return_value=100):
            worker._defer_quality()
        self.run_job(worker)
        self.run_job(worker, "quality")
        self.assertEqual(calls, ["Balanced", "Full quality"])

    def test_nonfinite_old_estimate_can_recalibrate(self):
        worker, _ = self.worker()
        worker.estimates["quality"] = float("inf")
        worker._observe("quality", 2)
        self.assertTrue(math.isfinite(worker.estimates["quality"]))

    def test_memory_time_and_cpu_selection(self):
        worker, _ = self.worker()
        estimate = worker.estimates.__getitem__
        for device, memory, budget, expected in [
            ("cuda", 3_000_000_000, 6, "balanced"), ("cuda", 3_000_000_000, 4, "eco"),
            ("cuda", 1_000_000_000, 120, "eco"), ("cpu", 0, 120, "cpu")]:
            self.assertEqual(pick_profile("quality", device, memory, budget, estimate)[0], expected)
        for mode, device, budget in [("auto", "cuda", 2), ("eco", "cuda", 2), ("auto", "cpu", 10)]:
            with self.assertRaises(GenerationError):
                pick_profile(mode, device, 3_000_000_000, budget, estimate)
        worker.estimates["quality"] = 1
        self.assertEqual(pick_profile("auto", "cuda", 3_000_000_000, 2, estimate)[0], "quality")

    def test_oom_has_one_smaller_retry(self):
        worker, calls = self.worker([torch.OutOfMemoryError(), True])
        job = self.run_job(worker)
        self.assertEqual(calls, ["Full quality", "Low resource"])
        self.assertEqual(job.result["metadata"]["attempts"], 2)

    def test_second_oom_stops(self):
        worker, calls = self.worker([torch.OutOfMemoryError(), torch.OutOfMemoryError()])
        self.assertEqual(self.run_job(worker).status, "failed")
        self.assertEqual(len(calls), 2)

    def test_no_budget_starts_no_inference(self):
        worker, calls = self.worker()
        worker.budget = 2
        self.assertEqual(self.run_job(worker).status, "failed")
        self.assertEqual(calls, [])

    def test_quality_gate_cannot_add_third_attempt_after_oom(self):
        worker, calls = self.worker([torch.OutOfMemoryError(), False])
        self.assertEqual(self.run_job(worker).status, "failed")
        self.assertEqual(len(calls), 2)
