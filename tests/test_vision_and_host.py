from pathlib import Path
from types import SimpleNamespace
import unittest

from fastapi import HTTPException
import numpy as np
from PIL import Image

from p3.app import ready
from p3.check_environment import verify_gpu
from p3.vision import enhance, resize


class VisionAndHostingTests(unittest.TestCase):
    def test_speckle_repair_reduces_known_pixel_error(self):
        clean = np.tile(np.linspace(50, 190, 128, dtype=np.uint8), (96, 1))
        clean = np.repeat(clean[:, :, None], 3, axis=2)
        corrupt = clean.copy()
        points = [(20, 20), (25, 40), (40, 60), (60, 80), (75, 100)]
        for y, x in points:
            corrupt[y, x] = 255
        result = enhance(Image.fromarray(corrupt))
        actual = np.asarray(result.image).astype(float)
        before = np.mean([np.abs(corrupt[y, x].astype(float) - clean[y, x]).mean() for y, x in points])
        after = np.mean([np.abs(actual[y, x] - clean[y, x]).mean() for y, x in points])
        self.assertTrue(result.usable)
        self.assertLess(after, before * 0.1)
        self.assertIn("Speckle repair", result.applied)

    def test_clean_gradient_is_preserved(self):
        clean = np.tile(np.linspace(40, 210, 128, dtype=np.uint8), (96, 1))
        clean = np.repeat(clean[:, :, None], 3, axis=2)
        result = enhance(Image.fromarray(clean))
        self.assertTrue(result.usable)
        self.assertEqual(result.artifact_fraction, 0)
        self.assertLessEqual(np.abs(np.asarray(result.image).astype(float) - clean).max(), 1)

    def test_uniform_failure_retains_raw_pixels(self):
        raw = Image.new("RGB", (64, 64), "white")
        result = enhance(raw)
        self.assertFalse(result.usable)
        self.assertEqual(result.image.tobytes(), raw.tobytes())

    def test_resize_preserves_ratio(self):
        raw = Image.new("RGB", (128, 96))
        self.assertEqual(resize(raw, 512).size, (512, 384))
        self.assertEqual(resize(raw, 64).size, (64, 48))

    def test_readiness_fails_until_worker_is_ready(self):
        for status in ("loading", "failed", "ready"):
            request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
                worker=SimpleNamespace(health=lambda: {"status": status}))))
            if status == "ready":
                self.assertEqual(ready(request), {"status": "ready"})
            else:
                with self.assertRaises(HTTPException) as failure:
                    ready(request)
                self.assertEqual(failure.exception.status_code, 503)

    def test_hardware_cannot_be_relabelled(self):
        identity = {"gpu": {"name": "NVIDIA GeForce RTX 4050 Laptop GPU"}}
        with self.assertRaises(ValueError):
            verify_gpu(identity, "Tesla T4")
        verify_gpu(identity, identity["gpu"]["name"])
