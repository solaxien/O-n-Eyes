from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from p3.fetch_coco import prepare, valid_reference
from p3.models import model_files
from p3.provenance import digest, establish_protocol, exclusive_run, file_hash, write_json
from p3.train_lora import load_checked_state, save_state, training_protocol, validate_encodings


class TrainingCacheTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".runs/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_replaced_image_weight_and_code_change_training_identity(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text('fixture')
        Image.new("RGB", (32, 32), "red").save(self.root / "image.png")
        rows = [{"id": "sample", "reference": "image.png"}]
        model = self.root / "weights"
        for name in model_files("fp16", include_safety=False):
            path = model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture weights")
        protocol = lambda: training_protocol(manifest, rows, model, 384, 4, runtime={"test": "cpu"})
        original = protocol()
        Image.new("RGB", (32, 32), "blue").save(self.root / "image.png")
        self.assertNotEqual(digest(original), digest(protocol()))
        work = self.root / "work"
        with exclusive_run(work):
            establish_protocol(work, original)
            with self.assertRaisesRegex(ValueError, "Protocol changed"):
                establish_protocol(work, protocol())
        current = protocol()
        (model / "unet/config.json").write_bytes(b"changed config")
        self.assertNotEqual(digest(current), digest(protocol()))
        with patch("p3.train_lora.source_hashes", return_value={"training": "changed"}):
            self.assertNotEqual(digest(current), digest(protocol()))

    def test_cached_shape_dtype_and_nonfinite_values_are_rejected(self):
        data = {"latents": torch.zeros(1,4,48,48,dtype=torch.float16),
                "embeddings": torch.zeros(1,77,768,dtype=torch.float16),
                "empty": torch.zeros(1,77,768,dtype=torch.float16)}
        validate_encodings(data, 1, 384)
        with self.assertRaises(ValueError):
            validate_encodings(data, 2, 384)
        data["latents"][0,0,0,0] = float("nan")
        with self.assertRaises(ValueError):
            validate_encodings(data, 1, 384)

    def test_tampered_checkpoint_is_rejected(self):
        path = self.root / "state.pt"
        proof = self.root / "state.json"
        save_state(path, {"weight": torch.ones(2)})
        write_json(proof, {"protocol_hash": "test", "sha256": file_hash(path)})
        self.assertEqual(load_checked_state(path, proof, "test")["weight"].sum(), 2)
        path.write_bytes(b"broken tensor file")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            load_checked_state(path, proof, "test")

    def test_corrupt_reference_is_downloaded_and_repaired(self):
        import json
        row = {"id": "coco-000000000001", "split": "val2017", "source_file": "val2017/000000000001.jpg"}
        (self.root / "manifest.jsonl").write_text(json.dumps(row) + "\n")
        target = self.root / "reference" / (row["id"] + ".png")
        target.parent.mkdir()
        target.write_bytes(b"truncated PNG")
        content = BytesIO()
        Image.new("RGB", (100, 80), "red").save(content, "JPEG")
        with patch("p3.fetch_coco.fetch", return_value=SimpleNamespace(content=content.getvalue())) as fetch:
            prepare("validation", 1, self.root, 1)
            self.assertTrue(valid_reference(target))
            self.assertEqual(fetch.call_count, 1)
            prepare("validation", 1, self.root, 1)
            self.assertEqual(fetch.call_count, 1)
