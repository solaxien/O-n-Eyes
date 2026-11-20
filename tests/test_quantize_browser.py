import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAVE_ONNX = all(importlib.util.find_spec(m) for m in ("onnx", "onnxruntime", "onnx_ir"))


def toy_export(root: Path):
    """A tiny export with the browser U-Net's input/output names and a real MatMul."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from p3.quantize_browser import shard

    rng = np.random.default_rng(0)
    weight = numpy_helper.from_array((rng.standard_normal((768, 64)) * 0.05).astype(np.float16), "W")
    nodes = [
        helper.make_node("MatMul", ["encoder_hidden_states", "W"], ["context"]),
        helper.make_node("ReduceMean", ["context"], ["pooled"], axes=[1, 2], keepdims=1),
        helper.make_node("Cast", ["pooled"], ["pooled32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["sample"], ["sample32"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["sample32", "timestep"], ["scaled"]),
        helper.make_node("Add", ["scaled", "pooled32"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "unet", [
        helper.make_tensor_value_info("sample", TensorProto.FLOAT16, [1, 4, 8, 8]),
        helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT16, [1, 77, 768]),
    ], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    files = shard(model, root / "unet.onnx")
    inputs = {"sample": rng.standard_normal((1, 4, 8, 8)).astype(np.float16),
              "timestep": np.array([500], np.float32),
              "encoder_hidden_states": rng.standard_normal((1, 77, 768)).astype(np.float16)}
    reference = {"inputs": {}, "output": {"path": "unet-expected.f32", "shape": [1, 4, 8, 8]}}
    for name, array in inputs.items():
        array.tofile(root / f"unet-{name}.tensor")
        reference["inputs"][name] = {"path": f"unet-{name}.tensor", "shape": list(array.shape), "dtype": str(array.dtype)}
    import onnxruntime as ort
    expected = ort.InferenceSession(str(root / "unet.onnx")).run(None, inputs)[0]
    expected.astype("<f4").tofile(root / "unet-expected.f32")
    unet = {"graph": "unet.onnx", "reference": reference,
            "files": [{"path": n, "bytes": (root / n).stat().st_size, "sha256": "x"} for n in ["unet.onnx", *files]]}
    (root / "manifest.json").write_text(json.dumps({"components": {"text": {}, "unet": unet, "decoder": {}}}))
    (root / "vocab.json").write_text("{}")


@unittest.skipUnless(HAVE_ONNX, "onnx, onnxruntime and onnx-ir are needed")
class QuantizeBrowserTests(unittest.TestCase):
    def test_quantized_export_is_separate_sharded_and_checked(self):
        from p3.quantize_browser import build

        with tempfile.TemporaryDirectory() as tmp:
            export, out = Path(tmp) / "fp16", Path(tmp) / "int8"
            export.mkdir()
            toy_export(export)
            report = build(export, out, bits=8, block_size=32)
            self.assertIn("MatMulNBits", report["operators"])
            self.assertLess(report["unet_bytes"]["quantized"], report["unet_bytes"]["fp16"])
            self.assertLess(report["agreement"]["quantized_vs_fp16_onnx"]["relative_rmse"], 0.02)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertTrue(all((out / f["path"]).exists() for f in manifest["components"]["unet"]["files"]))
            self.assertTrue((out / "vocab.json").exists())
            self.assertTrue((export / "unet.onnx").exists(), "FP16 export is untouched")
            with self.assertRaises(SystemExit):
                build(export, out, bits=8, block_size=32)


if __name__ == "__main__":
    unittest.main()
