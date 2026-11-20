"""Phase 4: build a weight-only quantized U-Net variant of a browser export.

MatMul weights (attention and linear projections) become ONNX Runtime
`MatMulNBits` blocks (8-bit by default); convolutions stay FP16. The result is a
complete, separate export directory with its own manifest, so the FP16 release
is never modified. The script then checks numerical agreement on the export's
own reference inputs before anything is published.

    pip install onnx onnxruntime onnx-ir
    python -m p3.quantize_browser --export .models/browser/v3 --out .models/browser/v3-int8

Then test on desktop WebGPU first, then on the phone:
    P3_BROWSER_MODELS=.models/browser/v3-int8 npm run build   (in browser/)

Publish it only if WebGPU runs `MatMulNBits` correctly and the phone shows a
real storage or memory gain without unacceptable quality loss. Otherwise keep
FP16 and record this report as the rejected experiment.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx.external_data_helper import set_external_data

SHARD_BYTES = 64 * 1024 * 1024


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard(model: onnx.ModelProto, graph_path: Path, maximum: int = SHARD_BYTES) -> list[str]:
    """Move large initializers into ~64 MiB external files, like export_browser.py."""
    files, stream, offset = [], None, 0
    for tensor in model.graph.initializer:
        if len(tensor.raw_data) < 1024:
            continue
        size = len(tensor.raw_data)
        if stream is None or offset + size > maximum:
            if stream:
                stream.close()
            files.append(f"{graph_path.stem}-{len(files):02d}.bin")
            stream = (graph_path.parent / files[-1]).open("wb")
            offset = 0
        stream.write(tensor.raw_data)
        set_external_data(tensor, location=files[-1], offset=offset, length=size)
        tensor.ClearField("raw_data")
        offset += size
    if stream:
        stream.close()
    onnx.save_model(model, graph_path)
    return files


def quantize(model: onnx.ModelProto, bits: int, block_size: int) -> onnx.ModelProto:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(model, bits=bits, block_size=block_size, is_symmetric=True,
                                     op_types_to_quantize=("MatMul",))
    quantizer.process()
    return quantizer.model.model


def run_reference(graph: Path, reference: dict, root: Path) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    feeds = {}
    for name, spec in reference["inputs"].items():
        feeds[name] = np.fromfile(root / spec["path"], dtype=spec["dtype"]).reshape(spec["shape"])
    return session.run(None, feeds)[0].astype(np.float32)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    return {
        "relative_rmse": float(np.sqrt(np.mean(diff**2)) / max(1e-12, np.sqrt(np.mean(b**2)))),
        "max_abs": float(np.max(np.abs(diff))),
        "finite": bool(np.isfinite(a).all()),
    }


def build(export: Path, out: Path, bits: int, block_size: int) -> dict:
    if out.exists():
        raise SystemExit("Use a new output directory; the FP16 export is never overwritten.")
    manifest = json.loads((export / "manifest.json").read_text())
    unet = manifest["components"]["unet"]
    shutil.copytree(export, out, ignore=shutil.ignore_patterns("unet*.onnx", "unet-[0-9]*.bin"))
    model = onnx.load(str(export / unet["graph"]))  # loads external weights from the export
    matmuls = sum(node.op_type == "MatMul" for node in model.graph.node)
    quantized = quantize(model, bits, block_size)
    graph = out / "unet.onnx"
    externals = shard(quantized, graph)
    files = [{"path": name, "bytes": (out / name).stat().st_size, "sha256": sha(out / name)}
             for name in [graph.name, *externals]]
    original_bytes = sum(f["bytes"] for f in unet["files"])
    report = {
        "method": f"weight-only MatMulNBits, {bits}-bit, block {block_size}, symmetric; convolutions stay FP16",
        "matmul_nodes": matmuls,
        "operators": sorted({node.op_type for node in quantized.graph.node}),
        "unet_bytes": {"fp16": original_bytes, "quantized": sum(f["bytes"] for f in files)},
    }
    reference = unet.get("reference")
    if reference:
        expected = np.fromfile(export / reference["output"]["path"], dtype="<f4").reshape(reference["output"]["shape"])
        fp16 = run_reference(export / unet["graph"], reference, export)
        q = run_reference(graph, reference, export)
        report["agreement"] = {"quantized_vs_fp16_onnx": compare(q, fp16), "quantized_vs_pytorch_reference": compare(q, expected),
                               "fp16_onnx_vs_pytorch_reference": compare(fp16, expected)}
    manifest["components"]["unet"] = {**unet, "files": files, "operators": report["operators"],
                                      "precision": f"{bits}-bit weight-only MatMul, FP16 convolutions"}
    manifest["quantization"] = report
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "quantize-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="Existing FP16 browser export directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(build(args.export, args.out, args.bits, args.block_size), indent=2))


if __name__ == "__main__":
    main()
