"""Convert a pinned model to the original SD layout used by the Android runtime."""

import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from p3.models import CHECKPOINTS, COMPONENTS, resolve_model
from p3.provenance import file_hash, software_identity, source_hashes, write_json


CONVERTER_SHA256 = "749b62b86db37b539e64f00f286e8f8395adf73ed49097764de4df8c2bab0df6"


def fuse_adapter(source: Path, precision: str, adapter: Path, staged: Path) -> dict:
    """Use the same Diffusers/PEFT fusion as the hosted generator, at the same dtype."""
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from safetensors import safe_open
    import torch
    dtype = torch.float16 if precision == "fp16" else torch.float32
    unet = UNet2DConditionModel.from_pretrained(
        source / "unet", local_files_only=True, use_safetensors=True,
        variant="fp16" if precision == "fp16" else None, torch_dtype=dtype,
    )
    state, alphas = StableDiffusionPipeline.lora_state_dict(
        str(adapter.parent), weight_name=adapter.name, local_files_only=True
    )
    if not state or any(not key.startswith("unet.") for key in state):
        raise ValueError("Mobile export supports U-Net LoRA only; refusing to silently drop other components.")
    StableDiffusionPipeline.load_lora_into_unet(state, network_alphas=alphas, unet=unet)
    unet.fuse_lora(safe_fusing=True)
    unet.unload_lora()
    base_name = "diffusion_pytorch_model" + (".fp16" if precision == "fp16" else "") + ".safetensors"
    changed = []
    with safe_open(source / "unet" / base_name, framework="pt", device="cpu") as base:
        for name, tensor in unet.state_dict().items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite fused tensor {name}")
            if not torch.equal(tensor.cpu(), base.get_tensor(name)):
                changed.append(name)
    if not changed:
        raise ValueError("Adapter fusion changed no tensors; refusing an unadapted export.")
    # Never save through a hard link to the original model file.
    unet.save_pretrained(staged / "unet", safe_serialization=True)
    del unet, state
    gc.collect()
    return {"sha256": file_hash(adapter), "name": adapter.name, "scale": 1.0,
            "fusion": "Diffusers/PEFT safe fusion", "changed_tensors": changed}


def verify_conversion(staged: Path, output: Path, converter: Path) -> dict:
    """Compare every mapped tensor against the staged source, including fused weights."""
    import numpy as np
    from safetensors import safe_open
    functions = runpy.run_path(str(converter))
    prefixes = {"unet": "model.diffusion_model.", "vae": "first_stage_model.",
                "text_encoder": "cond_stage_model.transformer."}
    converters = {"unet": functions["convert_unet_state_dict"],
                  "vae": functions["convert_vae_state_dict"], "text_encoder": functions["convert_text_enc_state_dict"]}
    count = 0
    with safe_open(output, framework="np") as result:
        remaining = set(result.keys())
        for component, stem in COMPONENTS.items():
            if component not in prefixes:
                continue
            with safe_open(staged / component / f"{stem}.safetensors", framework="np") as source:
                mapped = converters[component]({key: source.get_tensor(key) for key in source.keys()})
                for name, expected in mapped.items():
                    key = prefixes[component] + name
                    actual = result.get_tensor(key)
                    if not np.isfinite(actual).all() or not np.array_equal(actual, expected.astype(actual.dtype)):
                        raise ValueError(f"Export tensor mismatch: {key}")
                    remaining.remove(key)
                    count += 1
                del mapped
        if remaining:
            raise ValueError("Export contains unexpected tensors")
    return {"mapped_tensors_verified": count, "finite": True, "decoded_validation": "pending native render"}


def export(key: str, precision: str, converter: Path, output: Path, adapter: Path | None = None):
    if file_hash(converter) != CONVERTER_SHA256:
        raise ValueError("Use the pinned Diffusers v0.37.0 converter; its checksum did not match.")
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError("Choose a new export filename to preserve existing artifacts and provenance.")
    if adapter and key != "tiny":
        raise ValueError("The T38 adapter is trained for BK-SDM-Tiny; select --model tiny.")
    source, _ = resolve_model(key, precision, include_safety=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sd-export-", dir=output.parent) as scratch:
        staged = Path(scratch)
        for component in ("unet", "vae", "text_encoder"):
            if component == "unet" and adapter is not None:
                continue
            directory = staged / component
            directory.mkdir()
            stem = COMPONENTS[component]
            filename = f"{stem}.fp16.safetensors" if precision == "fp16" else f"{stem}.safetensors"
            tensor_file = source / component / filename
            alias = directory / f"{stem}.safetensors"
            try:
                os.link(tensor_file, alias)
            except OSError:
                shutil.copyfile(tensor_file, alias)
        adaptation = fuse_adapter(source, precision, adapter, staged) if adapter else None
        temporary_output = staged / "converted.safetensors"
        command = [
            sys.executable, str(converter), "--model_path", str(staged),
            "--checkpoint_path", str(temporary_output), "--use_safetensors",
        ]
        if precision == "fp16":
            command.append("--half")
        subprocess.run(command, check=True)
        verification = verify_conversion(staged, temporary_output, converter)
        provenance = {"schema": 2, "checkpoint": asdict(CHECKPOINTS[key]), "precision": precision,
                      "adapter": adaptation, "scope": "diffusion weights; no safety checker or classical CV",
                      "source_weights": {component: file_hash(staged / component / f"{COMPONENTS[component]}.safetensors")
                                         for component in ("unet", "vae", "text_encoder")},
                      "converter_sha256": file_hash(converter), "software": software_identity(),
                      "code": source_hashes("export_mobile.py"), "validation": verification,
                      "artifact_sha256": file_hash(temporary_output), "artifact_bytes": temporary_output.stat().st_size}
        os.replace(temporary_output, output)
        write_json(output.with_suffix(output.suffix + ".json"), provenance)
        print(f"Verified {verification['mapped_tensors_verified']} tensors; adapter fused: {bool(adaptation)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=CHECKPOINTS)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, help="Fuse the trained Tiny U-Net LoRA before conversion.")
    args = parser.parse_args()
    export(args.model, args.precision, args.converter, args.out, args.adapter)
