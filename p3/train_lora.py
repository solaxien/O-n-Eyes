"""Fine-tune only the BK-SDM-Tiny attention adapters on COCO train2017."""

import argparse
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torchvision.transforms.functional import pil_to_tensor

from p3.benchmark import read_manifest, write_json
from p3.models import CHECKPOINTS, model_files, resolve_model
from p3.provenance import (atomic_bytes, digest, establish_protocol, exclusive_run, file_hash,
                           read_json, runtime_identity, session_id, source_hashes)


DEVICE = "cuda"
DTYPE = torch.float16
ACCUMULATE = 4
LEARNING_RATE = 1e-4
SEED = 38


def save_state(path: Path, data: dict):
    buffer = BytesIO()
    torch.save(data, buffer)
    atomic_bytes(path, buffer.getvalue())


def training_protocol(manifest, samples, model_dir, resolution, rank, runtime=None):
    images = {}
    for row in samples:
        path = manifest.parent / row["reference"]
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        images[row["id"]] = file_hash(path)
    return {
        "schema": 2, "kind": "lora-training", "checkpoint": CHECKPOINTS["tiny"].revision,
        "manifest": samples, "train_manifest_sha256": file_hash(manifest), "image_sha256": images,
        "source_files": {name: file_hash(model_dir / name) for name in model_files("fp16", include_safety=False)},
        "runtime": runtime if runtime is not None else runtime_identity(DEVICE),
        "code": source_hashes("train_lora.py", "models.py", "provenance.py"),
        "resolution": resolution, "rank": rank, "seed": SEED, "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATE, "training_pairs": len(samples), "dtype": "fp16",
        "encoding": "RGB/Lanczos center fit; VAE posterior sample; CLIP max_length=77",
    }


def validate_encodings(data: dict, count: int, resolution: int):
    shapes = {"latents": (count, 4, resolution // 8, resolution // 8),
              "embeddings": (count, 77, 768), "empty": (1, 77, 768)}
    if set(data) != set(shapes):
        raise ValueError("Encoding cache has unexpected fields")
    for key, shape in shapes.items():
        value = data[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != DTYPE or not torch.isfinite(value).all():
            raise ValueError(f"Invalid cached {key} shape, dtype, or values")


def load_checked_state(path, metadata_path, fingerprint):
    proof = read_json(metadata_path)
    if proof.get("protocol_hash") != fingerprint or proof.get("sha256") != file_hash(path):
        raise ValueError("Training cache/checkpoint integrity mismatch. Preserve it and use a new --work directory.")
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def encode_training_set(pipe, manifest: Path, samples: list[dict], resolution: int, seed: int) -> dict:
    pipe.text_encoder.to(DEVICE).eval()
    pipe.vae.to(DEVICE).eval()
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    latents, embeddings = [], []
    for index in range(0, len(samples), 4):
        batch = samples[index:index + 4]
        pixels = []
        for sample in batch:
            with Image.open(manifest.parent / sample["reference"]) as opened:
                photo = ImageOps.fit(opened.convert("RGB"), (resolution, resolution), Image.Resampling.LANCZOS)
                pixels.append(pil_to_tensor(photo))
        images = torch.stack(pixels).to(DEVICE, DTYPE) / 127.5 - 1
        encoded = pipe.vae.encode(images).latent_dist.sample(generator=rng) * pipe.vae.config.scaling_factor
        tokens = pipe.tokenizer(
            [sample["prompt"] for sample in batch], max_length=pipe.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.to(DEVICE)
        latents.append(encoded.cpu())
        embeddings.append(pipe.text_encoder(tokens)[0].cpu())
        if (index + len(batch)) % 100 == 0:
            print(f"Encoded {index + len(batch)}/{len(samples)} training pairs", flush=True)
    empty_tokens = pipe.tokenizer(
        "", max_length=pipe.tokenizer.model_max_length, padding="max_length", return_tensors="pt"
    ).input_ids.to(DEVICE)
    empty = pipe.text_encoder(empty_tokens)[0].cpu()
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    return {"latents": torch.cat(latents), "embeddings": torch.cat(embeddings), "empty": empty}


def train(manifest: Path, output: Path, work: Path, resolution: int, rank: int):
    if output.exists():
        raise FileExistsError("Choose a new --out filename; existing trained adapters are preserved.")
    with exclusive_run(output.parent / ".training-lock"), exclusive_run(work):
        if output.exists():
            raise FileExistsError("An adapter already exists at --out")
        _train(manifest, output, work, resolution, rank)


def _train(manifest: Path, output: Path, work: Path, resolution: int, rank: int):
    if not torch.cuda.is_available():
        raise RuntimeError("COCO LoRA training requires a CUDA PyTorch build and a CUDA GPU.")
    samples = read_manifest(manifest)
    if any(sample.get("split") != "train2017" or "reference" not in sample for sample in samples):
        raise ValueError("Provide a COCO train2017 manifest with image paths. Validation rows are reserved for evaluation.")
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_dir, _ = resolve_model("tiny", "fp16", include_safety=False)
    settings = training_protocol(manifest, samples, model_dir, resolution, rank)
    fingerprint = establish_protocol(work, settings)
    torch.manual_seed(SEED)
    started = perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True, variant="fp16",
        torch_dtype=DTYPE, safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    cache = work / "encodings.pt"
    if cache.exists():
        encoded = load_checked_state(cache, work / "encodings.json", fingerprint)["data"]
    else:
        encoded = encode_training_set(pipe, manifest, samples, resolution, SEED)
        validate_encodings(encoded, len(samples), resolution)
        save_state(cache, {"settings": settings, "data": encoded})
        write_json(work / "encodings.json", {"protocol_hash": fingerprint, "sha256": file_hash(cache)})
    validate_encodings(encoded, len(samples), resolution)
    model = pipe.unet
    del pipe
    torch.cuda.empty_cache()
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    ))
    model.to(DEVICE).train()
    model.enable_gradient_checkpointing()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(SEED)).tolist()
    rng = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    checkpoint = work / "resume.json"
    offset, updates, losses = 0, 0, []
    if checkpoint.is_file():
        pointer = read_json(checkpoint)
        name = pointer["file"]
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint path")
        resume = load_checked_state(work / "checkpoints" / name, checkpoint, fingerprint)
        if resume["settings"] != settings:
            raise ValueError("The resume checkpoint does not match this training run.")
        expected = get_peft_model_state_dict(model)
        if set(resume["adapter"]) != set(expected) or any(
            value.shape != expected[key].shape or not torch.isfinite(value).all()
            for key, value in resume["adapter"].items()
        ):
            raise ValueError("Invalid resumed adapter tensors")
        if not 0 <= resume["offset"] <= len(samples) or resume["updates"] != math.ceil(resume["offset"] / ACCUMULATE):
            raise ValueError("Invalid training progress checkpoint")
        set_peft_model_state_dict(model, resume["adapter"])
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        rng.set_state(resume["rng"])
        offset, updates, losses = resume["offset"], resume["updates"], resume["losses"]

    while offset < len(order):
        batch = order[offset:offset + ACCUMULATE]
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index in batch:
            latent = encoded["latents"][index:index + 1].to(DEVICE)
            if torch.rand((), generator=rng, device=DEVICE).item() < 0.1:
                conditioning = encoded["empty"].to(DEVICE)
            else:
                conditioning = encoded["embeddings"][index:index + 1].to(DEVICE)
            noise = torch.randn(latent.shape, device=DEVICE, dtype=DTYPE, generator=rng)
            timestep = torch.randint(noise_scheduler.config.num_train_timesteps, (1,), device=DEVICE, generator=rng)
            noisy = noise_scheduler.add_noise(latent, noise, timestep)
            target = noise_scheduler.get_velocity(latent, noise, timestep) if noise_scheduler.config.prediction_type == "v_prediction" else noise
            with torch.autocast("cuda", dtype=DTYPE):
                predicted = model(noisy, timestep, encoder_hidden_states=conditioning, return_dict=False)[0]
                loss = F.mse_loss(predicted.float(), target.float())
            value = loss.detach().item()
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite training loss at sample {offset}.")
            total_loss += value
            scaler.scale(loss / len(batch)).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        offset += len(batch)
        updates += 1
        losses.append(total_loss / len(batch))
        if updates % 25 == 0 or offset == len(order):
            state_path = work / "checkpoints" / f"{updates}-{session_id()}.pt"
            save_state(state_path, {
                "settings": settings, "offset": offset, "updates": updates, "losses": losses,
                "adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng.get_state(),
            })
            write_json(checkpoint, {"file": state_path.name, "protocol_hash": fingerprint, "sha256": file_hash(state_path)})
            print(f"Update {updates} · {offset}/{len(order)} pairs · mean loss {sum(losses[-25:]) / len(losses[-25:]):.5f}", flush=True)

    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
    with TemporaryDirectory(prefix="adapter-", dir=output.parent) as scratch:
        StableDiffusionPipeline.save_lora_weights(
            scratch, unet_lora_layers=state, weight_name=output.name, safe_serialization=True
        )
        (Path(scratch) / output.name).replace(output)
    write_json(work / "training.json", settings | {
        "optimizer_updates": updates,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "first_25_mean_loss": sum(losses[:25]) / len(losses[:25]),
        "last_25_mean_loss": sum(losses[-25:]) / len(losses[-25:]),
        "device": torch.cuda.get_device_name(),
        "seconds_this_invocation": perf_counter() - started,
        "adapter_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work", default=Path(".runs/training"), type=Path)
    parser.add_argument("--resolution", default=384, type=int, choices=(384, 512))
    parser.add_argument("--rank", default=4, type=int, choices=(4, 8))
    args = parser.parse_args()
    train(args.manifest, args.out, args.work, args.resolution, args.rank)
