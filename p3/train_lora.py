"""Fine-tune only the BK-SDM-Tiny attention adapters on COCO train2017."""

import argparse
import hashlib
import json
import math
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
from p3.models import CHECKPOINTS, resolve_model


DEVICE = "cuda"
DTYPE = torch.float16
ACCUMULATE = 4
LEARNING_RATE = 1e-4
SEED = 38


def save_state(path: Path, data: dict):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


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
    if not torch.cuda.is_available():
        raise RuntimeError("COCO LoRA training requires a CUDA PyTorch build and a CUDA GPU.")
    samples = read_manifest(manifest)
    if any(sample.get("split") != "train2017" or "reference" not in sample for sample in samples):
        raise ValueError("Provide a COCO train2017 manifest with image paths. Validation rows are reserved for evaluation.")
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "checkpoint": CHECKPOINTS["tiny"].revision,
        "train_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "resolution": resolution,
        "rank": rank,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "accumulation": ACCUMULATE,
        "training_pairs": len(samples),
    }
    torch.manual_seed(SEED)
    started = perf_counter()
    model_dir, _ = resolve_model("tiny", "fp16")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True, variant="fp16",
        torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False,
    )
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    cache = work / "encodings.pt"
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=True)
        if stored["settings"] != settings:
            raise ValueError("Training input or settings changed. Start with a new --work directory.")
        encoded = stored["data"]
    else:
        encoded = encode_training_set(pipe, manifest, samples, resolution, SEED)
        save_state(cache, {"settings": settings, "data": encoded})
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
    checkpoint = work / "resume.pt"
    offset, updates, losses = 0, 0, []
    if checkpoint.is_file():
        resume = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if resume["settings"] != settings:
            raise ValueError("The resume checkpoint does not match this training run.")
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
            save_state(checkpoint, {
                "settings": settings, "offset": offset, "updates": updates, "losses": losses,
                "adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng.get_state(),
            })
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
