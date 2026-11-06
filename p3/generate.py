"""Seeded latent diffusion with explicit profiling of each inference stage."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import os

from PIL import Image
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler, StableDiffusionPipeline

from p3.models import CHECKPOINTS, load_fp32_components, resolve_model, resolve_taesd
from p3.vision import enhance, resize


class GenerationError(Exception):
    """A generation failure suitable for display to the caller."""


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    size: int
    scheduler: str = "dpmpp"


PROFILES = {
    "balanced": Profile("Balanced", 8, 512),
    "quality": Profile("Full quality", 20, 512),
    "eco": Profile("Low resource", 6, 384),
    "cpu": Profile("CPU fallback", 5, 320),
}


def pick_profile(
    mode: str,
    device: str,
    free_bytes: int,
    remaining: float,
    estimate: Callable[[str], float],
) -> tuple[str, str]:
    if device == "cpu":
        return "cpu", "CPU fallback selected because a CUDA worker is unavailable."
    if mode == "eco":
        return "eco", "Low resource mode selected."
    requested = "quality" if mode == "auto" else mode
    if free_bytes < 1_500_000_000:
        return "eco", "Reduced resolution and steps for available GPU memory."
    if remaining < estimate(requested):
        if requested == "quality" and remaining >= estimate("balanced"):
            return "balanced", "Balanced mode fits the remaining request budget."
        return "eco", "Reduced resolution and steps for the remaining request budget."
    return requested, "Selected profile fits the current inference budget."


@dataclass
class GeneratedImage:
    raw: Image.Image
    image: Image.Image
    metadata: dict


Progress = Callable[[str, int, int], None]


def default_device() -> str:
    device = os.environ.get("P3_DEVICE", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("P3_DEVICE must be auto, cuda, or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA PyTorch build or set P3_DEVICE=cpu.")
    return device


class Generator:
    def __init__(
        self,
        checkpoint: str = "tiny",
        precision: str = "fp16",
        device: str = "cuda",
        *,
        decoder: str = "vae",
        adapter: Path | None = None,
        weight_precision: str | None = None,
        check_safety: bool = True,
    ):
        if device == "cpu" and precision == "fp16":
            raise ValueError("Use FP32 on the CPU")
        if decoder not in {"vae", "taesd"}:
            raise ValueError("Choose vae or taesd for the decoder")
        self.checkpoint = checkpoint
        self.device = device
        self.precision = precision
        self.weight_precision = weight_precision or precision
        self.decoder = decoder
        self.adapter = adapter
        self.check_safety = check_safety
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        if device == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 8))
        model_dir, self.model_bytes = resolve_model(checkpoint, self.weight_precision, include_safety=check_safety)
        load_options = {} if check_safety else {
            "safety_checker": None, "feature_extractor": None, "requires_safety_checker": False,
        }
        started = perf_counter()
        if os.name == "nt" and device == "cuda" and precision == self.weight_precision == "fp32" and not check_safety:
            load_options.update(load_fp32_components(model_dir))
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=True,
            variant="fp16" if self.weight_precision == "fp16" else None,
            torch_dtype=self.dtype,
            **load_options,
        )
        if decoder == "taesd":
            taesd_dir, taesd_bytes = resolve_taesd()
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                taesd_dir, local_files_only=True, use_safetensors=True, torch_dtype=self.dtype
            )
            self.model_bytes += taesd_bytes - sum(
                path.stat().st_size for path in (model_dir / "vae").iterdir() if path.is_file()
            )
        if adapter is not None:
            if not adapter.is_file() or adapter.suffix != ".safetensors":
                raise ValueError("P3_ADAPTER must point to a local LoRA .safetensors file.")
            self.pipe.load_lora_weights(str(adapter))
            self.pipe.fuse_lora(safe_fusing=True)
            self.pipe.unload_lora_weights()
            self.model_bytes += adapter.stat().st_size
        self.pipe.to(device)
        self.dpm_scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2
        )
        self.hardware = torch.cuda.get_device_name() if device == "cuda" else "CPU"
        self.load_ms = self._elapsed(started)

    def _elapsed(self, started: float) -> float:
        if self.device == "cuda":
            torch.cuda.synchronize()
        return round((perf_counter() - started) * 1000, 2)

    def free_bytes(self) -> int:
        if self.device == "cpu":
            return 0
        free, _ = torch.cuda.mem_get_info()
        return free + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        profile: Profile,
        *,
        process_image: bool = True,
        progress: Progress | None = None,
    ) -> GeneratedImage:
        start = perf_counter()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pipe = self.pipe
        scheduler = pipe.scheduler if profile.scheduler == "checkpoint" else self.dpm_scheduler
        if progress:
            progress("Encoding prompt", 0, profile.steps)
        token_count = len(pipe.tokenizer(prompt, verbose=False).input_ids)
        if token_count > pipe.tokenizer.model_max_length:
            raise GenerationError(
                f"This prompt has {token_count} tokens. Keep it within {pipe.tokenizer.model_max_length} "
                "by shortening the style description and keeping the objects and relationships."
            )
        positive, negative = pipe.encode_prompt(
            prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True
        )
        embeddings = torch.cat((negative, positive))
        encode_ms = self._elapsed(start)

        denoise_start = perf_counter()
        scheduler.set_timesteps(profile.steps, device=self.device)
        denoiser_calls = len(scheduler.timesteps)
        # Sample FP32 on CPU, then cast. Paired runs start with the same noise
        # before any rounding required by the chosen inference precision.
        rng = torch.Generator(device="cpu").manual_seed(seed)
        side = profile.size // pipe.vae_scale_factor
        shape = (1, pipe.unet.config.in_channels, side, side)
        latents = torch.randn(shape, generator=rng, dtype=torch.float32)
        latents = latents.to(device=self.device, dtype=self.dtype) * scheduler.init_noise_sigma
        if progress:
            progress("Denoising", 0, denoiser_calls)
        for step, timestep in enumerate(scheduler.timesteps, start=1):
            batch = scheduler.scale_model_input(torch.cat((latents, latents)), timestep)
            predicted = pipe.unet(batch, timestep, encoder_hidden_states=embeddings, return_dict=False)[0]
            unconditional, conditional = predicted.chunk(2)
            guided = unconditional + 7.5 * (conditional - unconditional)
            latents = scheduler.step(guided, timestep, latents, return_dict=False)[0]
            if progress:
                progress("Denoising", step, denoiser_calls)
        denoise_ms = self._elapsed(denoise_start)

        if progress:
            progress("Decoding image", denoiser_calls, denoiser_calls)
        decode_start = perf_counter()
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        if not torch.isfinite(decoded).all().item():
            raise GenerationError("The decoder produced invalid pixels. Try a different seed.")
        if self.check_safety:
            decoded, blocked = pipe.run_safety_checker(decoded, self.device, embeddings.dtype)
            if blocked and blocked[0]:
                raise GenerationError("The checkpoint's safety checker blocked the result. Try a different prompt.")
        raw = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        decode_ms = self._elapsed(decode_start)

        if progress:
            progress("Checking image quality", denoiser_calls, denoiser_calls)
        cv_start = perf_counter()
        checked = enhance(raw) if process_image else None
        image = resize(checked.image if checked else raw, 512)
        cv_ms = self._elapsed(cv_start)
        metadata = {
            "model": CHECKPOINTS[self.checkpoint].repo,
            "revision": CHECKPOINTS[self.checkpoint].revision,
            "hardware": self.hardware,
            "backend": self.device,
            "precision": self.precision,
            "weight_precision": self.weight_precision,
            "decoder": self.decoder,
            "safety_check": self.check_safety,
            "adapter": self.adapter.name if self.adapter else None,
            "scheduler": type(scheduler).__name__,
            "profile": profile.name,
            "requested_steps": profile.steps,
            "steps": denoiser_calls,
            "native_size": profile.size,
            "display_size": image.size[0],
            "seed": seed,
            "model_bytes": self.model_bytes,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
            "stages_ms": {
                "encode": encode_ms,
                "denoise": denoise_ms,
                "decode_and_safety": decode_ms,
                "cv_and_resize": cv_ms,
            },
            "generation_ms": self._elapsed(start),
            "cv": {
                "applied": checked.applied,
                "before": checked.before,
                "after": checked.after,
                "artifact_fraction": checked.artifact_fraction,
                "usable": checked.usable,
            } if checked else None,
        }
        return GeneratedImage(raw, image, metadata)
