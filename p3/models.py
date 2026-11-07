"""Download only the files consumed by each pinned inference bundle."""

from dataclasses import dataclass
from pathlib import Path
import os

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class Checkpoint:
    repo: str
    revision: str


CHECKPOINTS = {
    "tiny": Checkpoint("nota-ai/bk-sdm-tiny-2m", "aad3e0e8ba61b7cb9f64869dc4e586f8ad9d3665"),
    "baseline": Checkpoint("CompVis/stable-diffusion-v1-4", "133a221b8aa7292a167afc5127cb63fb5005638b"),
}

COMPONENTS = {
    "text_encoder": "model",
    "unet": "diffusion_pytorch_model",
    "vae": "diffusion_pytorch_model",
    "safety_checker": "model",
}

TAESD = Checkpoint("madebyollin/taesd", "614f76814bbe30edbe2e627ace1c2234c81a2c0e")


def models_root() -> Path:
    return Path(os.environ.get("P3_MODELS", ".models"))


def resolve_taesd() -> tuple[Path, int]:
    destination = models_root() / "taesd"
    files = ("config.json", "diffusion_pytorch_model.safetensors")
    if not all((destination / path).is_file() for path in files):
        snapshot_download(
            TAESD.repo, revision=TAESD.revision, local_dir=destination, allow_patterns=files, token=False
        )
    return destination, sum((destination / path).stat().st_size for path in files)


def model_files(precision: str, *, include_safety: bool = True) -> list[str]:
    suffix = ".fp16.safetensors" if precision == "fp16" else ".safetensors"
    files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
    ]
    if include_safety:
        files.append("feature_extractor/preprocessor_config.json")
    for component, filename in COMPONENTS.items():
        if component == "safety_checker" and not include_safety:
            continue
        files.extend((f"{component}/config.json", f"{component}/{filename}{suffix}"))
    return files


def resolve_model(key: str, precision: str, *, include_safety: bool = True) -> tuple[Path, int]:
    checkpoint = CHECKPOINTS[key]
    destination = models_root() / key / precision
    files = model_files(precision, include_safety=include_safety)

    if not all((destination / path).is_file() for path in files):
        snapshot_download(
            checkpoint.repo,
            revision=checkpoint.revision,
            local_dir=destination,
            allow_patterns=files,
            max_workers=4,
            token=False,
        )
    return destination, sum((destination / path).stat().st_size for path in files)


def load_fp32_components(directory: Path) -> dict:
    """Stream the original FP32 tensors to CUDA without a PyTorch file mapping."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from safetensors import safe_open
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel

    with init_empty_weights():
        components = {
            "unet": UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(directory / "unet")),
            "vae": AutoencoderKL.from_config(AutoencoderKL.load_config(directory / "vae")),
            "text_encoder": CLIPTextModel(CLIPTextConfig.from_pretrained(directory / "text_encoder", local_files_only=True)),
        }
    for name, module in components.items():
        file = directory / name / f"{COMPONENTS[name]}.safetensors"
        with safe_open(file, framework="np") as weights:
            source_names = {key: key for key in weights.keys()}
            if name == "vae":
                # Reuse the pinned Diffusers migration for the legacy VAE attention names.
                module._fix_state_dict_keys_on_load(source_names)
            for key in module.state_dict():
                tensor = torch.tensor(weights.get_tensor(source_names[key]), device="cuda")
                set_module_tensor_to_device(module, key, "cuda", value=tensor)
        module.eval()
    return components
