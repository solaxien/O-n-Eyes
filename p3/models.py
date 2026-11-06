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


def resolve_model(key: str, precision: str) -> tuple[Path, int]:
    checkpoint = CHECKPOINTS[key]
    destination = models_root() / key / precision
    suffix = ".fp16.safetensors" if precision == "fp16" else ".safetensors"
    files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "feature_extractor/preprocessor_config.json",
        "tokenizer/merges.txt",
        "tokenizer/vocab.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
    ]
    for component, filename in COMPONENTS.items():
        files.extend((f"{component}/config.json", f"{component}/{filename}{suffix}"))

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
