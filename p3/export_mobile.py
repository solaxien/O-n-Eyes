"""Convert a pinned model to the original SD layout used by the Android runtime."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from p3.models import CHECKPOINTS, COMPONENTS, resolve_model


def export(key: str, precision: str, converter: Path, output: Path):
    source, _ = resolve_model(key, precision)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="sd-export-", dir=output.parent) as scratch:
        staged = Path(scratch)
        for component in ("unet", "vae", "text_encoder"):
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
        command = [
            sys.executable, str(converter), "--model_path", str(staged),
            "--checkpoint_path", str(output), "--use_safetensors",
        ]
        if precision == "fp16":
            command.append("--half")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=CHECKPOINTS)
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--converter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.model, args.precision, args.converter, args.out)
