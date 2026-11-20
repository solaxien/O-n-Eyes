"""Create a deterministic caption and reference-image sample from COCO val2017."""

import argparse
import json
from pathlib import Path
import random
from zipfile import ZipFile

from PIL import Image, ImageOps


def prepare(annotations: Path, images: Path, destination: Path, count: int):
    refs = destination / "reference"
    refs.mkdir(parents=True, exist_ok=True)
    with ZipFile(annotations) as archive:
        data = json.loads(archive.read("annotations/captions_val2017.json"))
    filenames = {row["id"]: row["file_name"] for row in data["images"]}
    captions = {}
    for row in sorted(data["annotations"], key=lambda row: row["id"]):
        captions.setdefault(row["image_id"], row["caption"])
    rng = random.Random(38)
    selected = rng.sample(sorted(filenames.keys() & captions.keys()), count)
    with ZipFile(images) as archive, (destination / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for image_id in selected:
            name = f"coco-{image_id:012d}"
            reference = f"reference/{name}.png"
            with archive.open(f"val2017/{filenames[image_id]}") as source:
                with Image.open(source) as image:
                    ImageOps.fit(image.convert("RGB"), (512, 512), Image.Resampling.LANCZOS).save(destination / reference)
            record = {"id": name, "prompt": captions[image_id], "seed": rng.getrandbits(32), "reference": reference}
            manifest.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    options = parser.parse_args()
    prepare(options.annotations, options.images, options.out, options.count)
