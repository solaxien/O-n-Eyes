"""Fetch fixed-size COCO 2017 splits without downloading the full image archives."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import random
import re
from time import sleep

import httpx
from PIL import Image, ImageOps


ROWS_URL = "https://datasets-server.huggingface.co/rows"
CAPTION_DATASET = "phiyodr/coco2017"
IMAGE_BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"


def fetch(client: httpx.Client, url: str, params: dict | None = None) -> httpx.Response:
    for attempt in range(5):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            if attempt == 4:
                raise
            sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def prepare(split: str, count: int, destination: Path, workers: int):
    split_dir = "train2017" if split == "train" else "val2017"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "reference").mkdir(exist_ok=True)
    manifest = destination / "manifest.jsonl"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if manifest.is_file():
            records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            if len(records) != count or any(row["split"] != split_dir for row in records):
                raise ValueError("Use a separate output directory for each split and count.")
        else:
            records = []
            rng = random.Random(38 if split == "validation" else 3800)
            for offset in range(0, count, 100):
                rows = fetch(client, ROWS_URL, {
                    "dataset": CAPTION_DATASET, "config": "default", "split": split,
                    "offset": offset, "length": min(100, count - offset),
                }).json()["rows"]
                for item in rows:
                    row = item["row"]
                    filename = row["file_name"]
                    if not re.fullmatch(fr"{split_dir}/\d{{12}}\.jpg", filename):
                        raise ValueError(f"COCO row is outside the selected split: {filename}")
                    key = f"coco-{row['image_id']:012d}"
                    records.append({
                        "id": key, "prompt": row["captions"][0].strip(), "seed": rng.getrandbits(32),
                        "reference": f"reference/{key}.png", "split": split_dir, "source_file": filename,
                    })
            if len(records) != count or len({row["id"] for row in records}) != count:
                raise RuntimeError("The selected dataset rows are incomplete or contain duplicate images.")
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            temporary.replace(manifest)

        def download(row):
            source_file = row["source_file"]
            if not re.fullmatch(fr"{split_dir}/\d{{12}}\.jpg", source_file):
                raise ValueError("Invalid COCO image path")
            target = destination / "reference" / f"{row['id']}.png"
            if target.is_file():
                return
            content = fetch(client, f"{IMAGE_BUCKET}/{source_file}").content
            temporary = target.with_suffix(".tmp")
            with Image.open(BytesIO(content)) as photo:
                image = ImageOps.fit(photo.convert("RGB"), (512, 512), Image.Resampling.LANCZOS)
                image.save(temporary, format="PNG")
            temporary.replace(target)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, _ in enumerate(pool.map(download, records), start=1):
                if index % 50 == 0 or index == count:
                    print(f"{split_dir}: {index}/{count} reference images", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.count <= 5000 or not 1 <= args.workers <= 16:
        parser.error("Use 1–5,000 images and 1–16 download workers.")
    prepare(args.split, args.count, args.out, args.workers)
