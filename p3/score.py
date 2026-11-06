"""Offline CLIP alignment, object-level composition checks, and paired COCO FID."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, CLIPModel, CLIPProcessor, YolosForObjectDetection

from p3.benchmark import read_manifest, write_json


CLIP = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
YOLOS = "hustvl/yolos-tiny"
YOLOS_REVISION = "95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de"
COLORS = ("black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow")
DETECTOR_THRESHOLD = 0.55


def _open(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class Alignment:
    def __init__(self, device: str, with_detector: bool):
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP, revision=CLIP_REVISION, token=False)
        self.clip = CLIPModel.from_pretrained(
            CLIP, revision=CLIP_REVISION, token=False, use_safetensors=False
        ).to(device).eval()
        if with_detector:
            self.detector_processor = AutoImageProcessor.from_pretrained(
                YOLOS, revision=YOLOS_REVISION, token=False
            )
            self.detector = YolosForObjectDetection.from_pretrained(
                YOLOS, revision=YOLOS_REVISION, token=False, use_safetensors=True
            ).to(device).eval()

    def cosine(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        batch = self.clip_processor(
            images=images, text=texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return (self.clip(**batch).logits_per_image / self.clip.logit_scale.exp()).cpu()

    def composition(self, image: Image.Image, sample: dict) -> dict:
        width, height = image.size
        inputs = self.detector_processor(images=image, return_tensors="pt").to(self.device)
        predictions = self.detector_processor.post_process_object_detection(
            self.detector(**inputs), threshold=DETECTOR_THRESHOLD, target_sizes=torch.tensor([[height, width]])
        )[0]
        detections = {}
        for label, box in zip(predictions["labels"].tolist(), predictions["boxes"].tolist()):
            name = self.detector.config.id2label[label]
            detections.setdefault(name, []).append(box)

        checks = {}
        groups = {}
        for target in sample["objects"]:
            label, key = target["label"], target["id"]
            boxes = detections.get(label, [])
            checks[f"{key} count"] = len(boxes) == target["count"]
            groups[key] = boxes if checks[f"{key} count"] else []
            if "color" in target:
                good = False
                if groups[key]:
                    crops = [image.crop((max(0, x), max(0, y), min(width, x2), min(height, y2))) for x, y, x2, y2 in boxes]
                    prompts = [f"a photo of a {color} {label}" for color in COLORS]
                    predicted_colors = [COLORS[index] for index in self.cosine(crops, prompts).argmax(dim=1).tolist()]
                    good = all(color == target["color"] for color in predicted_colors)
                checks[f"{key} color"] = good

        for subject, relation, other in sample.get("relations", []):
            if relation not in {"left", "right", "above", "below"}:
                raise ValueError(f"Unsupported relation {relation}")
            horizontal = relation in {"left", "right"}
            axis = 0 if horizontal else 1
            margin = 0.05 * (width if horizontal else height)
            first = [(box[axis] + box[axis + 2]) / 2 for box in groups[subject]]
            second = [(box[axis] + box[axis + 2]) / 2 for box in groups[other]]
            if relation in {"right", "below"}:
                first, second = second, first
            checks[f"{subject} {relation} {other}"] = bool(first and second and max(first) + margin < min(second))
        return {"pass": all(checks.values()), "checks": checks}


def _fid(manifest: Path, root: Path, pairs: list[tuple[dict, Path]], device: str) -> dict:
    from torch_fidelity import calculate_metrics

    if len(pairs) < 100 or any("reference" not in sample for sample, _ in pairs):
        raise ValueError("FID requires at least 100 matched images and reference paths.")
    with TemporaryDirectory(prefix="fid-", dir=root) as temporary:
        scratch = Path(temporary)
        generated, reference = scratch / "generated", scratch / "reference"
        generated.mkdir()
        reference.mkdir()
        for sample, image in pairs:
            name = f"{sample['id']}.png"
            shutil.copyfile(image, generated / name)
            shutil.copyfile(manifest.parent / sample["reference"], reference / name)
        values = calculate_metrics(
            input1=str(generated), input2=str(reference), cuda=device == "cuda", batch_size=16,
            fid=True, rng_seed=38, verbose=False,
        )
    return {
        "fid": float(values["frechet_inception_distance"]),
        "fid_n": len(pairs),
    }


@torch.inference_mode()
def score_runs(manifest: Path, runs: list[Path], device: str, fid: bool):
    rows = read_manifest(manifest)
    if fid:
        rows = [
            sample for sample in rows
            if all((root / "images" / f"{sample['id']}.png").is_file() for root in runs)
        ]
    alignment = Alignment(device, any(row.get("objects") for row in rows))
    torch.hub.set_dir(".cache/torch")
    for root in runs:
        pairs = [(sample, root / "images" / f"{sample['id']}.png") for sample in rows]
        pairs = [(sample, path) for sample, path in pairs if path.is_file()]
        if not pairs:
            raise ValueError(f"No generated images in {root}")
        by_sample = {}
        for index in range(0, len(pairs), 8):
            batch = pairs[index:index + 8]
            images = [_open(path) for _, path in batch]
            cosine = alignment.cosine(images, [sample["prompt"] for sample, _ in batch]).diagonal().tolist()
            for (sample, _), image, clip in zip(batch, images, cosine):
                result = {"clip": clip}
                if sample.get("objects"):
                    result["composition"] = alignment.composition(image, sample)
                by_sample[sample["id"]] = result
        composition_n = sum(bool(row.get("objects")) for row in rows)
        passed = sum(result.get("composition", {}).get("pass", False) for result in by_sample.values())
        scores = {
            "clip_mean": float(np.mean([result["clip"] for result in by_sample.values()])),
            "clip_n": len(by_sample),
            "composition_pass": passed / composition_n if composition_n else None,
            "composition_n": composition_n,
            "detector_confidence": DETECTOR_THRESHOLD if composition_n else None,
            "by_sample": by_sample,
        }
        if fid:
            scores.update(_fid(manifest, root, pairs, device))
        write_json(root / "scores.json", scores)
        print(root.name, {key: value for key, value in scores.items() if key != "by_sample"}, flush=True)
