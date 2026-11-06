"""Offline CLIP alignment, object-level composition checks, and paired COCO FID."""

import json
from pathlib import Path

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


def _generated_image(root: Path, sample: dict) -> Path | None:
    key = sample["id"]
    image = root / "images" / f"{key}.png"
    record = root / "runs" / f"{key}.json"
    if image.is_file() and record.is_file():
        data = json.loads(record.read_text(encoding="utf-8"))
        if data.get("sample") == sample and "error" not in data:
            return image
    return None


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


def _inception_features(extractor, paths: list[Path], device: str) -> np.ndarray:
    features = []
    for index in range(0, len(paths), 16):
        batch = np.stack([np.asarray(_open(path)) for path in paths[index:index + 16]])
        images = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
        features.append(extractor(images)[0].cpu().numpy())
    return np.concatenate(features).astype(np.float64)


def frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_mean, second_mean = first.mean(axis=0), second.mean(axis=0)
    left = (first - first_mean) / np.sqrt(len(first) - 1)
    right = (second - second_mean) / np.sqrt(len(second) - 1)
    # The cross term equals the sum of singular values of the product of the
    # centered sample matrices. This also works for rank-deficient covariances.
    cross = np.linalg.svd(left @ right.T, compute_uv=False).sum()
    score = np.square(first_mean - second_mean).sum() + np.square(left).sum() + np.square(right).sum() - 2 * cross
    return float(max(0, score))


@torch.inference_mode()
def score_runs(manifest: Path, runs: list[Path], device: str, fid: bool):
    rows = read_manifest(manifest)
    generated = {
        root: {sample["id"]: path for sample in rows if (path := _generated_image(root, sample)) is not None}
        for root in runs
    }
    if fid:
        rows = [sample for sample in rows if all(sample["id"] in paths for paths in generated.values())]
        if len(rows) < 100 or any("reference" not in sample for sample in rows):
            raise ValueError("FID requires at least 100 common generated images and matching reference paths.")
    torch.hub.set_dir(".cache/torch")
    alignment = Alignment(device, any(row.get("objects") for row in rows))
    if fid:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        inception = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
        references = _inception_features(inception, [manifest.parent / row["reference"] for row in rows], device)
    for root in runs:
        paths = generated[root]
        pairs = [(sample, paths[sample["id"]]) for sample in rows if sample["id"] in paths]
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
            features = _inception_features(inception, [path for _, path in pairs], device)
            scores.update(fid=frechet_distance(features, references), fid_n=len(pairs))
        write_json(root / "scores.json", scores)
        print(root.name, {key: value for key, value in scores.items() if key != "by_sample"}, flush=True)
