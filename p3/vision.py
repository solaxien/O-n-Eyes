"""Conservative, measurable corrections on a decoded RGB image."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


CV_STAGES = ("repair", "edges", "color", "gate")


@dataclass(frozen=True)
class VisionResult:
    image: Image.Image
    applied: list[str]
    before: dict[str, float]
    after: dict[str, float]
    artifact_fraction: float
    usable: bool


def _inspect(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lo, p10, p90, hi = np.percentile(gray, (1, 10, 90, 99))
    metrics = {
        "luminance_span": float(hi - lo),
        "contrast": float(p90 - p10),
        "clipped_fraction": float(np.mean((gray < 3) | (gray > 252))),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    }
    return gray, metrics


def _flat_pixels(median: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(median, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(median, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy) < 28


def _repair_mask(residual: np.ndarray, flat: np.ndarray) -> np.ndarray:
    candidates = ((residual >= 48) & flat).astype(np.uint8)
    candidates[[0, -1], :] = 0
    candidates[:, [0, -1]] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    small = (stats[:, cv2.CC_STAT_AREA] <= 6)
    small[0] = False
    return small[labels].astype(np.uint8)


def _adjust_lab(rgb: np.ndarray, denoise: bool, contrast: float, color: bool) -> tuple[np.ndarray, list[str]]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luma, a, b = cv2.split(lab)
    stages = []
    if denoise:
        filtered = cv2.bilateralFilter(luma, 5, 12, 2)
        luma = cv2.addWeighted(luma, 0.65, filtered, 0.35, 0)
        stages.append("Edge-aware denoising")
    if color and 12 <= contrast < 55:
        adjusted = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(luma)
        luma = cv2.addWeighted(luma, 0.82, adjusted, 0.18, 0)
        stages.append("Local contrast")
    if color:
        a_offset = a.astype(np.int16) - 128
        b_offset = b.astype(np.int16) - 128
        neutral = (luma > 45) & (luma < 215) & (np.abs(a_offset) <= 10) & (np.abs(b_offset) <= 10)
        if neutral.mean() >= 0.35:
            cast = np.median(np.column_stack((a_offset[neutral], b_offset[neutral])), axis=0)
            if np.max(np.abs(cast)) >= 4:
                da, db = np.clip(np.rint(cast * 0.35), -3, 3).astype(np.int16)
                a = np.clip(a.astype(np.int16) - da, 0, 255).astype(np.uint8)
                b = np.clip(b.astype(np.int16) - db, 0, 255).astype(np.uint8)
                stages.append("Chroma balance")
    if stages:
        rgb = cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2RGB)
    return rgb, stages


def enhance(image: Image.Image, omit: frozenset[str] = frozenset()) -> VisionResult:
    original = np.asarray(image.convert("RGB"))
    gray, before = _inspect(original)
    median = cv2.medianBlur(gray, 3)
    residual = cv2.absdiff(gray, median)
    flat = _flat_pixels(median)
    mask = _repair_mask(residual, flat)
    artifact_fraction = float(mask.mean())
    usable = before["luminance_span"] >= 8 and artifact_fraction <= 0.035
    if not usable and "gate" not in omit:
        return VisionResult(image, ["Quality gate flagged decode"], before, before, artifact_fraction, False)

    rgb = original
    stages = []
    if "repair" not in omit and 0 < artifact_fraction <= 0.004:
        rgb = cv2.inpaint(rgb, mask, 2, cv2.INPAINT_TELEA)
        stages.append("Speckle repair")

    noisy_flat_regions = np.mean(flat & (residual >= 7) & (residual < 32)) > 0.025
    rgb, adjustment_stages = _adjust_lab(rgb, noisy_flat_regions and "edges" not in omit, before["contrast"], "color" not in omit)
    stages.extend(adjustment_stages)
    _, after = _inspect(rgb) if stages else (gray, before)

    # Assess structural edge loss on the median image so repairing a hot pixel
    # does not look like losing genuine detail.
    if stages and "gate" not in omit:
        after_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edge_count_before = np.count_nonzero(~flat)
        edge_count_after = np.count_nonzero(~_flat_pixels(cv2.medianBlur(after_gray, 3)))
        lost_edges = edge_count_before > 100 and edge_count_after < 0.72 * edge_count_before
        added_clipping = after["clipped_fraction"] > before["clipped_fraction"] + 0.025
        if lost_edges or added_clipping or after["luminance_span"] < 8:
            return VisionResult(image, ["Quality gate kept raw image"], before, before, artifact_fraction, usable)

    return VisionResult(Image.fromarray(rgb), stages, before, after, artifact_fraction, usable)


def resize(image: Image.Image, long_side: int, *, linear: bool = False) -> Image.Image:
    width, height = image.size
    scale = long_side / max(width, height)
    if scale == 1:
        return image
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    if linear:
        interpolation = cv2.INTER_LINEAR
    return Image.fromarray(cv2.resize(np.asarray(image), target, interpolation=interpolation))
