// Compact on-device semantic verifier.
//
// A YOLOX-Nano COCO detector (Apache-2.0, 3.6 MB) finds objects in the final
// image. Box-level rules then check object presence, counts, colour inside the
// detected box, and simple 2D relations. Anything the detector cannot see is
// reported as "not_verifiable", never as passed or failed.
//
// This verifier may steer local retries and candidate selection. It must not
// grade its own success in the final benchmark: use a separate, stronger
// evaluator offline for that.
import { hsv, isColor } from "./vision.js";

export const VERIFIER_INPUT = 416;
export const DEFAULT_THRESHOLDS = Object.freeze({
  candidate: 0.25, // detections kept for inspection
  presence: 0.35, // an object counts as present
  count: 0.45, // a detection counts towards an exact count
  nms: 0.45,
  relationMargin: 0.05, // fraction of the image side
});

export const COCO_CLASSES = Object.freeze([
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush",
]);

// Prompt vocabulary (prompt.js) mapped onto detector classes. Labels absent
// from this map and from COCO_CLASSES cannot be verified on this device.
const ALIASES = Object.freeze({
  man: "person",
  woman: "person",
  child: "person",
  television: "tv",
  table: "dining table",
});

export function detectorClass(label) {
  const mapped = ALIASES[label] || label;
  return COCO_CLASSES.includes(mapped) ? mapped : null;
}

/**
 * RGBA pixels -> YOLOX input: BGR, CHW, float32 0..255, top-left letterbox
 * padded with 114, bilinear resize. Matches the official YOLOX ONNX demo.
 */
export function preprocess(pixels, width, height, size = VERIFIER_INPUT) {
  const ratio = Math.min(size / width, size / height),
    nw = Math.max(1, Math.floor(width * ratio)),
    nh = Math.max(1, Math.floor(height * ratio)),
    plane = size * size,
    data = new Float32Array(3 * plane).fill(114);
  const sx = width / nw,
    sy = height / nh;
  for (let y = 0; y < nh; y++) {
    const fy = Math.max(0, Math.min(height - 1, (y + 0.5) * sy - 0.5)),
      y0 = Math.floor(fy),
      y1 = Math.min(height - 1, y0 + 1),
      wy = fy - y0;
    for (let x = 0; x < nw; x++) {
      const fx = Math.max(0, Math.min(width - 1, (x + 0.5) * sx - 0.5)),
        x0 = Math.floor(fx),
        x1 = Math.min(width - 1, x0 + 1),
        wx = fx - x0;
      for (let c = 0; c < 3; c++) {
        const p = (yy, xx) => pixels[(yy * width + xx) * 4 + c],
          value =
            (1 - wy) * ((1 - wx) * p(y0, x0) + wx * p(y0, x1)) +
            wy * ((1 - wx) * p(y1, x0) + wx * p(y1, x1));
        // RGB channel c goes to BGR plane 2 - c.
        data[(2 - c) * plane + y * size + x] = value;
      }
    }
  }
  return { data, ratio, size };
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])),
    h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])),
    inter = w * h,
    union =
      (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Decode raw YOLOX output [N, 85] (grid offsets, log sizes, objectness, 80
 * class scores) into boxes in source-image pixels after per-class NMS.
 */
export function decode(
  output,
  { size = VERIFIER_INPUT, ratio = 1, width = Infinity, height = Infinity, threshold = DEFAULT_THRESHOLDS.candidate, nms = DEFAULT_THRESHOLDS.nms } = {},
) {
  const classes = COCO_CLASSES.length,
    stride = 5 + classes,
    candidates = [];
  let row = 0;
  for (const s of [8, 16, 32]) {
    const cells = size / s;
    for (let gy = 0; gy < cells; gy++)
      for (let gx = 0; gx < cells; gx++, row++) {
        const o = row * stride,
          objectness = output[o + 4];
        if (objectness < threshold) continue;
        let best = 0,
          cls = 0;
        for (let k = 0; k < classes; k++)
          if (output[o + 5 + k] > best) {
            best = output[o + 5 + k];
            cls = k;
          }
        const score = objectness * best;
        if (score < threshold) continue;
        const cx = (output[o] + gx) * s,
          cy = (output[o + 1] + gy) * s,
          w = Math.exp(output[o + 2]) * s,
          h = Math.exp(output[o + 3]) * s;
        candidates.push({
          label: COCO_CLASSES[cls],
          score,
          box: [
            Math.max(0, (cx - w / 2) / ratio),
            Math.max(0, (cy - h / 2) / ratio),
            Math.min(width, (cx + w / 2) / ratio),
            Math.min(height, (cy + h / 2) / ratio),
          ],
        });
      }
  }
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const item of candidates)
    if (!kept.some((k) => k.label === item.label && iou(k.box, item.box) > nms))
      kept.push(item);
  return kept;
}

const CHROMATIC = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"];
const NEUTRAL = ["black", "white", "gray"];

/**
 * Colour of the central 60% of a box. Returns the share of sampled pixels
 * matching each named colour, and whether the requested one dominates.
 */
export function boxColor(pixels, width, height, box, requested) {
  const [x1, y1, x2, y2] = box,
    padX = (x2 - x1) * 0.2,
    padY = (y2 - y1) * 0.2,
    left = Math.max(0, Math.floor(x1 + padX)),
    right = Math.min(width - 1, Math.ceil(x2 - padX)),
    top = Math.max(0, Math.floor(y1 + padY)),
    bottom = Math.min(height - 1, Math.ceil(y2 - padY)),
    names = [...new Set([...CHROMATIC, ...NEUTRAL, requested])],
    counts = Object.fromEntries(names.map((n) => [n, 0]));
  let total = 0;
  const step = Math.max(1, Math.floor(Math.min(right - left, bottom - top) / 48));
  for (let y = top; y <= bottom; y += step)
    for (let x = left; x <= right; x += step) {
      const i = (y * width + x) * 4,
        value = hsv(pixels[i], pixels[i + 1], pixels[i + 2]);
      total++;
      for (const name of names) if (isColor(name, value)) counts[name]++;
    }
  const shares = Object.fromEntries(
    Object.entries(counts).map(([k, v]) => [k, total ? v / total : 0]),
  );
  const neutralRequest = NEUTRAL.includes(requested) || requested === "silver",
    rivals = (neutralRequest ? NEUTRAL : CHROMATIC).filter(
      (n) =>
        n !== requested &&
        !(requested === "golden" && (n === "yellow" || n === "orange")) &&
        !(requested === "silver" && (n === "gray" || n === "white")),
    );
  const strongestRival = Math.max(0, ...rivals.map((n) => shares[n] || 0));
  const share = shares[requested] || 0;
  return {
    share,
    strongest_rival: strongestRival,
    pass: total > 0 && share >= 0.15 && share >= strongestRival,
    sampled: total,
  };
}

function center(box) {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

/** Returns true, false, or null (relation not checkable in 2D). */
export function relationHolds(relation, a, b, width, height, margin = DEFAULT_THRESHOLDS.relationMargin) {
  const [ax, ay] = center(a.box),
    [bx, by] = center(b.box),
    mx = margin * width,
    my = margin * height;
  switch (relation) {
    case "left of": return ax < bx - mx;
    case "right of": return ax > bx + mx;
    case "above": return ay < by - my;
    case "below":
    case "under": return ay > by + my;
    case "beside": return Math.abs(ax - bx) > mx && Math.abs(ax - bx) >= Math.abs(ay - by);
    default: return null; // "in front of", "behind": depth is not observable here.
  }
}

/**
 * Check each parsed clause against detections. Status is one of
 * "pass", "fail" or "not_verifiable".
 */
export function verify(analysis, detections, pixels, width, height, thresholds = DEFAULT_THRESHOLDS) {
  const t = { ...DEFAULT_THRESHOLDS, ...thresholds },
    object_checks = [],
    count_checks = [],
    color_checks = [],
    relation_checks = [],
    best = new Map();
  for (const item of analysis.objects || []) {
    const name = `${item.color ? item.color + " " : ""}${item.label}`,
      cls = detectorClass(item.label);
    if (!cls) {
      object_checks.push({ label: name, status: "not_verifiable", reason: "Not a detector class on this device." });
      continue;
    }
    const present = detections.filter((d) => d.label === cls && d.score >= t.presence);
    if (present.length) best.set(item.label, present[0]);
    object_checks.push({
      label: name,
      detector_class: cls,
      status: present.length ? "pass" : "fail",
      confidence: present[0]?.score ?? Math.max(0, ...detections.filter((d) => d.label === cls).map((d) => d.score)),
    });
    if (item.count > 1) {
      const found = detections.filter((d) => d.label === cls && d.score >= t.count).length;
      count_checks.push({
        label: `${item.count} ${item.label}`,
        requested: item.count,
        found,
        status: found === item.count ? "pass" : "fail",
      });
    }
    if (item.color) {
      if (!present.length)
        color_checks.push({ label: name, status: "fail", reason: "Object was not detected, so its colour cannot match." });
      else {
        const colour = boxColor(pixels, width, height, present[0].box, item.color);
        color_checks.push({
          label: name,
          status: colour.pass ? "pass" : "fail",
          share: colour.share,
          strongest_rival: colour.strongest_rival,
        });
      }
    }
  }
  for (const rel of analysis.relations || []) {
    const label = `${rel.subject} ${rel.relation} ${rel.object}`,
      a = best.get(rel.subject),
      b = best.get(rel.object),
      checkable = detectorClass(rel.subject) && detectorClass(rel.object);
    if (!checkable) {
      relation_checks.push({ label, status: "not_verifiable", reason: "An object in this relation is not a detector class." });
      continue;
    }
    if (!a || !b) {
      relation_checks.push({ label, status: "fail", reason: "Both objects must be detected to check their layout." });
      continue;
    }
    const holds = relationHolds(rel.relation, a, b, width, height, t.relationMargin);
    relation_checks.push(
      holds === null
        ? { label, status: "not_verifiable", reason: "Depth relations cannot be checked from 2D boxes." }
        : { label, status: holds ? "pass" : "fail" },
    );
  }
  const all = [...object_checks, ...count_checks, ...color_checks, ...relation_checks],
    passed = all.filter((c) => c.status === "pass").length,
    failed = all.filter((c) => c.status === "fail").length,
    notVerifiable = all.filter((c) => c.status === "not_verifiable").length;
  return {
    object_checks,
    count_checks,
    color_checks,
    relation_checks,
    summary: {
      checked: passed + failed,
      passed,
      failed,
      not_verifiable: notVerifiable,
      critical_failed: failed > 0,
      score: passed + failed ? passed / (passed + failed) : null,
    },
    detections: detections.filter((d) => d.score >= t.presence).slice(0, 20).map((d) => ({
      label: d.label,
      score: Number(d.score.toFixed(3)),
      box: d.box.map((v) => Math.round(v)),
    })),
    alignment: null,
    alignment_note: "No image-text alignment model is bundled in this release.",
    verifier_scope: "local compact verifier (YOLOX-Nano, COCO classes)",
    thresholds: t,
  };
}
