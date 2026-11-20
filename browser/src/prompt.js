// A small local prompt contract. It is used for planning and transparent UI
// feedback only; it never claims that an image satisfies a clause unless an
// on-device verifier has been installed and run.
const COLORS = [
  "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
  "white", "gray", "grey", "brown", "golden", "silver",
];
const COUNT_WORDS = new Map([
  ["one", 1], ["two", 2], ["three", 3], ["four", 4], ["five", 5],
  ["six", 6], ["seven", 7], ["eight", 8], ["nine", 9], ["ten", 10],
]);
const OBJECTS = [
  "traffic light", "fire hydrant", "potted plant", "dining table", "wine glass",
  "teddy bear", "cell phone", "sports ball", "parking meter", "stop sign",
  "motorcycle", "backpack", "suitcase", "microwave", "refrigerator", "television",
  "bicycle", "airplane", "banana", "apple", "orange", "broccoli", "carrot",
  "sandwich", "pizza", "donut", "cake", "fork", "chair", "couch", "table", "bed",
  "toilet", "laptop", "mouse", "remote", "keyboard", "clock", "vase", "scissors",
  "toothbrush", "person", "man", "woman", "child", "dog", "cat", "bird", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "car", "bus", "train",
  "truck", "boat", "cup", "bottle", "book", "umbrella", "kite", "bench", "tree",
  "house", "building", "flower", "mountain", "lake", "beach", "road", "school", "window", "station",
].sort((a, b) => b.length - a.length);
const RELATIONS = [
  ["in front of", "in front of"],
  ["to the left of", "left of"],
  ["to the right of", "right of"],
  ["underneath", "under"],
  ["below", "below"],
  ["above", "above"],
  ["behind", "behind"],
  ["beside", "beside"],
  ["next to", "beside"],
  ["under", "under"],
  ["left of", "left of"],
  ["right of", "right of"],
];

function escape(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function mentionBefore(text, offset) {
  const before = text.slice(0, offset);
  let chosen = null;
  for (const label of OBJECTS) {
    const match = [...before.matchAll(new RegExp(`\\b${escape(label)}s?\\b`, "g"))].at(-1);
    if (match) {
      const candidate = { label, start: match.index, end: match.index + match[0].length };
      if (!chosen || candidate.start > chosen.start) chosen = candidate;
    }
  }
  return chosen;
}

function mentionAfter(text, offset) {
  const after = text.slice(offset);
  let chosen = null;
  for (const label of OBJECTS) {
    const match = new RegExp(`\\b${escape(label)}s?\\b`).exec(after);
    if (match) {
      const candidate = { label, start: offset + match.index, end: offset + match.index + match[0].length };
      if (!chosen || candidate.start < chosen.start) chosen = candidate;
    }
  }
  return chosen;
}

function objectAt(text, label, match) {
  const left = text.slice(Math.max(0, match.index - 24), match.index);
  const countMatch = new RegExp(`(?:\\b(${[...COUNT_WORDS.keys()].join("|")}|\\d+)\\s+)?(?:\\b(${COLORS.join("|")})\\s+)?$`, "i").exec(left);
  const rawCount = countMatch?.[1]?.toLowerCase();
  const count = rawCount ? (COUNT_WORDS.get(rawCount) || Number(rawCount)) : 1;
  const color = countMatch?.[2]?.toLowerCase();
  return { label, count, ...(color ? { color: color === "grey" ? "gray" : color } : {}), offset: match.index };
}

export function analyzePrompt(value) {
  const prompt = String(value || "").trim();
  const lower = prompt.toLowerCase().replace(/\s+/g, " ");
  const objects = [];
  const occupied = [];
  for (const label of OBJECTS) {
    for (const match of lower.matchAll(new RegExp(`\\b${escape(label)}s?\\b`, "g"))) {
      const range = [match.index, match.index + match[0].length];
      if (occupied.some(([start, end]) => range[0] < end && start < range[1])) continue;
      occupied.push(range);
      objects.push(objectAt(lower, label, match));
    }
  }
  objects.sort((a, b) => a.offset - b.offset);
  const relations = [];
  const relationKeys = new Set();
  for (const [needle, label] of RELATIONS) {
    for (const match of lower.matchAll(new RegExp(`\\b${escape(needle)}\\b`, "g"))) {
      const subject = mentionBefore(lower, match.index);
      const object = mentionAfter(lower, match.index + match[0].length);
      if (subject && object && subject.label !== object.label) {
        const key = `${subject.label}|${label}|${object.label}`;
        if (!relationKeys.has(key)) {
          relationKeys.add(key);
          relations.push({ subject: subject.label, relation: label, object: object.label });
        }
      }
    }
  }
  const attributes = objects.filter((item) => item.color || item.count > 1).length;
  const complexity = Math.min(
    1,
    0.12 + Math.max(0, objects.length - 1) * 0.18 + attributes * 0.12 + relations.length * 0.22,
  );
  return {
    prompt,
    objects,
    relations,
    complexity,
    needs_semantic_verification: objects.length > 0 || relations.length > 0,
  };
}

export function conditioningPrompt(analysis) {
  if (!analysis.objects.length) return analysis.prompt;
  const clauses = analysis.objects.map((item) => {
    const count = item.count > 1 ? `${item.count} ` : "";
    return `${count}${item.color ? item.color + " " : ""}${item.label}${item.count > 1 ? "s" : ""}`;
  });
  const relationText = analysis.relations.map(
    (item) => `${item.subject} ${item.relation} ${item.object}`,
  );
  return [analysis.prompt, `Preserve: ${clauses.join("; ")}.`, relationText.length ? `Layout: ${relationText.join("; ")}.` : ""]
    .filter(Boolean)
    .join(" ");
}

export function semanticSummary(analysis) {
  const objectText = analysis.objects.length
    ? analysis.objects.map((item) => `${item.count > 1 ? item.count + " " : ""}${item.color ? item.color + " " : ""}${item.label}`).join(", ")
    : "no recognized objects";
  const relationText = analysis.relations.length
    ? analysis.relations.map((item) => `${item.subject} ${item.relation} ${item.object}`).join("; ")
    : "no recognized spatial relation";
  return { objects: objectText, relations: relationText };
}
