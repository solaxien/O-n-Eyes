"""Normalise composition manifests for the offline evaluator.

The Android manifest uses prompt words ("table", "child", "under") and some
clauses a 2D COCO detector cannot judge ("mountain", "behind"). This module maps
what can be checked onto detector vocabulary and lists the rest explicitly as
not verifiable, so they are never counted as passes or failures.
"""

LABEL_ALIASES = {"table": "dining table", "child": "person", "man": "person", "woman": "person",
                 "television": "tv"}
RELATION_ALIASES = {"left": "left", "left of": "left", "right": "right", "right of": "right",
                    "above": "above", "below": "below", "under": "below", "underneath": "below"}
COCO_LABELS = frozenset("""person bicycle car motorcycle airplane bus train truck boat traffic_light
fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe
backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat
baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana
apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed
dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink
refrigerator book clock vase scissors teddy_bear hair_drier toothbrush""".replace("_", " ").split())


def _coco(label: str) -> str | None:
    mapped = LABEL_ALIASES.get(label, label)
    return mapped if mapped in COCO_LABELS or mapped.replace(" ", "_") in {
        x.replace(" ", "_") for x in COCO_LABELS} else None


# Multi-word COCO labels were split above; rebuild them for exact matching.
COCO_LABELS = frozenset(COCO_LABELS | {"traffic light", "fire hydrant", "stop sign", "parking meter",
                                       "sports ball", "baseball bat", "baseball glove", "tennis racket",
                                       "wine glass", "hot dog", "potted plant", "dining table",
                                       "cell phone", "teddy bear", "hair drier"})


def normalise(sample: dict) -> dict:
    """Return a copy with detector labels, ids, supported relations and a not-verifiable list."""
    objects, skipped, ids = [], [], {}
    for item in sample.get("objects", []):
        key = item.get("id", item["label"])
        label = _coco(item["label"])
        if label is None:
            skipped.append(f"object {item['label']}")
            continue
        ids[key] = label
        objects.append({**item, "id": key, "label": label})
    relations = []
    for subject, relation, other in sample.get("relations", []):
        mapped = RELATION_ALIASES.get(relation)
        if mapped is None or subject not in ids or other not in ids:
            skipped.append(f"relation {subject} {relation} {other}")
            continue
        relations.append([subject, mapped, other])
    return {**sample, "objects": objects, "relations": relations, "not_verifiable": skipped}
