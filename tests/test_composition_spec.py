import json
import unittest
from pathlib import Path

from p3.composition_spec import normalise


class CompositionSpecTests(unittest.TestCase):
    def test_prompt_words_map_to_detector_labels(self):
        sample = {"objects": [{"label": "dog", "count": 1}, {"label": "table", "count": 1}],
                  "relations": [["dog", "under", "table"]]}
        result = normalise(sample)
        self.assertEqual([o["label"] for o in result["objects"]], ["dog", "dining table"])
        self.assertEqual(result["relations"], [["dog", "below", "table"]])
        self.assertEqual(result["not_verifiable"], [])

    def test_unverifiable_clauses_are_listed_not_scored(self):
        sample = {"objects": [{"label": "boat", "count": 1}, {"label": "lake", "count": 1}],
                  "relations": [["boat", "in front of", "lake"]]}
        result = normalise(sample)
        self.assertEqual([o["label"] for o in result["objects"]], ["boat"])
        self.assertEqual(result["relations"], [])
        self.assertEqual(len(result["not_verifiable"]), 2)

    def test_historical_manifest_is_unchanged(self):
        for line in Path("data/composition.jsonl").read_text().splitlines():
            sample = json.loads(line)
            result = normalise(sample)
            self.assertEqual(result["not_verifiable"], [], sample["id"])
            self.assertEqual(len(result["objects"]), len(sample["objects"]))

    def test_android_manifest_is_scorable(self):
        rows = [json.loads(line) for line in Path("data/android-composition-v2.jsonl").read_text().splitlines()]
        results = {row["id"]: normalise(row) for row in rows}
        # One landscape prompt has no COCO object at all; it is reported, not scored.
        self.assertEqual([key for key, r in results.items() if not r["objects"]], ["house-lake-winter"])
        self.assertTrue(results["house-lake-winter"]["not_verifiable"])


if __name__ == "__main__":
    unittest.main()
