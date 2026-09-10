from pathlib import Path
import tempfile
import unittest

from datasets import Dataset
from PIL import Image

from foliorecall.trainer import batch_plan, preprocessing, validate_training_data


class TrainingTests(unittest.TestCase):
    def test_reuse_across_batches_without_conflicting_pairs(self):
        dataset = Dataset.from_list([{"query": f"q{i}", "positive": {"path": f"p{i}"},
                                     "negative": {"path": f"n{i % 2}"}} for i in range(4)])
        batches = batch_plan(dataset, 4, 42)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(4)))
        self.assertGreater(len(batches), 1)
        for batch in batches:
            values = [dataset[i][k]["path"] for i in batch for k in ("positive", "negative")]
            self.assertEqual(len(values), len(set(values)))

    def test_global_reuse_is_valid_but_dev_overlap_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            preview = Path(folder) / "page.png"
            Image.new("RGB", (8, 8)).save(preview)
            pages = [{"page_id": p, "preview": str(preview)} for p in ("a", "b", "n", "c", "d")]
            def row(q, p, n):
                return {"query": q, "positive": p, "negative": n, "original_negative_ids": [n]}
            split = {"train": [row("qa", "a", "n"), row("qb", "b", "n")], "dev": [row("qc", "c", "d")]}
            validate_training_data(split, pages, pages[-2:])
            with self.assertRaisesRegex(ValueError, "交叉"):
                validate_training_data(split, pages, pages[-3:])

    def test_image_wrapper_preserves_processing(self):
        class Encoder:
            def preprocess(self, inputs, **kwargs):
                return {"is_image": isinstance(inputs[0], Image.Image), "size": inputs[0].size, **kwargs}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "page.png"
            Image.new("RGB", (13, 17)).save(path)
            config = {"max_length": 8192, "min_pixels": 4096, "max_pixels": 1310720}
            result = preprocessing(Encoder(), config)([{"path": str(path)}], prompt="prompt")
            self.assertTrue(result["is_image"])
            self.assertEqual(result["size"], (13, 17))
            self.assertEqual(result["processing_kwargs"]["image"]["max_pixels"], 1310720)
            self.assertEqual(result["prompt"], "prompt")


if __name__ == "__main__":
    unittest.main()
