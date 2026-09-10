from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from PIL import Image

from foliorecall.trainer import batch_plan, preprocessing, validate_training_data, restore_history, RunChecks
from foliorecall.io import read_json, write_json, write_rows


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

    def test_older_checkpoint_keeps_attempt_but_excludes_abandoned_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            run = output / "run-from-2"
            run.mkdir()
            history = [{"step": i, "seconds": 10 * i} for i in range(1, 5)]
            write_json(output / "progress.json", {"steps": history, "completed_steps": 4})
            write_json(output / "result.json", {"complete": True})
            retained = restore_history(output, 2, run)
            self.assertEqual([row["step"] for row in retained], [1, 2])
            self.assertEqual(read_json(run / "previous-progress.json")["steps"], history)
            self.assertTrue(read_json(run / "previous-result.json")["complete"])
            self.assertFalse((output / "result.json").exists())
            self.assertEqual(read_json(output / "progress.json")["completed_steps"], 2)
            write_json(output / "progress.json", {"steps": [history[0], history[0]]})
            with self.assertRaisesRegex(ValueError, "重复"):
                restore_history(output, 2, run)

    def test_checkpoint_evaluation_preserves_training_peak_and_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            dev = output / "dev"
            write_rows(dev / "pages.jsonl", [{"page_id": "a"}])
            write_rows(dev / "queries.jsonl", [{"query_id": "q", "query": "text"}])
            write_json(dev / "qrels.json", {"q": {"a": 1}})
            write_json(output / "checkpoint-1/adapter_config.json", {})
            model = torch.nn.Linear(1, 1).train()
            callback = RunChecks(model, output, {}, None, 3600, {}, dev, False)
            control = SimpleNamespace(should_save=False, should_training_stop=False)
            state = SimpleNamespace(global_step=1)
            optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
            with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), patch(
                    "torch.cuda.max_memory_allocated", side_effect=[900, 600, 800]), patch(
                    "foliorecall.trainer.encode_pages", return_value=np.ones((1, 2))), patch(
                    "foliorecall.trainer.save_index", return_value=object()), patch(
                    "foliorecall.trainer.evaluate", return_value={"peak_cuda_bytes": 400}):
                callback.on_step_begin(None, state, control, optimizer=optimizer)
                callback.on_step_end(None, state, control)
                callback.on_save(None, state, control)
                self.assertTrue(model.training)
                state.global_step = 2
                callback.on_step_begin(None, state, control, optimizer=optimizer)
                callback.on_step_end(None, state, control)
            self.assertEqual([row["peak_cuda_bytes"] for row in callback.history], [900, 800])
            self.assertEqual(read_json(output / "checkpoint-1/dev-build.json")["peak_cuda_bytes"], 600)
            self.assertEqual(read_json(output / "checkpoint-1/dev-result.json")["peak_cuda_bytes"], 400)
            state.global_step = 1
            with patch("torch.cuda.reset_peak_memory_stats"), patch(
                    "foliorecall.trainer.encode_pages", side_effect=RuntimeError("probe failure")):
                with self.assertRaisesRegex(RuntimeError, "probe failure"):
                    callback.on_save(None, state, control)
            self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
