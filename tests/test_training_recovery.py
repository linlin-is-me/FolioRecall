from functools import partial
import io
import tempfile
import unittest

from datasets import Dataset
import torch
from sentence_transformers import SentenceTransformerTrainingArguments

from foliorecall.trainer import AdapterTrainer, batch_plan, fixed_batch_sampler


class TrainingSamplerTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_list([
            {"query": f"query {i}", "positive": {"path": f"page-{i}"},
             "negative": {"path": f"negative-{i % 7}"}}
            for i in range(32)
        ])

    def arguments(self, output, seed):
        return SentenceTransformerTrainingArguments(
            output_dir=output, use_cpu=True, report_to="none", seed=seed,
            batch_sampler=partial(fixed_batch_sampler, fixed_seed=seed),
        )

    def actual_batches(self, args):
        trainer = AdapterTrainer.__new__(AdapterTrainer)
        trainer.args = args
        # Match ST's Dataset branch: it passes a seeded generator but omits seed.
        return trainer.get_batch_sampler(
            self.dataset, batch_size=4, drop_last=False,
            generator=torch.Generator().manual_seed(args.seed),
        )

    def test_upstream_sampler_matches_saved_seed_and_changes_with_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            first = self.actual_batches(self.arguments(folder, 42))
            second = self.actual_batches(self.arguments(folder, 7))
        self.assertEqual(first, batch_plan(self.dataset, 4, 42))
        self.assertEqual(second, batch_plan(self.dataset, 4, 7))
        self.assertNotEqual(first, second)

    def test_training_arguments_roundtrip_preserves_sampler_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.arguments(folder, 42)
            buffer = io.BytesIO()
            torch.save(args, buffer)
            buffer.seek(0)
            # This is the same trusted, locally produced training_args payload.
            restored = torch.load(buffer, weights_only=False)
            self.assertEqual(restored.seed, 42)
            self.assertEqual(restored.batch_sampler.keywords, {"fixed_seed": 42})
            self.assertEqual(self.actual_batches(restored), batch_plan(self.dataset, 4, 42))

    def test_sampler_requires_explicit_configured_seed(self):
        with self.assertRaisesRegex(ValueError, "fixed_seed"):
            fixed_batch_sampler(self.dataset, batch_size=4, seed=0)


if __name__ == "__main__":
    unittest.main()
