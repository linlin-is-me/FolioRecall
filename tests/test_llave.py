import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.util import cos_sim

from foliorecall.io import read_json, write_json, write_rows
from foliorecall.trainer import loss_settings, train
from scripts.run_llave import paired


class HardnessTests(unittest.TestCase):
    def representations(self):
        return [v.clone().requires_grad_() for v in (
            torch.tensor([[1., .2, .1], [.1, 1., .3]]),
            torch.tensor([[.9, .3, .2], [.2, .8, .1]]),
            torch.tensor([[.3, .6, .1], [.8, .1, .4]]))]

    def loss(self, reps, **kwargs):
        return CachedMultipleNegativesRankingLoss(torch.nn.Sequential(torch.nn.Identity()), mini_batch_size=1,
            scale=20., **kwargs).calculate_loss([[v] for v in reps])

    def test_missing_and_explicit_disabled_loss_and_gradient_match(self):
        results = []
        for config in ({}, {"hardness_mode": None, "hardness_strength": 0},
                       {"hardness_mode": "in_batch_negatives", "hardness_strength": 0}):
            reps = self.representations()
            loss = self.loss(reps, **loss_settings(config))
            results.append((loss.detach(), torch.autograd.grad(loss, reps)))
        for value, grads in results[1:]:
            self.assertTrue(torch.equal(results[0][0], value))
            for a, b in zip(results[0][1], grads):
                self.assertTrue(torch.equal(a, b))

    def test_weighting_scope_and_detached_gradient(self):
        reps = self.representations()
        actual = self.loss(reps, hardness_mode="in_batch_negatives", hardness_strength=9.)
        scores = cos_sim(reps[0], torch.cat(reps[1:]))
        # Columns are [positive0, positive1, negative0, negative1].
        other_sample = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]])
        expected = torch.nn.functional.cross_entropy(20 * scores + 9 * scores.detach() * other_sample,
                                                     torch.arange(2))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        actual_grads = torch.autograd.grad(actual, reps, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, reps, retain_graph=True)
        wrong = torch.nn.functional.cross_entropy(20 * scores + 9 * scores * other_sample, torch.arange(2))
        wrong_grads = torch.autograd.grad(wrong, reps)
        for a, b in zip(actual_grads, expected_grads):
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(any(g.abs().max() > 0 for g in actual_grads))
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(actual_grads, wrong_grads)))

    def test_invalid_loss_parameters_fail_before_teacher_loading(self):
        invalid = [{"hardness_mode": v} for v in (False, 1, [], "unknown")]
        invalid += [{"hardness_strength": v} for v in (True, "9", -1, float("nan"), float("inf"))]
        for config in invalid:
            with self.subTest(config=config), patch("foliorecall.trainer.load_encoder") as load:
                with self.assertRaises(ValueError):
                    train({"train": config}, "missing", "missing")
                load.assert_not_called()

    def test_resume_cannot_switch_loss_and_records_actual_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            data, output = Path(folder) / "data", Path(folder) / "run"
            row = {"query_id": "q", "query": "text", "positive": "p", "negative": "n"}
            config = {"device": "cuda", "seed": 42, "train": {
                "epochs": 1, "gradient_accumulation_steps": 1, "batch_size": 1}}
            write_json(data / "split.json", {"train": [row], "dev": []})
            write_rows(data / "pages.jsonl", [])
            write_rows(data / "dev/pages.jsonl", [])
            write_rows(data / "dev/queries.jsonl", [])
            write_json(data / "dev/qrels.json", {})
            saved = {"config": config, "profile": False, "data": str(data.resolve()), "rows": [row],
                     "batches": [[0]], "max_steps": 1, "batching_version": 2, "batching_seed": 42}
            write_json(output / "settings.json", saved)
            candidate = {**config, "train": {**config["train"], "hardness_mode": "in_batch_negatives", "hardness_strength": 9}}
            with patch("torch.cuda.is_available", return_value=True), patch(
                    "foliorecall.trainer.validate_training_data", return_value={}), patch(
                    "foliorecall.trainer.training_dataset", return_value=[]), patch(
                    "foliorecall.trainer.batch_plan", return_value=[[0]]), patch(
                    "foliorecall.trainer.load_encoder") as load:
                with self.assertRaisesRegex(ValueError, "恢复配置不一致"):
                    train(candidate, data, output, resume=str(output / "checkpoint-0"))
                load.assert_not_called()
            self.assertEqual(read_json(output / "settings.json"), saved)

    def test_paired_report_distinguishes_ranking_changes_and_metric_changes(self):
        rows = [{"query_id": str(i), "query": f"query {i}", "ranking": [{"page_id": "a"}, {"page_id": "b"}],
                 "nDCG@10": 1., "Recall@5": 1., "Recall@10": 1.} for i in range(200)]
        candidate = [{**row, "ranking": list(reversed(row["ranking"]))} for row in rows]
        candidate[0] = {**candidate[0], "nDCG@10": .5}
        summary, details = paired({"results": rows}, {"results": candidate})
        self.assertEqual(summary["ranking_changed"], 200)
        self.assertEqual(summary["ranking_changed_metrics_tied"], 199)
        self.assertEqual(summary["nDCG@10"]["regressed"], 1)
        self.assertAlmostEqual(summary["nDCG@10"]["mean_delta"], -.0025)
        with self.assertRaisesRegex(ValueError, "200"):
            paired({"results": rows}, {"results": candidate[:-1]})
        candidate[0] = {**candidate[0], "query": "wrong"}
        with self.assertRaisesRegex(ValueError, "原文"):
            paired({"results": rows}, {"results": candidate})


if __name__ == "__main__":
    unittest.main()
