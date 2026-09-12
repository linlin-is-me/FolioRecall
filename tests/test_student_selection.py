from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.io import read_json, write_json


class SelectionTests(unittest.TestCase):
    def run_selection(self, scores):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            training = root / "training"
            row = {"query_id": "q", "query": "example", "nDCG@10": 0.9}
            metrics = {"nDCG@10": 0.9, "Recall@10": 1.0, "Recall@5": 1.0}
            write_json(initial / "result.json", {"metrics": metrics, "results": [row]})
            write_json(initial / "run.json", {"config": {"query_config": {}}})
            evaluations = []
            for step, (ndcg, recall) in enumerate(scores, 1):
                folder = training / f"checkpoint-{step}"
                current = dict(metrics, **{"nDCG@10": ndcg, "Recall@10": recall})
                write_json(folder / "query-config.json", {})
                write_json(folder / "dev-evaluation" / "result.json",
                           {"metrics": current, "results": [dict(row, **{"nDCG@10": ndcg})]})
                evaluations.append({"step": step, "metrics": current, "checkpoint": str(folder)})
            write_json(training / "checkpoint-evaluations.json", evaluations)
            with patch.object(sys, "argv", ["select", "--training", str(training),
                    "--initialization", str(initial), "--output", str(root / "selection")]), \
                    patch("foliorecall.io.provenance"):
                runpy.run_path("scripts/select_distilled_student.py", run_name="__main__")
            return read_json(root / "selection" / "result.json")

    def test_ties_retain_public_initialization_then_earlier_epoch(self):
        result = self.run_selection([(0.9, 1.0), (0.9, 1.0)])
        self.assertEqual(result["selected"]["step"], 0)
        self.assertEqual(result["best_trained"]["step"], 1)
        self.assertFalse(result["expansion_quality_gate"])

    def test_ndcg_gain_cannot_enable_expansion_when_recall_drops(self):
        result = self.run_selection([(0.91, 0.99), (0.905, 1.0)])
        self.assertEqual(result["selected"]["step"], 1)
        self.assertFalse(result["expansion_quality_gate"])
