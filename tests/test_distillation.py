from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from foliorecall.distillation import QueryAlignmentLoss, projection_parameter_names
from foliorecall.io import write_json, write_rows, read_json
from foliorecall.search import encoding_identity
from foliorecall.targets import training_queries, cache_teacher, load_targets


class ProjectionNamesTests(unittest.TestCase):
    def test_published_module_names_are_resolved_by_parameter_identity(self):
        from collections import OrderedDict
        from sentence_transformers.models import Dense
        model = nn.Sequential(OrderedDict([
            ("0_Transformer", nn.Identity()), ("1_Pooling", nn.Identity()),
            ("2_Dense", Dense(4, 4)), ("3_Dense", Dense(4, 3))]))
        self.assertEqual(projection_parameter_names(model),
                         {"2_Dense.linear.weight", "3_Dense.linear.weight"})


class TargetTests(unittest.TestCase):
    def data(self, root, count=130):
        rows = [{"query_id": str(i), "query": f"train {i}"} for i in range(count)]
        write_json(root / "split.json", {"train": rows, "dev": [{"query_id": "held", "query": "held out"}]})
        write_rows(root / "dev" / "pages.jsonl", [{"page_id": "held"}])
        return rows

    def test_train_queries_exclude_development_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.data(root, 3)
            self.assertEqual(len(training_queries(root, 3)), 3)
            split = read_json(root / "split.json")
            split["train"][0]["query"] = " HELD  OUT "
            write_json(root / "split.json", split)
            with self.assertRaisesRegex(ValueError, "交叉"):
                training_queries(root, 3)

    def test_chunk_resume_and_id_text_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.data(root / "data")
            config = {"dimension": 2, "model_id": "toy", "revision": "fixed", "normalize": True, "device": "cpu"}
            teacher = root / "teacher.json"
            write_json(teacher, {"status": "selected", "encoding_config": config,
                "index_encoding_identity": encoding_identity(config)})
            def encode(model, texts, cfg):
                return np.tile(np.array([[1, 0]], dtype=np.float32), (len(texts), 1))
            with patch("foliorecall.targets.provenance"), patch("foliorecall.encoding.load_encoder", return_value=object()), \
                 patch("foliorecall.encoding.encode_queries", side_effect=encode) as encoder:
                first = cache_teacher(str(teacher), root / "data", root / "targets", 130, 64)
                self.assertEqual(first["completed"], 64)
                second = cache_teacher(str(teacher), root / "data", root / "targets", 130)
                self.assertTrue(second["complete"])
                self.assertEqual(encoder.call_count, 3)
            query_config = {"dimension": 2, "target_encoding_identity": encoding_identity(config)}
            loaded_rows, vectors, _ = load_targets(root / "targets", query_config)
            self.assertEqual(rows, loaded_rows)
            self.assertEqual(vectors.shape, (130, 2))
            loaded_rows[0], loaded_rows[1] = loaded_rows[1], loaded_rows[0]
            write_rows(root / "targets" / "queries.jsonl", loaded_rows)
            with self.assertRaisesRegex(ValueError, "顺序不匹配"):
                load_targets(root / "targets", query_config)
            with self.assertRaisesRegex(ValueError, "教师"):
                load_targets(root / "targets", {"dimension": 2, "target_encoding_identity": {}})


class LossTests(unittest.TestCase):
    def test_alignment_updates_backbone_and_projection_without_target_gradients(self):
        class TinyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(3, 4)
                self.projection = nn.Linear(4, 2)
            def forward(self, features):
                return {"sentence_embedding": self.projection(torch.tanh(self.backbone(features["x"])))}
        torch.manual_seed(42)
        model = TinyStudent()
        objective = QueryAlignmentLoss(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        inputs = [{"x": torch.randn(4, 3)}]
        targets = torch.randn(4, 2, requires_grad=True)
        loss = objective(inputs, targets)
        loss.backward()
        self.assertIsNone(targets.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertTrue(all(not torch.equal(before[n], p) for n, p in model.named_parameters()))
        copy = TinyStudent()
        copy.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(copy(inputs[0])["sentence_embedding"], model(inputs[0])["sentence_embedding"]))

    def test_vector_labels_work_with_installed_st_trainer(self):
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer import modules as models
        with tempfile.TemporaryDirectory() as temporary:
            model = SentenceTransformer(modules=[models.BoW(["one", "two", "three"]), models.Dense(3, 2), models.Normalize()], device="cpu")
            before = model[1].linear.weight.detach().clone()
            dataset = Dataset.from_dict({"query": ["one two", "three", "one", "two three"],
                "label": [[1., 0.], [0., 1.], [1., 0.], [0., 1.]]})
            args = SentenceTransformerTrainingArguments(output_dir=temporary, max_steps=2,
                per_device_train_batch_size=2, use_cpu=True, bf16=False, learning_rate=0.01,
                save_strategy="steps", save_steps=1, report_to="none", disable_tqdm=True, remove_unused_columns=False)
            trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=QueryAlignmentLoss(model))
            trainer.train()
            self.assertFalse(torch.equal(before, model[1].linear.weight))
            checkpoint = str(Path(temporary) / "checkpoint-2")
            restored = SentenceTransformer(checkpoint, device="cpu")
            np.testing.assert_allclose(model.encode(["one"]), restored.encode(["one"]), atol=1e-6)
            resumed_args = SentenceTransformerTrainingArguments(output_dir=temporary, max_steps=3,
                per_device_train_batch_size=2, use_cpu=True, learning_rate=0.01,
                save_strategy="no", report_to="none", disable_tqdm=True, remove_unused_columns=False)
            resumed = SentenceTransformerTrainer(model=restored, args=resumed_args, train_dataset=dataset, loss=QueryAlignmentLoss(restored))
            resumed.train(resume_from_checkpoint=checkpoint)
            self.assertEqual(resumed.state.global_step, 3)


if __name__ == "__main__":
    unittest.main()
