import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np

from foliorecall.query import validate_query_config, retrieve
from foliorecall.search import encoding_identity, save_index
from foliorecall.io import write_json


class StudentQueryTests(unittest.TestCase):
    def setUp(self):
        self.teacher = json.loads(Path("configs/baseline.json").read_text())
        self.student = json.loads(Path("configs/student-en-cpu.json").read_text())

    def test_different_student_allowed_but_teacher_changes_rejected(self):
        validate_query_config(self.student, self.teacher)
        for key, value in (("revision", "other"), ("prompt", "other"), ("adapter", "lora"),
                           ("dimension", 1024), ("normalize", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "不匹配"):
                validate_query_config(self.student, dict(self.teacher, **{key: value}))
        self.assertEqual(encoding_identity(self.teacher), self.student["target_encoding_identity"])

    def test_missing_identity_or_foreign_tokenizer_rejected(self):
        for modified in (dict(self.student, target_encoding_identity=None), dict(self.student, tokenizer_revision="other")):
            with self.assertRaises(ValueError):
                validate_query_config(modified, self.teacher)

    def test_cli_rejects_student_identity_before_index_or_model_loading(self):
        from foliorecall.__main__ import main
        arguments = ["foliorecall", "query", "--index", "unused", "--query-config", "student.json", "question"]
        with patch.object(sys, "argv", arguments), \
             patch("foliorecall.__main__.read_json", side_effect=[self.teacher, dict(self.student, target_encoding_identity={})]), \
             patch("foliorecall.search.load_index") as index_loader, \
             patch("foliorecall.query.load_query_encoder") as model_loader, \
             self.assertRaisesRegex(ValueError, "不匹配"):
            main()
        index_loader.assert_not_called()
        model_loader.assert_not_called()

    def test_cpu_request_timing_includes_serialization(self):
        with patch("foliorecall.query.encode_query_texts", return_value=[[1]]) as encode, \
             patch("foliorecall.query.search", return_value=[[{"page_id": "a"}]]), \
             patch("foliorecall.query.time.perf_counter", side_effect=[0., 2., 3., 4.]), \
             patch("torch.cuda.synchronize") as sync:
            rows, payload, timing = retrieve(None, self.student, None, [], "question", 5)
            self.assertEqual(json.loads(payload), rows)
            self.assertEqual(timing, {"encode_seconds": 2., "query_seconds": 3., "request_seconds": 4.})
            sync.assert_not_called()
            encode.assert_called_once()

    def test_gpu_request_synchronizes_both_boundaries(self):
        with patch("foliorecall.query.encode_query_texts", return_value=[[1]]), \
             patch("foliorecall.query.search", return_value=[[{"page_id": "a"}]]), \
             patch("torch.cuda.synchronize") as sync:
            retrieve(None, dict(self.student, device="cuda"), None, [], "question")
            self.assertEqual(sync.call_count, 2)

    def test_empty_query_rejected_before_encoding(self):
        with patch("foliorecall.query.encode_query_texts") as encode, self.assertRaises(ValueError):
            retrieve(None, self.student, None, [], " ")
        encode.assert_not_called()


class AdapterIndexQueryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        preview = self.folder / "page.png"
        preview.write_bytes(b"preview placeholder; this test does not decode images")
        self.rows = [{"page_id": "a", "doc_id": "doc", "source": "local",
                      "page_number": 1, "preview": str(preview)}]
        self.historical = {"dimension": 2, "device": "cpu", "dtype": "float32",
                           "adapter": str(self.folder / "checkpoint-750")}
        self.exported = dict(self.historical, adapter=".")
        for name, config in (("historical", self.historical), ("exported", self.exported)):
            write_json(self.folder / f"{name}.json", config)
            save_index(np.array([[1., 0.]], dtype=np.float32), self.rows, config, self.folder / name)

    def test_export_adapter_path_mismatch_fails_before_loading_weights(self):
        from foliorecall.__main__ import main
        arguments = ["foliorecall", "query", "--index", str(self.folder / "historical"),
                     "--config", str(self.folder / "exported.json"), "question", "--json"]
        with patch.object(sys, "argv", arguments), \
             patch("foliorecall.query.load_query_encoder") as loader, \
             self.assertRaisesRegex(ValueError, "查询编码配置与索引不匹配"):
            main()
        loader.assert_not_called()

    def test_original_and_exported_configs_each_query_their_matching_index(self):
        from foliorecall.__main__ import main
        for name, config in (("historical", self.historical), ("exported", self.exported)):
            with self.subTest(index=name):
                arguments = ["foliorecall", "query", "--index", str(self.folder / name),
                             "--config", str(self.folder / f"{name}.json"), "question", "--json"]
                stdout = StringIO()
                with patch.object(sys, "argv", arguments), \
                     patch("foliorecall.query.load_query_encoder", return_value=object()) as loader, \
                     patch("foliorecall.query.encode_query_texts", return_value=np.array([[1., 0.]], dtype=np.float32)), \
                     redirect_stdout(stdout):
                    self.assertEqual(main(), 0)
                loader.assert_called_once_with(config)
                self.assertEqual(json.loads(stdout.getvalue()), [dict(self.rows[0], score=1.)])


if __name__ == "__main__":
    unittest.main()
