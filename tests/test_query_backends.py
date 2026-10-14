"""Backend-specific contracts; quality and memory use real experiment artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.query import validate_backend, validate_query_config, model_file_sizes, load_query_encoder


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(Path('configs/student-ml-cpu.json').read_text())

    def test_default_and_explicit_torch_keep_representation(self):
        teacher=json.loads(Path('configs/baseline.json').read_text())
        validate_query_config(self.config,teacher)
        validate_query_config(dict(self.config,backend='torch'),teacher)

    def test_onnx_requires_cpu_local_graph_and_keeps_teacher_validation(self):
        cfg=dict(self.config,backend='onnx',model_path='package',onnx_file='onnx/model.onnx')
        validate_backend(cfg)
        for change in ({'device':'cuda'},{'backend':'unknown'},{'onnx_file':'../other.onnx'},
                       {'onnx_file':''},{'model_path':''}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_backend(dict(cfg,**change))
        with self.assertRaisesRegex(ValueError,'目标教师'):
            validate_query_config(dict(cfg,target_encoding_identity={}),json.loads(Path('configs/baseline.json').read_text()))

    def test_missing_graph_never_calls_sentence_transformer_or_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(self.config,backend='onnx',model_path=folder,onnx_file='onnx/model.onnx')
            Path(folder,'query-config.json').write_text(json.dumps(self.config))
            with patch('sentence_transformers.SentenceTransformer') as loader, self.assertRaisesRegex(ValueError,'不自动导出'):
                load_query_encoder(cfg)
            loader.assert_not_called()

    def test_package_counts_external_graph_and_fp32_heads(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for path,size in [('onnx/model.onnx',10),('onnx/model.onnx_data',20),
                              ('2_Dense/model.safetensors',30),('tokenizer.json',40)]:
                p=root/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b'x'*size)
            sizes=model_file_sizes(dict(self.config,backend='onnx',model_path=folder,onnx_file='onnx/model.onnx'))
            self.assertEqual(sizes['base_weight_bytes'],30)
            self.assertEqual(sizes['weight_file_bytes'],60)
            self.assertEqual(sizes['model_package_bytes'],100)


if __name__=='__main__':unittest.main()
