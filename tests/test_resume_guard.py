from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.distillation import validate_resume
from foliorecall.io import write_json, write_rows
from foliorecall.query import model_file_sizes


class ResumeGuardTests(unittest.TestCase):
    def test_gpu_memory_boundaries_are_separate_without_using_cuda(self):
        import torch
        from foliorecall.query import benchmark
        from types import SimpleNamespace
        model = torch.nn.Linear(1, 1)
        times = {'encode_seconds': .01, 'query_seconds': .02, 'request_seconds': .03}
        with patch('foliorecall.query.retrieve', return_value=([{'page_id': 'p'}], '[]', times)), \
             patch('foliorecall.query.model_file_sizes', return_value={'weight_file_bytes': 8}), \
             patch('torch.cuda.max_memory_allocated', side_effect=[300, 200]), \
             patch('torch.cuda.max_memory_reserved', side_effect=[500, 400]), \
             patch('torch.cuda.reset_peak_memory_stats') as reset, \
             patch('torch.cuda.get_device_name', return_value='mock GPU'):
            r = benchmark(model, {'device': 'cuda', 'dtype': 'bfloat16'}, SimpleNamespace(ntotal=1),
                [{'page_id': 'p'}], [{'query_id': 'q', 'query': 'test'}], {'q': {'p': 1}}, 1, 1)
        reset.assert_called_once()
        self.assertEqual(r['loading_warmup_cuda_peak'], {'allocated_bytes': 300, 'reserved_bytes': 500})
        self.assertEqual(r['peak_cuda_bytes'], 200)
        self.assertEqual(r['peak_cuda_reserved_bytes'], 400)

    def test_latest_incomplete_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{'query_id': 'a', 'query': 'question'}]
            settings = {'max_steps': 4}
            write_json(root / 'settings.json', settings)
            write_rows(root / 'training-queries.jsonl', rows)
            checkpoint = root / 'checkpoint-2'
            write_json(checkpoint / 'trainer_state.json', {'global_step': 2, 'max_steps': 4})
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                (checkpoint / name).write_bytes(b'test state')
            self.assertEqual(validate_resume(root, checkpoint, settings, rows), 2)
            with self.assertRaisesRegex(ValueError, '顺序'):
                validate_resume(root, checkpoint, settings, [dict(rows[0], query='changed')])
            with self.assertRaisesRegex(ValueError, '目标步骤'):
                validate_resume(root, checkpoint, settings, rows, 2)
            (root / 'checkpoint-3').mkdir()
            with self.assertRaisesRegex(ValueError, '最新'):
                validate_resume(root, checkpoint, settings, rows)
            write_json(root / 'result.json', {'complete': True})
            with self.assertRaisesRegex(ValueError, '已完成'):
                validate_resume(root, checkpoint, settings, rows)

    def test_model_size_includes_adapter_but_not_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model.safetensors').write_bytes(b'base')
            (root / 'tokenizer.json').write_bytes(b'{}')
            (root / 'optimizer.pt').write_bytes(b'not a deployment weight')
            adapter = root / 'adapter'
            adapter.mkdir()
            (adapter / 'adapter_model.safetensors').write_bytes(b'lora')
            (adapter / 'adapter_config.json').write_bytes(b'{}')
            result = model_file_sizes({'model_path': str(root), 'adapter': str(adapter)})
            self.assertEqual(result['weight_file_bytes'], 8)
            self.assertEqual(result['adapter_weight_bytes'], 4)
            self.assertEqual(result['model_package_bytes'], 12)


if __name__ == '__main__':
    unittest.main()
