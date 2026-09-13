from pathlib import Path
import tempfile
import unittest

from foliorecall.distillation import validate_resume
from foliorecall.io import write_json, write_rows
from foliorecall.query import model_file_sizes


class ResumeGuardTests(unittest.TestCase):
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
