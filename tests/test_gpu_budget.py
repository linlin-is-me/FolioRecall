import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.io import write_json

spec = importlib.util.spec_from_file_location('stage4_gpu_runner', Path(__file__).resolve().parents[1] / 'scripts/run_stage4_gpu.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GPUBudgetTests(unittest.TestCase):
    def test_authorized_extension_retains_prior_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                root = Path('outputs/stage4/gpu-budget')
                write_json(root / 'runs/old/process.json', {'seconds': 35000})
                write_json(root / 'budget.json', {'limit_seconds': 71000, 'used_seconds': 0})
                with patch.object(sys, 'argv', ['runner', 'new', '--gpu-available', '--command', 'unused']), \
                     patch.object(runner.subprocess, 'check_output', return_value='external process') as gpu:
                    with self.assertRaisesRegex(ValueError, '已有GPU计算进程'):
                        runner.main()
                    gpu.assert_called_once()
                    # The historical records, not a stale cached used_seconds, govern limits.
                    write_json(root / 'budget.json', {'limit_seconds': 38000, 'used_seconds': 0})
                    with self.assertRaisesRegex(ValueError, '预算不足'):
                        runner.main()
                    gpu.assert_called_once()
            finally:
                os.chdir(previous)

    def test_no_window_never_calls_gpu_or_creates_budget(self):
        with patch.object(sys, 'argv', ['runner', 'test', '--command', 'unused']), \
             patch.object(runner.subprocess, 'check_output') as gpu:
            with self.assertRaisesRegex(ValueError, '尚未确认'):
                runner.main()
            gpu.assert_not_called()

    def test_orphan_and_reserve_block_before_device_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                record = Path('outputs/stage4/gpu-budget/runs/old')
                record.mkdir(parents=True)
                with patch.object(sys, 'argv', ['runner', 'new', '--gpu-available', '--command', 'unused']), \
                     patch.object(runner.subprocess, 'check_output') as gpu:
                    with self.assertRaisesRegex(ValueError, '未闭合'):
                        runner.main()
                    write_json(record / 'process.json', {'seconds': 35000})
                    with self.assertRaisesRegex(ValueError, '预算不足'):
                        runner.main()
                    gpu.assert_not_called()
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
