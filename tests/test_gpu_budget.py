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
    def test_independent_budget_missing_never_reads_legacy_or_launches(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(sys, 'argv',
                ['runner', 'a5', '--gpu-available', '--budget-root', str(Path(temp)/'a5'), '--command', 'unused']), \
                patch.object(runner.subprocess, 'check_output') as gpu:
            with self.assertRaisesRegex(ValueError, '独立GPU额度'):
                runner.main()
            gpu.assert_not_called()

    def test_monotonic_window_does_not_restart_or_survive_unknown_reboot(self):
        budget = {'window_seconds': 7200, 'window_started_monotonic': 100., 'boot_id': 'a'}
        self.assertEqual(runner.remaining_window(budget, 7000, 'a'), 300)
        self.assertEqual(runner.remaining_window(budget, 7301, 'a'), -1)
        for now, boot in ((99, 'a'), (101, 'b')):
            with self.assertRaisesRegex(ValueError, '时间源'):
                runner.remaining_window(budget, now, boot)

    def test_memory_and_busy_sampling_block_without_killing_other_work(self):
        budget = {'minimum_free_mib': 6144, 'idle_samples': 3, 'maximum_idle_utilization': 10}
        with patch.object(runner.subprocess, 'check_output', side_effect=['', '5000,0']), \
                patch.object(runner.os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, '显存不足'):
                runner.check_device(budget)
            kill.assert_not_called()
        with patch.object(runner.subprocess, 'check_output', side_effect=['', '7000,20']*3), \
                patch.object(runner.time, 'sleep') as sleep:
            with self.assertRaisesRegex(ValueError, '仍繁忙'):
                runner.check_device(budget)
            self.assertEqual(sleep.call_count, 2)

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
