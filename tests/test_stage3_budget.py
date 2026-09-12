import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BudgetTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "budget runner targets WSL/Linux")
    def test_timeout_is_recorded_and_clears_active_task(self):
        script = Path("scripts/run_stage3_task.py").resolve()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs/stage3"
            root.mkdir(parents=True)
            (root / "budget.json").write_text(json.dumps({"limit_seconds": 0.2, "used_seconds": 0, "tasks": []}))
            result = subprocess.run([sys.executable, str(script), "timeout-test", "--", sys.executable,
                "-c", "import time; time.sleep(10)"], cwd=temporary, capture_output=True, timeout=5,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            self.assertEqual(result.returncode, 124, result.stderr.decode())
            state = json.loads((root / "budget.json").read_text())
            self.assertNotIn("active", state)
            self.assertEqual(state["tasks"][0]["returncode"], 124)
            self.assertGreaterEqual(state["used_seconds"], 0.2)
            self.assertLess(state["used_seconds"], 2)


if __name__ == "__main__":
    unittest.main()
