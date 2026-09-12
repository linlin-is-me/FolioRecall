"""One sequential model task, recorded against the shared eight-hour budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import signal
import fcntl

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
root = Path("outputs/stage3")
root.mkdir(parents=True, exist_ok=True)
lock = (root / "budget.lock").open("a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("另一个模型任务持有预算锁")
status_path = root / "budget.json"
status = json.loads(status_path.read_text()) if status_path.exists() else {"limit_seconds": 28800, "used_seconds": 0, "tasks": []}
if status.get("active"):
    raise SystemExit("已有活动或未收尾任务，先检查进程和日志")
if any(task["name"] == args.name for task in status["tasks"]):
    raise SystemExit("任务名称已存在；新尝试使用新名称")
remaining = status["limit_seconds"] - status["used_seconds"]
if remaining <= 0:
    raise SystemExit("8小时模型计算预算已用完")
command = args.command[1:] if args.command[:1] == ["--"] else args.command
log = root / f"{args.name}.log"
started = time.monotonic()
env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
env["FOLIORECALL_TASK_SECONDS"] = str(remaining)
record = {"name": args.name, "command": command, "started_unix": time.time(), "log": str(log),
          "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", "not overridden")}
with log.open("x") as output:
    proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    status["active"] = dict(record, pid=proc.pid)
    status_path.write_text(json.dumps(status, indent=2)+"\n")
    try:
        code = proc.wait(timeout=remaining)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 130
elapsed = time.monotonic()-started
record.update(seconds=elapsed, returncode=code, ended_unix=time.time())
status["used_seconds"] += elapsed
status["tasks"].append(record)
status.pop("active", None)
status_path.write_text(json.dumps(status, indent=2)+"\n")
print(json.dumps(record), flush=True)
print(log.read_text()[-4000:], flush=True)
raise SystemExit(code)
