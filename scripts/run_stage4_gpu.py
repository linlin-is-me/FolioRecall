"""One explicitly enabled GPU job, preserving the authorized stage-four ledger."""
import argparse
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.indexing import atomic_json, exclusive_lock
from foliorecall.io import provenance, read_json, write_json


def remaining_window(budget, now=None, boot_id=None):
    """Monotonic window cannot be silently renewed after a reboot or clock reset."""
    window = budget.get('window_seconds')
    if window is None:
        return float('inf')
    if type(window) not in (int, float) or not math.isfinite(window) or window <= 120:
        raise ValueError('GPU窗口上限无效')
    if 'window_started_monotonic' not in budget:
        return window
    boot_id = boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    elapsed = (time.monotonic() if now is None else now) - budget['window_started_monotonic']
    if budget.get('boot_id') != boot_id or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('GPU窗口时间源已变化；不得重置授权窗口')
    return window - elapsed


def check_device(budget):
    samples = int(budget.get('idle_samples', 1))
    observed = []
    for i in range(samples):
        status = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        if status.strip():
            raise ValueError(f'检测到已有GPU计算进程，保留该进程并暂停:\n{status}')
        if 'minimum_free_mib' not in budget:
            return observed
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        lines = value.strip().splitlines()
        if len(lines) != 1:
            raise ValueError('预期单卡GPU状态，无法确认可用设备')
        free, utilization = (float(x.strip()) for x in lines[0].split(','))
        observed.append({'free_mib': free, 'utilization_percent': utilization})
        if not all(math.isfinite(v) for v in (free, utilization)) or free < budget['minimum_free_mib']:
            raise ValueError(f'GPU可用显存不足，保留其他进程: {observed[-1]}')
        if utilization <= budget.get('maximum_idle_utilization', 10):
            return observed
        if i + 1 < samples:
            time.sleep(5)
    raise ValueError(f'GPU连续采样仍繁忙，暂不启动: {observed}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--gpu-available', action='store_true', help='Use only after the user confirms a GPU window')
    parser.add_argument('--max-seconds', type=float, default=32400)
    parser.add_argument('--use-reserve', action='store_true', help='Only for final validation or necessary fault recovery')
    parser.add_argument('--budget-root', default='outputs/stage4/gpu-budget', help='Independent pre-authorized budget directory')
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.gpu_available:
        raise ValueError('GPU时段尚未确认；本命令不会自动启动计算')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.name) or args.max_seconds <= 0 or not args.command:
        raise ValueError('运行名、时限或命令无效')
    root = Path(args.budget_root)
    independent = root.resolve() != Path('outputs/stage4/gpu-budget').resolve()
    if independent and not (root / 'budget.json').is_file():
        raise ValueError('独立GPU额度必须先显式记录，禁止继承阶段四默认额度')
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / '.lock'):
        runs = root / 'runs'
        runs.mkdir(exist_ok=True)
        records = []
        for folder in runs.iterdir():
            if folder.is_dir():
                if not (folder / 'process.json').is_file():
                    raise ValueError(f'上次GPU进程未闭合，先核对PID与耗时，不能忽略其预算: {folder}')
                records.append(read_json(folder / 'process.json'))
        if any(not math.isfinite(r['seconds']) or r['seconds'] < 0 for r in records):
            raise ValueError('GPU预算记录含无效耗时，请先核对既有记录')
        used = sum(r['seconds'] for r in records)
        budget_file = root / 'budget.json'
        budget = read_json(budget_file) if budget_file.exists() else {}
        limit = budget.get('limit_seconds') if independent else budget.get('limit_seconds', 36000)
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError('GPU预算上限无效，请核对已授权账本')
        reserve = 0 if args.use_reserve else budget.get('reserve_seconds', 3600)
        if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0:
            raise ValueError('GPU保留时间无效')
        # Reserve two minutes for cooperative checkpointing and process cleanup.
        hard_limit = min(args.max_seconds, limit - used - reserve, remaining_window(budget))
        if hard_limit <= 120:
            raise ValueError('剩余GPU预算不足；保留缓存，等待新的资源安排')
        device_samples = check_device(budget)
        hard_limit = min(hard_limit, remaining_window(budget))
        if hard_limit <= 120:
            raise ValueError('GPU窗口剩余时间不足')
        output = runs / args.name
        if output.exists():
            raise ValueError('运行名已存在，请为续接或重测使用新名称')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,power.draw,temperature.gpu,driver_version', '--format=csv'], text=True)
        provenance(output, {'command': args.command, 'budget_seconds': limit, 'used_before': used,
            'hard_limit_seconds': hard_limit, 'reserve_seconds': reserve, 'device_snapshot': gpu,
            'user_gpu_window_confirmed': True, 'device_samples': device_samples, 'budget_root': str(root.resolve())})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', FOLIORECALL_TASK_SECONDS=str(hard_limit - 120))
        started = time.monotonic()
        if 'window_seconds' in budget and 'window_started_monotonic' not in budget:
            budget.update(window_started_monotonic=started, window_started_unix=time.time(),
                          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            atomic_json(budget_file, budget)
        process = None
        reason = 'completed'
        try:
            with (output / 'stdout.log').open('x') as log:
                process = subprocess.Popen(args.command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                write_json(output / 'started.json', {'pid': process.pid, 'started_unix': time.time(), 'hard_limit_seconds': hard_limit})
                try:
                    code = process.wait(timeout=hard_limit - 120)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    reason = 'time limit' if isinstance(exc, subprocess.TimeoutExpired) else 'user interruption'
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        code = process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait()
                        reason += '; forced stop after checkpoint grace'
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            seconds = time.monotonic() - started
            record = {'command': args.command, 'returncode': process.returncode if process else None,
                'seconds': seconds, 'reason': reason, 'ended_unix': time.time(),
                'scope': 'whole GPU-enabled child process including loading and idle residence; CPU/download/development excluded'}
            write_json(output / 'process.json', record)
            atomic_json(budget_file, dict(budget, limit_seconds=limit, used_seconds=used + seconds,
                remaining_seconds=max(0, limit-used-seconds), last_run=str(output)))
        print(record, flush=True)
        return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
