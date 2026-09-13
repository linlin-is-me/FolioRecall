"""One explicitly enabled GPU job, with a separate ten-hour stage-four budget."""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--gpu-available', action='store_true', help='Use only after the user confirms a GPU window')
    parser.add_argument('--max-seconds', type=float, default=32400)
    parser.add_argument('--use-reserve', action='store_true', help='Only for final validation or necessary fault recovery')
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.gpu_available:
        raise ValueError('GPU时段尚未确认；本命令不会自动启动计算')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.name) or args.max_seconds <= 0 or not args.command:
        raise ValueError('运行名、时限或命令无效')
    root = Path('outputs/stage4/gpu-budget')
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
        used = sum(r['seconds'] for r in records)
        reserve = 0 if args.use_reserve else 3600
        # Reserve two minutes for cooperative checkpointing and process cleanup.
        hard_limit = min(args.max_seconds, 36000 - used - reserve)
        if hard_limit <= 120:
            raise ValueError('剩余GPU预算不足；保留缓存，等待新的资源安排')
        status = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
        if status.strip():
            raise ValueError(f'检测到已有GPU计算进程，保留该进程并暂停:\n{status}')
        output = runs / args.name
        if output.exists():
            raise ValueError('运行名已存在，请为续接或重测使用新名称')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,power.draw,temperature.gpu,driver_version', '--format=csv'], text=True)
        provenance(output, {'command': args.command, 'budget_seconds': 36000, 'used_before': used,
            'hard_limit_seconds': hard_limit, 'reserve_seconds': reserve, 'device_snapshot': gpu,
            'user_gpu_window_confirmed': True})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', FOLIORECALL_TASK_SECONDS=str(hard_limit - 120))
        started = time.monotonic()
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
            atomic_json(root / 'budget.json', {'limit_seconds': 36000, 'used_seconds': used + seconds,
                'remaining_seconds': max(0, 36000-used-seconds), 'last_run': str(output)})
        print(record, flush=True)
        return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
