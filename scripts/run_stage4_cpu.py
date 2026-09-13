"""Record one CPU-only task; stage-three budget and outputs are untouched."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, write_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path('outputs/stage4/runs') / args.name
    if output.exists():
        raise SystemExit('运行名已存在，请使用新名称')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise SystemExit('缺少命令')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    provenance(output, {'command': command, 'device': 'cpu', 'cuda_visible_devices': ''})
    started = time.perf_counter()
    with (output / 'stdout.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            code = process.wait()
    record = {'command': command, 'returncode': code, 'seconds': time.perf_counter()-started,
        'cuda_visible_devices': '', 'scope': 'whole child process including imports/loading; no GPU work',
        'ended_unix': time.time()}
    write_json(output / 'process.json', record)
    print(record, flush=True)
    print((output / 'stdout.log').read_text()[-3000:], flush=True)
    raise SystemExit(code)
