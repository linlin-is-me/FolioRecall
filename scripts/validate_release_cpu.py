"""Validate local release assets in an existing isolated CPU installation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, read_rows, write_json


def install(release, workspace, python):
    output = release / 'validation'
    if output.exists() or workspace.exists():
        raise ValueError('验证输出或解压目录已存在，请保留旧结果并指定新位置')
    provenance(output, {'release': str(release), 'workspace': str(workspace),
                        'python': python, 'cuda_visible_devices': '',
                        'scope': 'CPU local release installation; public downloads pending'})
    wheel = release / 'foliorecall-0.1.0rc2-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as package:
        assert 'foliorecall = foliorecall.__main__:entrypoint' in package.read('foliorecall-0.1.0rc2.dist-info/entry_points.txt').decode()
        assert not any(name.endswith(('.pdf', '.safetensors', '.faiss')) for name in package.namelist())
        assert 'training_data=None' in package.read('foliorecall/benchmark_data.py').decode()
    with tarfile.open(release / 'foliorecall-0.1.0rc2.tar.gz') as package:
        prefix = 'foliorecall-0.1.0rc2/'
        assert not any('/outputs/' in name or name.endswith(('.pdf', '.safetensors', '.faiss')) for name in package.getnames())
        for name in ('README.md', 'doc/首版使用与评测.md'):
            text = package.extractfile(prefix + name).read().decode()
            if name.startswith('doc/'):
                text = text.split('## 历史安装与 CPU 快速体验')[0]
            for block in re.findall(r'```bash\n(.*?)```', text, re.S):
                subprocess.run(['bash', '-n'], input=block, text=True, check=True)
        assert package.getmember(prefix + 'scripts/prepare_stage4_data.py')
    with zipfile.ZipFile(release / 'demo-bundle.zip') as package:
        # Archives are our freshly assembled local assets, still reject traversal.
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in package.namelist())
        assert not any(name.endswith('.pdf') for name in package.namelist())
        package.extractall(workspace)
    demo = workspace / 'demo-bundle'
    assert len(read_rows(demo / 'pages.jsonl')) == 42
    assert '本轮尚未进行应用 GPU 验证' not in (demo / 'QUICKSTART.md').read_text()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
               HF_HOME='/mnt/d/foliorecall_cache/stage4_cpu/huggingface',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               TMPDIR='/mnt/d/foliorecall_cache/tmp')
    env.pop('PYTHONPATH', None)
    executable = str(Path(python).parent / 'foliorecall')
    commands = [
        ('install', [python, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', str(wheel)]),
        ('dependencies', [python, '-m', 'pip', 'check']),
        ('query-help', [executable, 'query', '--help']),
        ('serve-help', [executable, 'serve', '--help']),
        ('queries', [python, str(Path(__file__).with_name('validate_cpu_install.py')),
                     '--workspace', str(demo), '--output', str(output / 'queries')])]
    runs = []
    for name, command in commands:
        started = time.perf_counter()
        with (output / f'{name}.stdout.log').open('x') as log:
            result = subprocess.run(command, cwd=demo, env=env, stdout=log, stderr=subprocess.STDOUT)
        runs.append({'name': name, 'command': command, 'returncode': result.returncode,
                     'seconds': time.perf_counter() - started})
        write_json(output / 'processes.json', runs)
        if result.returncode:
            raise RuntimeError(f'{name}失败，见对应stdout.log')
        print(f'{name}: passed', flush=True)
    write_json(output / 'package-check.json', {'passed': True, 'wheel_entrypoint': True,
               'optional_training_fix_in_wheel': True, 'source_commands_bash_syntax': True,
               'pages': 42, 'workspace': str(demo), 'public_downloads': 'not yet published'})


def finish(release):
    output = release / 'validation'
    installed = read_json(output / 'queries/result.json')
    browser = read_json(output / 'browser/interaction-checks.json')
    startup = read_json(output / 'server/startup.json')
    assert installed['cli_matches_shared_retrieval'] and installed['git_independent']
    assert len(installed['results']) == 3 and installed['torch_cuda_build'] is None
    assert all(browser[key] for key in ('downloadMatchesCLI', 'physicalPage10Verified', 'previewsLoaded', 'emptyQueryRejected'))
    assert not browser['pageErrors'] and startup['model_loads'] == 1 and startup['device'] == 'cpu'
    assert len(read_rows(output / 'server/requests.jsonl')) >= 2
    assert read_json(output / 'package-check.json')['passed']
    write_json(output / 'result.json', {
        'passed': True, 'package_version': installed['version'],
        'cpu_torch': installed['torch_version'], 'cuda_visible_devices': '',
        'teacher_not_cached': True, 'student_only_cache': installed['cached_models'],
        'non_git_install': True, 'relocated_pages': 42, 'real_queries': 3,
        'cli_json_matches': True, 'browser_preview_and_json': True,
        'physical_page_10': True, 'single_resident_model': True,
        'browser_screenshot': 'browser/retrieval.png',
        'gpu_evidence': 'retained stage4 runs; no GPU computation in this round',
        'public_download_check': 'pending remote authorization and publication'})
    print(output / 'result.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--release', default='outputs/stage4/release-v0.1.0rc2-post-audit')
    parser.add_argument('--workspace', default='/mnt/d/foliorecall_cache/stage4_cpu/release-rc2-post-audit-20260915')
    parser.add_argument('--python', default='/mnt/d/foliorecall_cache/envs/stage4_cpu/bin/python')
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    if args.finish:
        finish(Path(args.release).resolve())
    else:
        install(Path(args.release).resolve(), Path(args.workspace).resolve(), args.python)
