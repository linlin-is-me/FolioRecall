"""Installed-wheel PDF import only. GPU model calls are written for later use."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--workspace', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo, workspace, output = Path(args.repo).resolve(), Path(args.workspace).resolve(), Path(args.output).resolve()
if workspace.exists() or output.exists():
    raise SystemExit('安装态验收目录已存在，请使用新目录')
workspace.mkdir(parents=True)
os.chdir(workspace)
import foliorecall
from foliorecall.io import provenance, read_json, write_json
if Path(foliorecall.__file__).resolve().is_relative_to(repo):
    raise RuntimeError('安装态验证意外导入源码')
state = provenance(output, {'repo': str(repo), 'workspace': str(workspace), 'operation': 'CPU PDF import; no model computation'})
assert state['commit'] is None
sources = read_json(repo / 'data/pdfs/sources.json')
cli = str(Path(sys.executable).parent / 'foliorecall')
inputs = [str(repo / 'data/pdfs' / r['doc_name']) for r in sources]
command = [cli, 'import', *inputs, '--output', str(workspace / 'pages')]
with (output / 'import.log').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
import_result = read_json(workspace / 'pages/import-result.json')
if import_result['pages'] != 45 or import_result['errors']:
    raise RuntimeError('两份PDF导入结果不符合既有45页范围')
(workspace / 'configs').mkdir()
for name in ('baseline.json', 'student-ml-cpu.json', 'student-ml-cuda.json'):
    shutil.copyfile(repo / 'configs' / name, workspace / 'configs' / name)
shutil.copyfile(repo / 'outputs/stage4/demo-bundle/queries.json', workspace / 'queries.json')
write_json(output / 'result.json', {'package_version': foliorecall.__version__, 'installed_file': foliorecall.__file__,
    'workspace': str(workspace), 'import': import_result, 'cuda_visible_devices': '',
    'index_command_pending': [cli, 'index', '--config', str(workspace / 'configs/baseline.json'),
        '--pages', str(workspace / 'pages/pages.jsonl'), '--output', str(workspace / 'index'), '--chunk-size', '64'],
    'gpu_model_status': 'not run; awaiting user GPU window'})
print(output / 'result.json')
