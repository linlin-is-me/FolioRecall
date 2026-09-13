"""Run from outside the checkout, importing only the installed wheel."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
os.chdir(args.workspace)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import foliorecall
from foliorecall.io import provenance, read_json, write_json
from foliorecall.query import load_retriever, retrieve

if str(Path(foliorecall.__file__).resolve()).startswith('/mnt/d/myproject/FolioRecall/'):
    raise RuntimeError('验证意外导入了源码检出目录')
assert torch.version.cuda is None and not torch.cuda.is_available()
absent = {name: importlib.util.find_spec(name) is None for name in ('torchvision', 'datasets', 'peft', 'accelerate', 'pypdfium2')}
assert all(absent.values()), absent
cache = Path(os.environ['HF_HOME']) / 'hub'
models = sorted(p.name for p in cache.glob('models--*'))
assert models == ['models--nanovdr--NanoVDR-Q-DistilBERT-Qwen3VL2B-2048-ML'], models
output = Path(args.output)
if output.exists():
    raise RuntimeError('验证输出已存在')
teacher, student = read_json('configs/baseline.json'), read_json('configs/student-ml-cpu.json')
state = provenance(output, {'page_config': teacher, 'query_config': student})
assert state['commit'] is None and 'git_unavailable' in state
started = time.perf_counter()
model, index, pages = load_retriever('.', teacher, student)
loading = time.perf_counter()-started
queries = read_json('queries.json')
results = []
for item in queries:
    rows, payload, timing = retrieve(model, student, index, pages, item['query'], 5)
    assert all(Path(r['preview']).is_file() for r in rows)
    results.append(dict(item, ranking=rows, timing=timing))
command = [str(Path(sys.executable).parent / 'foliorecall'), 'query', '--index', '.',
    '--config', 'configs/baseline.json', '--query-config', 'configs/student-ml-cpu.json', queries[0]['query'], '--json']
before = time.perf_counter()
cli = subprocess.run(command, text=True, capture_output=True, check=True)
cli_seconds = time.perf_counter()-before
write_json(output / 'cli-result.json', json.loads(cli.stdout))
assert json.loads(cli.stdout) == results[0]['ranking']
summary = {'package_file': foliorecall.__file__, 'version': foliorecall.__version__,
    'torch_version': torch.__version__, 'torch_cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(), 'absent_optional_dependencies': absent,
    'cached_models': models, 'cwd': str(Path.cwd()), 'pages': index.ntotal,
    'model_loading_seconds': loading, 'single_cli_process_seconds': cli_seconds,
    'cli_matches_shared_retrieval': True, 'git_independent': True,
    'results': results, 'scope': 'installation and real query checks; timing is not the formal benchmark'}
write_json(output / 'result.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))
