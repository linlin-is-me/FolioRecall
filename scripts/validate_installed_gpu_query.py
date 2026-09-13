"""Actual installed GPU student query plus CLI consistency, outside Git."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
os.chdir(args.workspace)
import torch
import faiss
import foliorecall
from foliorecall.io import provenance, read_json, write_json
from foliorecall.query import load_retriever, retrieve

if '/site-packages/' not in str(Path(foliorecall.__file__).resolve()):
    raise RuntimeError('Must use installed package')
assert torch.cuda.is_available()
torch.set_num_threads(4)
faiss.omp_set_num_threads(1)
output = Path(args.output)
if output.exists():
    raise ValueError('Output already exists')
teacher = read_json('configs/baseline.json')
student = read_json('configs/student-ml-cuda.json')
provenance(output, {'teacher': teacher, 'student': student, 'index': 'index-resumed'})
started = time.perf_counter()
model, library, pages = load_retriever('index-resumed', teacher, student)
loading = time.perf_counter() - started
queries = read_json('queries.json')
results = []
for item in queries:
    ranked, payload, timing = retrieve(model, student, library, pages, item['query'], 5)
    assert all(Path(row['preview']).is_file() for row in ranked)
    results.append(dict(item, ranking=ranked, timing=timing))
del model
torch.cuda.empty_cache()
command = [str(Path(sys.executable).parent / 'foliorecall'), 'query', '--config', 'configs/baseline.json',
    '--query-config', 'configs/student-ml-cuda.json', '--index', 'index-resumed', queries[0]['query'], '--json']
cli = subprocess.run(command, text=True, capture_output=True, check=True)
cli_rows = json.loads(cli.stdout)
assert cli_rows == results[0]['ranking']
write_json(output / 'cli-result.json', cli_rows)
write_json(output / 'result.json', {'package_version': foliorecall.__version__,
    'torch_version': torch.__version__, 'device': torch.cuda.get_device_name(),
    'loading_seconds': loading, 'cli_matches_retrieve': True, 'results': results,
    'scope': 'installation correctness; not a formal latency benchmark; teacher weights not loaded by query'})
print(output / 'result.json')
