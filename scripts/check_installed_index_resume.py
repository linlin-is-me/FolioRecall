"""Compare actual GPU-produced indexes and exercise installed CPU query loading."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
os.chdir(args.workspace)
import foliorecall
from foliorecall.io import provenance, read_json, write_json
from foliorecall.query import load_query_encoder, encode_query_texts
from foliorecall.search import load_index, search
import torch

if '/site-packages/' not in str(Path(foliorecall.__file__).resolve()):
    raise RuntimeError('Probe must use the installed wheel')
output = Path(args.output)
if output.exists():
    raise ValueError('Output already exists')
config = read_json('configs/baseline.json')
provenance(output, {'workspace': str(Path.cwd()), 'config': config,
    'chunk_sizes': [64, 16], 'scope': '16-page chunks for this small pause probe only; formal tasks remain 64'})
first, pages = load_index('index', config)
second, resumed_pages = load_index('index-resumed', config)
assert pages == resumed_pages and first.ntotal == second.ntotal == 45
a, b = first.reconstruct_n(0, 45), second.reconstruct_n(0, 45)
if not np.array_equal(a, b):
    raise ValueError(f'Page vectors differ: {np.max(np.abs(a-b))}')
student = read_json('configs/student-ml-cpu.json')
torch.set_num_threads(4)
model = load_query_encoder(student)
queries = [r['query'] for r in read_json('queries.json')]
vectors = encode_query_texts(model, queries, student)
assert search(first, pages, vectors, 5) == search(second, resumed_pages, vectors, 5)
write_json(output / 'result.json', {'pages': 45, 'page_order_equal': True,
    'page_vectors_equal': True, 'max_abs_difference': 0.0, 'query_rankings_equal': True,
    'queries': queries, 'scope': 'GPU page encoding with fixed batch1; small resume probe chunks 16 vs normal 64; CPU student on same indexes'})
