"""Load portable export configs and compare to already evaluated checkpoints."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json
from foliorecall.encoding import load_encoder, encode_pages
from foliorecall.query import load_query_encoder, encode_query_texts
from foliorecall.search import load_index

parser = argparse.ArgumentParser()
parser.add_argument('--models', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo, models, output = Path.cwd(), Path(args.models).resolve(), Path(args.output).resolve()
if output.exists():
    raise ValueError('保留已有验证产物，请使用新目录')
provenance(output, {'models': str(models), 'operation': 'GPU portable package loading; reference is existing formal HR LoRA index and original step94'})
torch.set_num_threads(4)
index_path = repo / 'outputs/stage4/indexes/hr/lora750'
library, pages = load_index(index_path, read_json(index_path / 'config.json'))
expected_pages = library.reconstruct_n(0, 2)
os.chdir(models / 'lora750')
config = read_json('encoding.json')
model = load_encoder(config)
actual_pages = encode_pages(model, pages[:2], config)
if not np.array_equal(expected_pages, actual_pages):
    raise ValueError(f'导出LoRA页面向量不一致: {np.max(np.abs(expected_pages-actual_pages))}')
del model, library
torch.cuda.empty_cache()
queries = [r['query'] for r in read_json(repo / 'outputs/stage4/demo-bundle/queries.json')]
reference = read_json(repo / 'outputs/stage3/distill-3000/checkpoint-94/query-config.json')
source = load_query_encoder(reference)
expected = encode_query_texts(source, queries, reference)
del source
torch.cuda.empty_cache()
os.chdir(models / 'distilled94')
config = read_json('query-gpu.json')
model = load_query_encoder(config)
actual = encode_query_texts(model, queries, config)
if not np.array_equal(expected, actual):
    raise ValueError(f'导出学生GPU查询向量不一致: {np.max(np.abs(expected-actual))}')
write_json(output / 'result.json', {'lora_page_vectors_equal': True, 'pages': [r['page_id'] for r in pages[:2]],
    'student_query_vectors_equal': True, 'queries': queries, 'gpu': torch.cuda.get_device_name(),
    'dtype': 'bfloat16', 'scope': 'relative portable config loading on GPU; weights equal to retained fixed candidates; not a new quality experiment'})
print(output / 'result.json')
