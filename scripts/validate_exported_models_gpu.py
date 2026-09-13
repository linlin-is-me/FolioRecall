"""Check export equivalence and real CLI queries against matching LoRA indexes."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json
from foliorecall.encoding import load_encoder, encode_pages
from foliorecall.query import load_query_encoder, encode_query_texts, retrieve
from foliorecall.search import load_index, save_index


def cli_query(repo, cwd, index, config, query, output, *, expect_mismatch=False):
    """Run the source CLI separately, after releasing parent model weights."""
    command = [sys.executable, '-m', 'foliorecall', 'query', '--index', str(index),
               '--config', str(config), query, '--top-k', '2', '--json']
    env = dict(os.environ, PYTHONPATH=str(repo), TOKENIZERS_PARALLELISM='false')
    if expect_mismatch:
        env['CUDA_VISIBLE_DEVICES'] = ''
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    (output / 'stdout.log').write_text(result.stdout, encoding='utf-8')
    (output / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    write_json(output / 'process.json', {'command': command, 'cwd': str(cwd),
        'returncode': result.returncode, 'seconds': seconds, 'pythonpath': str(repo),
        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
        'scope': 'source CLI whole process; includes loading; not installed-wheel or benchmark evidence'})
    if expect_mismatch:
        if result.returncode == 0 or '查询编码配置与索引不匹配' not in result.stderr:
            raise ValueError(f'历史索引未按预期拒绝导出配置: {result.stderr}')
        return None
    if result.returncode:
        raise ValueError(f'真实 CLI 查询失败，见 {output}: {result.stderr[-1000:]}')
    return json.loads(result.stdout)


parser = argparse.ArgumentParser()
parser.add_argument('--models', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
repo, models, output = Path.cwd(), Path(args.models).resolve(), Path(args.output).resolve()
if output.exists():
    raise ValueError('保留已有验证产物，请使用新目录')
provenance(output, {'models': str(models), 'operation': 'GPU export equivalence and source CLI matching-index checks; reference is existing formal HR LoRA index and original step94'})

torch.set_num_threads(4)
index_path = repo / 'outputs/stage4/indexes/hr/lora750'
historical_config_path = repo / 'outputs/stage4/remaining-preparation/configs/lora750.json'
historical_config = read_json(historical_config_path)
library, pages = load_index(index_path, historical_config)
expected_pages = library.reconstruct_n(0, 2)
queries = [r['query'] for r in read_json(repo / 'outputs/stage4/demo-bundle/queries.json')]
query = queries[0]
cli_query(repo, models / 'lora750', index_path, 'encoding.json', query,
          output / 'historical-index-export-config-rejected', expect_mismatch=True)
os.chdir(models / 'lora750')
config = read_json('encoding.json')
model = load_encoder(config)
actual_pages = encode_pages(model, pages[:2], config)
if not np.array_equal(expected_pages, actual_pages):
    raise ValueError(f'导出LoRA页面向量不一致: {np.max(np.abs(expected_pages-actual_pages))}')

portable_index_path = output / 'lora-query-index'
save_index(actual_pages, pages[:2], config, portable_index_path)
portable_index, portable_pages = load_index(portable_index_path, config)
portable_ranking, portable_payload, _ = retrieve(model, config, portable_index, portable_pages, query, 2)
historical_ranking, historical_payload, _ = retrieve(model, historical_config, library, pages, query, 2)
write_json(output / 'shared-retrieval.json', {'query': query, 'top_k': 2,
    'portable_index': str(portable_index_path), 'portable_config': config,
    'portable_ranking': portable_ranking, 'portable_payload': portable_payload,
    'historical_index': str(index_path), 'historical_config': historical_config,
    'historical_ranking': historical_ranking, 'historical_payload': historical_payload})
del model, library, portable_index
gc.collect()
torch.cuda.empty_cache()
# Both commands enter load_retriever() through the real CLI. Parent holds no model.
portable_cli = cli_query(repo, models / 'lora750', portable_index_path, 'encoding.json', query,
                         output / 'portable-cli')
historical_cli = cli_query(repo, repo, index_path, historical_config_path, query,
                           output / 'historical-cli')
if portable_cli != portable_ranking or historical_cli != historical_ranking:
    raise ValueError('真实 CLI 与共用 retrieve 的结果 JSON 不一致')

reference = read_json(repo / 'outputs/stage3/distill-3000/checkpoint-94/query-config.json')
source = load_query_encoder(reference)
expected = encode_query_texts(source, queries, reference)
del source
gc.collect()
torch.cuda.empty_cache()
os.chdir(models / 'distilled94')
config = read_json('query-gpu.json')
model = load_query_encoder(config)
actual = encode_query_texts(model, queries, config)
if not np.array_equal(expected, actual):
    raise ValueError(f'导出学生GPU查询向量不一致: {np.max(np.abs(expected-actual))}')
write_json(output / 'result.json', {'lora_page_vectors_equal': True, 'pages': [r['page_id'] for r in pages[:2]],
    'portable_index': str(portable_index_path), 'portable_cli_matches_shared_retrieval': True,
    'historical_cli_matches_shared_retrieval': True, 'historical_index_rejects_export_config': True,
    'cli_query': query, 'cli_top_k': 2,
    'adapter_compatibility': 'historical indexes retain their original adapter path; export encoding.json is paired with indexes built using that config',
    'student_query_vectors_equal': True, 'queries': queries, 'gpu': torch.cuda.get_device_name(),
    'dtype': 'bfloat16', 'scope': 'GPU export equivalence and source CLI integration, with a new two-page index and historical HR index; not a new quality experiment or installed-wheel validation'})
print(output / 'result.json')
