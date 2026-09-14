"""Page-level upstream OCR BM25 control, without OCR or parameter tuning."""
import json
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np

from .evaluation import metrics, validate_candidate_corpus
from .benchmark_data import load_dataset_provenance
from .io import provenance, read_json, read_rows, write_json
from .query import process_memory


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def text_request(model, pages, text, top_k=10):
    started = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise ValueError('查询不能为空')
    tokens = tokenize(text)
    tokenized = time.perf_counter()
    scores = model.get_scores(tokens)
    order = sorted(range(len(pages)), key=lambda i: (float(scores[i]), str(pages[i]['page_id'])), reverse=True)[:top_k]
    ranked = [dict(pages[i], score=float(scores[i])) for i in order]
    searched = time.perf_counter()
    payload = json.dumps(ranked, ensure_ascii=False)
    return ranked, payload, {'tokenize_seconds': tokenized-started, 'query_seconds': searched-started,
                            'request_seconds': time.perf_counter()-started}


def evaluate_text(config, data, output):
    from importlib.metadata import version
    from rank_bm25 import BM25Okapi
    output, data = Path(output), Path(data)
    if output.exists():
        raise ValueError('文本评测目录已存在，请使用新输出目录')
    dataset = load_dataset_provenance(data)
    provenance(output, config)
    started = time.perf_counter()
    pages, texts = read_rows(data / 'pages.jsonl'), read_rows(data / 'texts.jsonl')
    validate_candidate_corpus(pages, texts)
    mapping = {r['page_id']: r['text'] for r in texts}
    tokens = [tokenize(mapping[r['page_id']]) for r in pages]
    nonempty = sum(bool(row) for row in tokens)
    if not nonempty:
        raise ValueError('整个页面库没有有效文本')
    params = {key: config['bm25'][key] for key in ('k1', 'b', 'epsilon')}
    model = BM25Okapi(tokens, **params)
    index_path = output / 'index.json'
    write_json(index_path, {'params': params, 'page_ids': [p['page_id'] for p in pages], 'tokens': tokens})
    build_seconds = time.perf_counter()-started
    queries, qrels = read_rows(data / 'queries.jsonl'), read_json(data / 'qrels.json')
    ids = [q['query_id'] for q in queries]
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(qrels):
        raise ValueError('查询 ID 与标签不一致')
    candidate_ids = {p['page_id'] for p in pages}
    if any(not set(rel).issubset(candidate_ids) for rel in qrels.values()):
        raise ValueError('候选库缺失标签页面')
    warmups, repeats = config['protocol']['warmups'], config['protocol']['repeats']
    for i in range(warmups):
        text_request(model, pages, queries[i % len(queries)]['query'])
    warm_memory = process_memory()
    results, requests, rounds = [], [], []
    def summarize(rows):
        return {key: {f'p{p}': float(np.percentile([r[key] for r in rows], p)) for p in (50, 95)}
                for key in ('tokenize_seconds', 'query_seconds', 'request_seconds')}
    for repeat in range(repeats):
        current = []
        for i, query in enumerate(queries):
            ranked, _, times = text_request(model, pages, query['query'])
            current.append(dict(times, repeat=repeat, query_id=query['query_id']))
            if repeat == 0:
                results.append({'query_id': query['query_id'], 'query': query['query'], 'ranking': ranked,
                    **metrics([r['page_id'] for r in ranked], qrels[query['query_id']])})
            elif [r['page_id'] for r in ranked] != [r['page_id'] for r in results[i]['ranking']]:
                raise ValueError('重复评测排序不一致')
        requests.extend(current)
        rounds.append(summarize(current))
        print(f'BM25 repeat {repeat+1}/{repeats} complete', flush=True)
    result = {'method': 'BM25Okapi on upstream page markdown', 'dataset': dataset,
        'queries': len(queries), 'candidates': len(pages), 'config': config['bm25'],
        'metrics': {key: float(np.mean([r[key] for r in results])) for key in ('nDCG@10', 'Recall@5', 'Recall@10')},
        'build_seconds': build_seconds, 'index_bytes': index_path.stat().st_size,
        'text_nonempty_pages': nonempty, 'text_coverage': nonempty/len(pages),
        'warmups': warmups, 'repeats': repeats, **summarize(requests), 'rounds': rounds,
        'warm_memory': warm_memory, 'final_memory': process_memory(),
        'rank_bm25_version': version('rank-bm25'), 'numpy_version': np.__version__,
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'), 'device': 'cpu',
        'timing_scope': 'input text to result JSON; excludes metrics; build includes text read/tokenization/IDF and index JSON save; upstream OCR cost unknown; no neural encoding',
        'results': results, 'requests': requests}
    write_json(output / 'result.json', result)
    return result
