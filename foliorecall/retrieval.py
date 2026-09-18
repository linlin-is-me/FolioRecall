"""Bound page-text indexes and fixed equal-weight reciprocal rank fusion."""
from pathlib import Path
import time

from .io import read_json, read_rows, write_json, provenance
from .evaluation import validate_candidate_corpus
from .text_baseline import tokenize, text_request, make_bm25

TOKENIZER = 'NFKC-casefold-unicode-word-v1'
PARAMS = {'k1': 1.5, 'b': 0.75, 'epsilon': 0.25}


def page_identity(pages):
    return [{k: p[k] for k in ('page_id', 'doc_id', 'page_number')} for p in pages]


def build_text_index(pages_path, texts_path, output):
    started = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise ValueError('文本索引目录已存在，请使用新目录')
    pages, texts = read_rows(pages_path), read_rows(texts_path)
    validate_candidate_corpus(pages, texts)
    mapping = {r['page_id']: r['text'] for r in texts}
    if any(not isinstance(v, str) for v in mapping.values()):
        raise ValueError('页面文本必须是字符串')
    tokens = [tokenize(mapping[p['page_id']]) for p in pages]
    nonempty = sum(bool(t) for t in tokens)
    if not nonempty:
        raise ValueError('整个页面库没有有效文本；请选择视觉检索')
    source = Path(texts_path).resolve()
    stat = source.stat()
    record = {'version': 1, 'tokenizer': TOKENIZER, 'params': PARAMS,
              'pages': page_identity(pages), 'tokens': tokens,
              'texts': [mapping[p['page_id']] for p in pages],
              'source': {'path': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
              'coverage': nonempty / len(pages), 'nonempty_pages': nonempty}
    # Validate BM25 construction before publishing a usable index.
    make_bm25(tokens, PARAMS)
    provenance(output, record | {'tokens': 'stored in index.json', 'texts': 'stored in index.json'})
    write_json(output / 'index.json', record)
    result = {'pages': len(pages), 'nonempty_pages': nonempty, 'coverage': record['coverage'],
              'index_bytes': (output / 'index.json').stat().st_size,
              'seconds': time.perf_counter() - started, 'index': str(output),
              'scope': 'text read, tokenization, BM25 construction and save; excludes upstream OCR'}
    write_json(output / 'result.json', result)
    return result


def read_retrieval_config(path):
    config = read_json(path)
    if config.get('text_index'):
        config['text_index'] = str((Path(path).resolve().parent / config['text_index']).resolve())
    return config


def load_retrieval(config, pages):
    if config is None or config == {'mode': 'visual'}:
        return None
    if config.get('mode') != 'hybrid' or config.get('rrf_k') != 60 or config.get('candidate_k') != 50:
        raise ValueError('本轮仅支持 hybrid、rrf_k=60、candidate_k=50')
    path = Path(config['text_index']) / 'index.json'
    record = read_json(path)
    if record.get('version') != 1 or record.get('tokenizer') != TOKENIZER or record.get('params') != PARAMS:
        raise ValueError('文本索引版本、分词或 BM25 参数不匹配；请重新执行 index-text')
    if record.get('pages') != page_identity(pages):
        raise ValueError('文本索引候选页面身份或顺序不匹配')
    texts, tokens = record['texts'], record['tokens']
    if len(texts) != len(pages) or len(tokens) != len(pages) or tokens != [tokenize(t) for t in texts]:
        raise ValueError('文本索引内容与页面绑定损坏')
    if not any(tokens):
        raise ValueError('整个页面库没有有效文本；请选择视觉检索')
    return {'config': config, 'model': make_bm25(tokens, PARAMS),
            'texts': dict(zip([p['page_id'] for p in pages], texts)), 'index_bytes': path.stat().st_size}


def rrf(visual, lexical, top_k=10):
    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError('混合检索 Top-k 必须在 1–50 之间')
    merged = {}
    for channel, rows in (('visual', visual), ('bm25', lexical)):
        seen = set()
        for rank, row in enumerate(rows, 1):
            pid = row['page_id']
            if pid in seen:
                raise ValueError('同一路检索出现重复页面 ID')
            seen.add(pid)
            if pid not in merged:
                merged[pid] = dict(row, score=0., score_kind='rrf', channels={})
            merged[pid]['score'] += 1. / (60 + rank)
            merged[pid]['channels'][channel] = {'rank': rank, 'score': row['score']}
    return sorted(merged.values(), key=lambda r: (r['score'], str(r['page_id'])), reverse=True)[:top_k]


def hybrid_ranking(state, pages, text, visual, top_k):
    lexical, _, _ = text_request(state['model'], pages, text, 50)
    return rrf(visual, lexical, top_k)
