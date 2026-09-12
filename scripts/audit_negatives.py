"""Score a fixed training-only pool; scores select review cases, never labels."""
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json, read_rows, write_json, write_rows, provenance


def select_pool(rows, seed=42):
    return random.Random(seed).sample(sorted(rows, key=lambda r: r['query_id']), 256)


def select_review(scored, seed=42):
    random_rows = random.Random(seed).sample(scored, 10)
    ids = {r['query_id'] for r in random_rows}
    unusual = sorted((r for r in scored if r['query_id'] not in ids),
                     key=lambda r: (r['margin'], r['query_id']))[:10]
    return [dict(r, selection='random') for r in random_rows] + [dict(r, selection='low_margin') for r in unusual]


def main():
    import numpy as np
    import torch
    from transformers import enable_full_determinism
    from foliorecall.encoding import load_encoder, encode_pages, encode_queries
    started = time.monotonic()
    output = Path('outputs/stage2/negative-audit')
    if output.exists():
        raise ValueError('Audit output already exists; inspect before resuming or repeating')
    config = read_json('configs/baseline.json')
    config['seed'] = 42
    enable_full_determinism(42)
    provenance(output, dict(config, full_determinism=True, purpose='training-negative review selection'))
    rows = select_pool(read_json('data/vdr-stage2/split.json')['train'])
    mapping = {p['page_id']: p for p in read_rows('data/vdr-stage2/pages.jsonl')}
    ids = sorted({r[k] for r in rows for k in ('positive', 'negative')})
    pages = [mapping[i] for i in ids]
    write_rows(output / 'samples.jsonl', rows)
    write_rows(output / 'pages.jsonl', pages)
    write_json(output / 'config.json', config)
    model = load_encoder(config)
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    vectors = encode_pages(model, pages, config)
    queries = encode_queries(model, [r['query'] for r in rows], config)
    lookup = {pid: i for i, pid in enumerate(ids)}
    scored = []
    for row, vector in zip(rows, queries):
        positive = float(vector @ vectors[lookup[row['positive']]])
        negative = float(vector @ vectors[lookup[row['negative']]])
        scored.append(dict(row, positive_score=positive, negative_score=negative, margin=positive-negative))
    np.savez(output / 'vectors.npz', pages=vectors, queries=queries)
    write_rows(output / 'scores.jsonl', scored)
    write_rows(output / 'review-input.jsonl', select_review(scored))
    write_json(output / 'result.json', {'complete': True, 'seed': 42, 'queries': len(rows), 'pages': len(pages),
        'model_loading_seconds': loaded-started, 'encoding_seconds': time.monotonic()-loaded,
        'invocation_seconds': time.monotonic()-started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'training-only selection for assistant review; no relevance labels inferred or changed'})


if __name__ == '__main__':
    main()
