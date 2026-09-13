"""Compare the saved four-step GPU experiments on CPU, without retraining."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, read_rows, write_json
from foliorecall.query import encode_query_texts, load_query_encoder
from foliorecall.search import load_index, search


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def check(a, b, output, index):
    a, b, output = Path(a), Path(b), Path(output)
    if output.exists():
        raise ValueError('验证目录已存在')
    provenance(output, {'continuous': str(a), 'resumed': str(b), 'index': str(index), 'device': 'cpu'})
    for root in (a, b):
        result = read_json(root / 'result.json')
        if not result['complete'] or result['completed_steps'] != 4 or result['target_steps'] != 4:
            raise ValueError('两组均须完成固定4步')
    if read_json(a / 'settings.json') != read_json(b / 'settings.json'):
        raise ValueError('连续与恢复组训练设置不同')
    names = [p.relative_to(a / 'checkpoint-4') for p in (a / 'checkpoint-4').rglob('*.safetensors')]
    from safetensors import safe_open
    count = 0
    for name in names:
        with safe_open(a / 'checkpoint-4' / name, framework='pt', device='cpu') as x, safe_open(b / 'checkpoint-4' / name, framework='pt', device='cpu') as y:
            assert list(x.keys()) == list(y.keys())
            for key in x.keys():
                if not torch.equal(x.get_tensor(key), y.get_tensor(key)):
                    raise ValueError(f'恢复后参数不逐位一致: {name}/{key}')
                count += 1
    if not count:
        raise ValueError('缺少模型参数')
    for file in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        # These are this project's own locally produced checkpoints.
        x, y = [torch.load(root / 'checkpoint-4' / file, map_location='cpu', weights_only=False) for root in (a, b)]
        if not same(x, y):
            raise ValueError(f'恢复后状态不一致: {file}')
    for file in ('trainer_state.json',):
        x, y = [read_json(root / 'checkpoint-4' / file) for root in (a, b)]
        for key in ('global_step', 'max_steps', 'epoch'):
            if x[key] != y[key]:
                raise ValueError(f'训练进度不同: {key}')
    queries = [r['query'] for r in read_rows(a / 'training-queries.jsonl')[:4]]
    torch.set_num_threads(4)
    vectors = []
    for root in (a, b):
        config = read_json(root / 'final/query-config.json')
        model = load_query_encoder(config)
        vectors.append(encode_query_texts(model, queries, config))
        del model
    if not np.array_equal(*vectors):
        raise ValueError('固定查询向量不逐位一致')
    library, pages = load_index(index, read_json(a / 'teacher-config.json'))
    rankings = [search(library, pages, v, 10) for v in vectors]
    if rankings[0] != rankings[1]:
        raise ValueError('固定查询检索结果不同')
    write_json(output / 'result.json', {'parameters_equal': count, 'optimizer_scheduler_rng_equal': True,
        'queries': queries, 'query_vectors_equal': True, 'rankings_equal': True,
        'scope': 'same-hardware four-step GPU continuity checked on CPU; not a claim of whole-training or cross-device reproducibility'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuous', required=True)
    parser.add_argument('--resumed', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', default='indexes/vdr-dev-original')
    args = parser.parse_args()
    check(args.continuous, args.resumed, args.output, args.index)
