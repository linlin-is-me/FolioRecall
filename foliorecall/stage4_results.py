"""Aggregate the frozen external matrix without inventing missing results."""
from pathlib import Path
import numpy as np

from .io import read_json, write_json


METRICS = ('nDCG@10', 'Recall@5', 'Recall@10')
COMPARISONS = [('lora750', 'original'), ('ml', 'original'), ('distilled94', 'ml')]


def paired_interval(differences, seed=42, repeats=10000):
    rng = np.random.default_rng(seed)
    samples = np.zeros(repeats)
    for values in differences:
        values = np.asarray(values, dtype=float)
        # Bounded memory even for longer future tasks.
        for start in range(0, repeats, 100):
            count = min(100, repeats-start)
            samples[start:start+count] += values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1) / len(differences)
    return {'difference': float(np.mean([np.mean(v) for v in differences])),
            'lower_95': float(np.percentile(samples, 2.5)), 'upper_95': float(np.percentile(samples, 97.5)),
            'seed': seed, 'resamples': repeats, 'unit': 'paired queries resampled within each task; equal task weights'}


def summarize_matrix(manifest, output):
    manifest = read_json(manifest)
    output = Path(output)
    if output.exists():
        raise ValueError('汇总目录已存在，请指定新目录')
    from .io import provenance
    provenance(output, {'operation': 'summarize saved external rankings only', 'matrix': manifest['matrix']})
    tasks = {t['name']: t for t in manifest['tasks']}
    loaded, cells, pending, unstable = {}, [], [], []
    for cell in manifest['matrix']:
        file = Path(cell['result'])
        if not file.exists():
            pending.append(cell)
            continue
        result = read_json(file)
        task = tasks[cell['task']]
        if result['queries'] != task['english_queries'] or result['candidates'] != task['pages']:
            raise ValueError(f'正式任务范围不符: {file}')
        if result['device'] != ('cuda' if cell['device'] == 'gpu' else 'cpu') or result['dtype'] != ('bfloat16' if cell['device'] == 'gpu' else 'float32'):
            raise ValueError(f'部署条件不符: {file}')
        if result['warmups'] != 5 or result['repeats'] != 3:
            raise ValueError(f'正式计时协议不符: {file}')
        ids = [r['query_id'] for r in result['results']]
        if len(set(ids)) != len(ids) or len(ids) != result['queries']:
            raise ValueError(f'查询ID重复或缺失: {file}')
        key = (cell['task'], cell['device'], cell['candidate'])
        loaded[key] = result
        resources = {k: v for k, v in result.items() if k not in ('results', 'requests', 'dataset')}
        if cell.get('index') and Path(cell['index']).is_dir():
            from .indexing import index_sizes
            resources['historical_index_bytes'] = result.get('index_bytes')
            resources.update(index_sizes(cell['index']))
            resources['index_bytes'] = resources['deployment_index_bytes']
            resources['index_size_scope'] = 'canonical deployable files; historical measurement retained separately'
        cells.append(dict(cell, summary=resources))
        medians = [r['request_seconds']['p50'] for r in result['rounds']]
        if max(medians) > 1.2 * min(medians):
            unstable.append({'task': cell['task'], 'device': cell['device'], 'candidate': cell['candidate'],
                             'round_p50': medians, 'action': 'inspect conditions; permit one independent repeat, retain both'})
    macro = {}
    for device in ('gpu', 'cpu'):
        candidates = ['original', 'lora750', 'en', 'ml', 'distilled94'] if device == 'gpu' else ['en', 'ml', 'distilled94']
        for candidate in candidates:
            records = [loaded.get((task, device, candidate)) for task in tasks]
            if all(r is not None for r in records):
                macro[f'{device}/{candidate}'] = {m: float(np.mean([r['metrics'][m] for r in records])) for m in METRICS}
    comparisons = []
    for candidate, reference in COMPARISONS:
        differences = {m: [] for m in METRICS}
        changes, domains = [], []
        for task in tasks:
            a, b = loaded.get((task, 'gpu', candidate)), loaded.get((task, 'gpu', reference))
            if a is None or b is None:
                continue
            amap, bmap = ({r['query_id']: r for r in result['results']} for result in (a, b))
            if set(amap) != set(bmap):
                raise ValueError(f'配对查询不符: {task}')
            ordered = sorted(amap)
            domains.append(task)
            for m in METRICS:
                differences[m].append([amap[q][m] - bmap[q][m] for q in ordered])
            for q in ordered:
                changes.append({'task': task, 'query_id': q, 'query': amap[q]['query'],
                    'delta': {m: amap[q][m]-bmap[q][m] for m in METRICS},
                    'candidate_ranking': amap[q]['ranking'], 'reference_ranking': bmap[q]['ranking']})
        if not domains:
            continue
        name = candidate + '-vs-' + reference
        write_json(output / f'{name}-queries.json', changes)
        comparisons.append({'name': name, 'completed_tasks': domains,
            'better': sum(r['delta']['nDCG@10'] > 0 for r in changes),
            'worse': sum(r['delta']['nDCG@10'] < 0 for r in changes),
            'equal': sum(r['delta']['nDCG@10'] == 0 for r in changes),
            'macro_intervals': {m: paired_interval(differences[m]) for m in METRICS} if len(domains) == len(tasks) else None,
            'review_five': sorted(changes, key=lambda r: (-abs(r['delta']['nDCG@10']), r['task'], r['query_id']))[:5]})
    result = {'complete_matrix': not pending, 'cells': cells, 'pending': pending, 'macro': macro,
              'comparisons': comparisons, 'latency_review': unstable,
              'default': 'original page teacher + public ML GPU BF16; unchanged',
              'limits': 'Original labels retained. HR20 previously used for flow checks. Upstream/document overlap not fully verified. Bootstrap describes fixed-task query variation only.'}
    write_json(output / 'result.json', result)
    lines = ['# 第四阶段正式矩阵', '', f'已完成 {len(cells)}/40 项；缺失 {len(pending)} 项。缺项不计作零分，不生成不完整候选宏平均。', '',
        '| 领域 | 设备 | 候选 | nDCG@10 | Recall@5/10 | 请求P50/P95 ms |', '|---|---|---|---:|---:|---:|']
    for c in cells:
        r = c['summary']; m = r['metrics']; t = r['request_seconds']
        lines.append(f"| {c['task']} | {c['device']} | {c['candidate']} | {m['nDCG@10']:.6f} | {m['Recall@5']:.6f}/{m['Recall@10']:.6f} | {t['p50']*1000:.2f}/{t['p95']*1000:.2f} |")
    (output / 'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
