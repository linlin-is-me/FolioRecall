"""Aggregate the frozen external matrix without inventing missing results."""
from pathlib import Path
import numpy as np

from .io import read_json, write_json


METRICS = ('nDCG@10', 'Recall@5', 'Recall@10')
COMPARISONS = [('lora750', 'original'), ('ml', 'original'), ('distilled94', 'ml')]


def index_costs(matrix, budget_root=Path('outputs/stage4/gpu-budget')):
    """Keep legacy timing boundaries and missing fields, and include all new segments."""
    costs, seen = [], set()
    processes = [(p, read_json(p)) for p in Path(budget_root).glob('runs/*/process.json')]
    for cell in matrix:
        if not cell.get('index'):
            continue
        key = (cell['task'], cell.get('teacher', 'original'), cell['index'])
        if key in seen:
            continue
        seen.add(key)
        folder = Path(cell['index'])
        item = dict(task=key[0], teacher=key[1], index=str(folder), complete=(folder / 'index.faiss').is_file())
        if item['complete']:
            from .indexing import index_sizes
            item.update(index_sizes(folder))
            item['build'] = read_json(folder / 'build.json') if (folder / 'build.json').is_file() else None
        matched = []
        for path, process in processes:
            command = process.get('command', [])
            if '--output' in command and Path(command[command.index('--output')+1]).resolve() == folder.resolve():
                matched.append(dict(record=str(path), **process))
        item['gpu_process_segments'] = matched
        item['gpu_process_seconds'] = sum(p['seconds'] for p in matched) if matched else None
        costs.append(item)
    return costs


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
    result['indexes'] = index_costs(manifest['matrix'])
    bm25 = {}
    for task in tasks:
        file = Path('outputs/stage4/bm25') / task / 'result.json'
        if file.is_file():
            record = read_json(file)
            if record['queries'] != tasks[task]['english_queries'] or record['candidates'] != tasks[task]['pages']:
                raise ValueError(f'BM25任务范围不符: {file}')
            bm25[task] = {k: record.get(k) for k in ('metrics', 'request_seconds', 'build_seconds', 'index_bytes',
                'text_coverage', 'warm_memory', 'final_memory')}
    result['bm25'] = bm25
    result['bm25_macro'] = {m: float(np.mean([r['metrics'][m] for r in bm25.values()])) for m in METRICS} if len(bm25) == len(tasks) else None
    result['gpu_encoding_speedups'] = []
    for task in tasks:
        teacher = loaded.get((task, 'gpu', 'original'))
        if not teacher or not teacher.get('gpu'):
            continue
        for candidate in ('en', 'ml', 'distilled94'):
            student = loaded.get((task, 'gpu', candidate))
            if student and student.get('gpu') == teacher['gpu']:
                result['gpu_encoding_speedups'].append({'task': task, 'candidate': candidate,
                    'gpu': teacher['gpu'], 'bf16_encode_p50_speedup': teacher['encode_seconds']['p50']/student['encode_seconds']['p50']})
    write_json(output / 'result.json', result)
    lines = ['# 第四阶段正式矩阵', '', f'已完成 {len(cells)}/40 项；缺失 {len(pending)} 项。缺项不计作零分，不生成不完整候选宏平均。', '',
        '| 领域 | 设备 | 候选 | nDCG@10 | Recall@5/10 | 请求P50/P95 ms |', '|---|---|---|---:|---:|---:|']
    for c in cells:
        r = c['summary']; m = r['metrics']; t = r['request_seconds']
        lines.append(f"| {c['task']} | {c['device']} | {c['candidate']} | {m['nDCG@10']:.6f} | {m['Recall@5']:.6f}/{m['Recall@10']:.6f} | {t['p50']*1000:.2f}/{t['p95']*1000:.2f} |")
    def number(value, scale=1, digits=2):
        return '未记录' if value is None else f'{value/scale:.{digits}f}'
    lines += ['', '## 分项成本', '', 'GB采用十进制；显存为PyTorch统计，不含全部CUDA上下文。CPU部署单列，缺失历史字段不补造。', '',
        '| 领域/设备/候选 | 编码 P50/P95 ms | 编码加搜索 P50/P95 ms | 加载 s | 暖态RSS/进程峰值 GB | 加载预热/暖请求 CUDA峰值 GB |',
        '|---|---:|---:|---:|---:|---:|']
    for c in cells:
        r = c['summary']
        pair = lambda key: '/'.join(number(r.get(key, {}).get(p), .001) for p in ('p50', 'p95'))
        warm = r.get('warm_memory', {}).get('rss_bytes')
        peak = r.get('final_memory', {}).get('peak_rss_bytes')
        loading_peak = (r.get('loading_warmup_cuda_peak') or {}).get('allocated_bytes')
        lines.append(f"| {c['task']}/{c['device']}/{c['candidate']} | {pair('encode_seconds')} | {pair('query_seconds')} | {number(r.get('model_loading_seconds'))} | {number(warm, 1e9)}/{number(peak, 1e9)} | {number(loading_peak, 1e9)}/{number(r.get('peak_cuda_bytes'), 1e9)} |")
    lines += ['', '| 领域/设备/候选 | 基座/适配器/总权重 MB | 完整加载包 MB | 加载预热/暖请求 CUDA保留峰值 GB |', '|---|---:|---:|---:|']
    for c in cells:
        r = c['summary']
        weights = '/'.join(number(r.get(k), 1e6) for k in ('base_weight_bytes', 'adapter_weight_bytes', 'weight_file_bytes'))
        reserved = (r.get('loading_warmup_cuda_peak') or {}).get('reserved_bytes')
        lines.append(f"| {c['task']}/{c['device']}/{c['candidate']} | {weights} | {number(r.get('model_package_bytes'), 1e6)} | {number(reserved, 1e9)}/{number(r.get('peak_cuda_reserved_bytes'), 1e9)} |")
    lines += ['', '## 页面建库', '', '| 领域/教师 | 部署索引 MB | 记录内建库 s | GPU完整进程累计 s | 测量边界 |', '|---|---:|---:|---:|---|']
    for item in result['indexes']:
        build = item.get('build') or {}
        lines.append(f"| {item['task']}/{item['teacher']} | {number(item.get('deployment_index_bytes'), 1e6)} | {number(build.get('seconds'))} | {number(item['gpu_process_seconds'])} | {build.get('timing_scope', '待建库或历史未记录')} |")
    lines += ['', '## 页面文本 BM25', '', '直接复用固定官方markdown结果；上游OCR成本未知。', '',
        '| 领域 | nDCG@10 | Recall@5/10 | 请求 P50/P95 ms |', '|---|---:|---:|---:|']
    for task, r in bm25.items():
        m, t = r['metrics'], r['request_seconds']
        lines.append(f"| {task} | {m['nDCG@10']:.6f} | {m['Recall@5']:.6f}/{m['Recall@10']:.6f} | {t['p50']*1000:.2f}/{t['p95']*1000:.2f} |")
    (output / 'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
