"""Prepare and run the fixed V2 teacher experiment using existing CLI commands.

Preparation and command listing never load an encoder. GPU work requires `run`.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import statistics
import subprocess
import sys

from foliorecall.benchmark_data import prepare_benchmark, validate_task
from foliorecall.data_preparation import normalized_query
from foliorecall.io import provenance, read_json, read_rows, write_json

ROOT = Path('outputs/extensions/B/heldout-seeds-v1')
SEEDS = (42, 123, 2026)
ORDER = (('ordinary', 42), ('llave', 42), ('llave', 123),
         ('ordinary', 123), ('ordinary', 2026), ('llave', 2026))
METRICS = ('nDCG@10', 'Recall@5', 'Recall@10')


def save_fixed(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f'已有准备记录不同，请使用新目录: {path}')
    write_json(path, value)


def prepare(root, suite, training_data):
    # Imports are CPU-only and deliberately absent from command listing/reporting.
    from foliorecall.trainer import batch_plan, training_dataset, validate_training_data
    root, training_data = root.resolve(), training_data.resolve()
    suite_config = read_json(suite)
    base = read_json('configs/lora-baseline.json')
    split = read_json(training_data / 'split.json')
    rows = split['train']
    if len(rows) != 256:
        raise ValueError('本轮固定256条filtered训练查询')
    mapping = validate_training_data(split, read_rows(training_data / 'pages.jsonl'),
                                    read_rows(training_data / 'dev/pages.jsonl'))
    dataset = training_dataset(rows, mapping)
    plans = {str(seed): batch_plan(dataset, 4, seed) for seed in SEEDS}
    if any(len(b) != 64 or any(len(x) != 4 for x in b) for b in plans.values()):
        raise ValueError('固定种子需要64个四样本批次')
    models = []
    original = deepcopy(base)
    original.pop('train')
    save_fixed(root / 'configs/original.json', original)
    models.append({'id': 'original', 'method': 'original', 'seed': None,
                   'encoding_config': str(root / 'configs/original.json')})
    for method, seed in ORDER:
        name = f'{method}-{seed}'
        config = deepcopy(base)
        config['seed'] = seed
        config['train'].update(hardness_mode='in_batch_negatives' if method == 'llave' else None,
                               hardness_strength=9.0 if method == 'llave' else 0.0)
        config_path = root / 'configs' / f'{name}.json'
        save_fixed(config_path, config)
        models.append({'id': name, 'method': method, 'seed': seed,
            'train_config': str(config_path),
            'encoding_config': str(root / 'training' / name / 'checkpoint-64/encoding.json')})
    manifest = {'scope': 'new project-query holdout; document-level isolation unverified',
        'suite': suite_config, 'training_data': str(training_data), 'data': str(root / 'data'),
        'seeds': list(SEEDS), 'training_steps': 64, 'models': models,
        'training_order': [f'{method}-{seed}' for method, seed in ORDER],
        'queries': sum(t['english_queries'] for t in suite_config['tasks']),
        'pages': sum(t['pages'] for t in suite_config['tasks']),
        'product_teacher_changed': False, 'new_api_calls': 0}
    save_fixed(root / 'suite.json', suite_config)
    save_fixed(root / 'batches.json', {'rows': rows, 'batches_by_seed': plans})
    save_fixed(root / 'manifest.json', manifest)
    if not (root / 'preparation-run/run.json').exists():
        provenance(root / 'preparation-run', manifest)
    return {'status': 'configs_prepared', 'models': len(models), 'training_runs': len(ORDER),
            'data_prepared': (root / 'data-preparation.json').exists(), 'training_started': False}


def audit_data(root, prior_training=Path('data/vdr-stage2'),
               prior_domains=Path('outputs/stage4/data'), metadata=Path('data/vdr/english-metadata.jsonl')):
    manifest = read_json(root / 'manifest.json')
    split = read_json(prior_training / 'split.json')
    old_queries = {'train_3000': split['train'], 'development_200': split['dev'],
        'five_domains': [r for p in prior_domains.glob('*/queries.jsonl') for r in read_rows(p)]}
    old_sets = {name: {normalized_query(r['query']) for r in rows} for name, rows in old_queries.items()}
    old_sets['vdr_english_metadata'] = {normalized_query(r.get('query')) for r in read_rows(metadata)
                                       if normalized_query(r.get('query'))}
    old_sources = {normalized_query(r.get(key)) for p in prior_domains.glob('*/pages.jsonl')
                   for r in read_rows(p) for key in ('doc_id', 'document_name', 'source') if r.get(key)}
    old_sources.update(normalized_query(r.get(key)) for r in read_rows(Path(manifest['training_data']) / 'pages.jsonl')
                       for key in ('doc_id', 'source') if r.get(key))
    tasks, all_texts = [], []
    for task in manifest['suite']['tasks']:
        data = Path(manifest['data']) / task['name']
        pages, queries, qrels = read_rows(data / 'pages.jsonl'), read_rows(data / 'queries.jsonl'), read_json(data / 'qrels.json')
        validate_task(pages, queries, qrels, task)
        from foliorecall.documents import validate_pages
        validate_pages(pages)
        text = [normalized_query(q['query']) for q in queries]
        if len(set(text)) != len(text):
            raise ValueError(f'留出集重复问题: {task["name"]}')
        all_texts.extend(text)
        overlaps = {key: [q['query_id'] for q in queries if normalized_query(q['query']) in values]
                    for key, values in old_sets.items()}
        source_hits = sorted({p['doc_id'] for p in pages if any(
            normalized_query(p.get(key)) in old_sources for key in ('source', 'document_name') if p.get(key))})
        tasks.append({'task': task['name'], 'queries': len(queries), 'pages': len(pages),
            'documents': len({p['doc_id'] for p in pages}), 'qrels': sum(map(len, qrels.values())),
            'normalized_query_overlap': overlaps, 'exact_known_source_matches': source_hits,
            'unknown_physical_page_numbers': sum(p['page_number'] is None for p in pages)})
    if len(all_texts) != len(set(all_texts)):
        raise ValueError('三个领域之间有重复查询')
    result = {'status': 'prepared', 'tasks': tasks, 'prior_query_counts': {k:len(v) for k,v in old_sets.items()},
        'queries': sum(t['queries'] for t in tasks), 'pages': sum(t['pages'] for t in tasks),
        'limitations': ['Exact query/source comparison does not prove page/document isolation',
                        'Opaque VDR image IDs prevent full original-document comparison',
                        'Upstream pretraining overlap unknown'], 'training_started': False}
    write_json(root / 'data-preparation.json', result)
    return result


def prepare_data(root):
    m = read_json(root / 'manifest.json')
    prepare_benchmark(m['suite'], m['data'], m['training_data'])
    return audit_data(root)


def commands(root, stage, model=None, task=None, resume=None):
    root = root.resolve()
    m = read_json(root / 'manifest.json')
    models = [r for r in m['models'] if model is None or r['id'] == model]
    tasks = [r for r in m['suite']['tasks'] if task is None or r['name'] == task]
    if not models or not tasks:
        raise ValueError('未知模型或领域')
    if resume and (stage != 'train' or model is None):
        raise ValueError('训练恢复须指定一个模型及stage=train')
    result = []
    prefix = [sys.executable, '-m', 'foliorecall']
    for r in models:
        if stage == 'train':
            if r['method'] == 'original':
                continue
            cmd = prefix + ['train', '--config', r['train_config'], '--data', m['training_data'],
                '--output', str(root / 'training' / r['id']), '--checkpoint-only', '--max-seconds', '3600']
            if resume:
                cmd += ['--resume', str(Path(resume).resolve())]
            result.append(cmd)
        else:
            for t in tasks:
                data = Path(m['data']) / t['name']
                index = root / 'indexes' / r['id'] / t['name']
                if stage == 'index':
                    if index.exists():
                        if not read_json(index / 'build.json')['complete']:
                            raise ValueError(f'索引目录存在但未完成: {index}')
                        continue
                    cmd = prefix + ['index', '--config', r['encoding_config'], '--pages', str(data / 'pages.jsonl'),
                        '--output', str(index), '--chunk-size', '64', '--max-seconds', '3600']
                    if (index.with_name(index.name + '.work') / 'manifest.json').is_file():
                        cmd += ['--resume']
                elif stage == 'evaluate':
                    cmd = prefix + ['evaluate', '--config', r['encoding_config'], '--index', str(index),
                        '--data', str(data), '--output', str(root / 'evaluations' / r['id'] / t['name'])]
                else:
                    raise ValueError('未知阶段')
                result.append(cmd)
    return result


def paired(reference, candidate):
    old, new = ({r['query_id']: r for r in item['results']} for item in (reference, candidate))
    if not old or len(old) != len(reference['results']) or len(new) != len(candidate['results']) or old.keys() != new.keys():
        raise ValueError('配对结果需要相同唯一查询')
    rows = []
    for qid, a in old.items():
        b = new[qid]
        if a['query'] != b['query']:
            raise ValueError('配对查询原文不一致')
        rows.append({'query_id': qid, 'query': a['query'],
            'delta': {k: b[k] - a[k] for k in METRICS},
            'reference_ranking': [p['page_id'] for p in a['ranking']],
            'candidate_ranking': [p['page_id'] for p in b['ranking']]})
    summary = {k: {'mean_delta': statistics.mean(r['delta'][k] for r in rows),
        'improved': sum(r['delta'][k] > 1e-12 for r in rows),
        'regressed': sum(r['delta'][k] < -1e-12 for r in rows),
        'tied': sum(abs(r['delta'][k]) <= 1e-12 for r in rows)} for k in METRICS}
    return summary, rows


def report(root):
    m = read_json(root / 'manifest.json')
    values, comparisons, resources, runtime = {}, {}, {}, {}
    for model in m['models']:
        if model['method'] != 'original':
            training = read_json(root / 'training' / model['id'] / 'result.json')
            settings = read_json(root / 'training' / model['id'] / 'settings.json')
            expected_config = read_json(model['train_config'])
            if (not training['complete'] or training['completed_steps'] != m['training_steps']
                    or settings['config'] != expected_config or expected_config['seed'] != model['seed']):
                raise ValueError('训练未完成或实际种子／配置不匹配')
            expected_loss = {'name': 'CachedMultipleNegativesRankingLoss', 'scale': 20.,
                'hardness_mode': 'in_batch_negatives' if model['method'] == 'llave' else None,
                'hardness_strength': 9.0 if model['method'] == 'llave' else 0.0}
            if training['loss'] != expected_loss:
                raise ValueError('实际训练损失与实验臂不一致')
            resources[model['id']] = {k:training[k] for k in (
                'invocation_seconds', 'model_loading_seconds', 'step_training_seconds', 'peak_cuda_bytes') if k in training}
        results, domains = [], {}
        runtime[model['id']] = {}
        for t in m['suite']['tasks']:
            result = read_json(root / 'evaluations' / model['id'] / t['name'] / 'result.json')
            data = Path(m['data']) / t['name']
            expected = {q['query_id']: q['query'] for q in read_rows(data / 'queries.jsonl')}
            if (result['candidates'] != t['pages'] or len(result['results']) != len(expected)
                    or {r['query_id']: r['query'] for r in result['results']} != expected):
                raise ValueError('评测数量、查询或完整候选库不一致')
            domains[t['name']] = {k: statistics.mean(r[k] for r in result['results']) for k in METRICS}
            build_path = root / 'indexes' / model['id'] / t['name'] / 'build.json'
            build = read_json(build_path) if build_path.is_file() else {}
            runtime[model['id']][t['name']] = {
                'index': {k:build[k] for k in ('seconds', 'encoding_seconds', 'model_loading_seconds',
                          'cache_write_seconds', 'index_save_seconds', 'deployment_index_bytes') if k in build},
                'query': {k:result[k] for k in ('query_seconds', 'encode_seconds', 'peak_cuda_bytes',
                          'model_loading_seconds', 'timing_scope', 'gpu') if k in result}}
            results.extend(result['results'])
        values[model['id']] = {'domains': domains, 'results': results,
            'domain_equal': {k: statistics.mean(d[k] for d in domains.values()) for k in METRICS},
            'query_weighted': {k: statistics.mean(r[k] for r in results) for k in METRICS}}
    for seed in m['seeds']:
        for a, b in [('original', f'ordinary-{seed}'), (f'ordinary-{seed}', f'llave-{seed}'), ('original', f'llave-{seed}')]:
            summary, rows = paired(values[a], values[b])
            key = f'{b}-minus-{a}'
            summary['domains'] = {t['name']: {k: values[b]['domains'][t['name']][k] - values[a]['domains'][t['name']][k]
                                            for k in METRICS} for t in m['suite']['tasks']}
            summary['domain_equal_delta'] = {k:values[b]['domain_equal'][k]-values[a]['domain_equal'][k] for k in METRICS}
            comparisons[key] = summary
            write_json(root / 'paired' / f'{key}.json', rows)
    seed_summary = {method: {scope: {k: {'values': [values[f'{method}-{s}'][scope][k] for s in m['seeds']],
        'mean': statistics.mean(values[f'{method}-{s}'][scope][k] for s in m['seeds']),
        'sample_std': statistics.stdev(values[f'{method}-{s}'][scope][k] for s in m['seeds']),
        'min': min(values[f'{method}-{s}'][scope][k] for s in m['seeds']),
        'max': max(values[f'{method}-{s}'][scope][k] for s in m['seeds'])} for k in METRICS}
        for scope in ('domain_equal', 'query_weighted')} for method in ('ordinary', 'llave')}
    consistency = {}
    for label, reference, candidate in [('ordinary-original', 'original', 'ordinary'),
                                        ('llave-ordinary', 'ordinary', 'llave'), ('llave-original', 'original', 'llave')]:
        seed_rows = []
        for seed in m['seeds']:
            a = reference if reference == 'original' else f'{reference}-{seed}'
            seed_rows.append({r['query_id']: r for r in paired(values[a], values[f'{candidate}-{seed}'])[1]})
        averaged = []
        for qid in seed_rows[0]:
            group = [rows[qid] for rows in seed_rows]
            averaged.append({'query_id': group[0]['query_id'], 'query': group[0]['query'],
                'seed_deltas': {str(s): r['delta'] for s,r in zip(m['seeds'],group)},
                'mean_delta': {k:statistics.mean(r['delta'][k] for r in group) for k in METRICS}})
        consistency[label] = {k:{
            'all_seeds_improved': sum(all(v[k] > 1e-12 for v in r['seed_deltas'].values()) for r in averaged),
            'all_seeds_regressed': sum(all(v[k] < -1e-12 for v in r['seed_deltas'].values()) for r in averaged),
            'mixed_signs': sum(any(v[k] > 1e-12 for v in r['seed_deltas'].values()) and
                               any(v[k] < -1e-12 for v in r['seed_deltas'].values()) for r in averaged)} for k in METRICS}
        write_json(root / 'paired' / f'{label}-across-seeds.json', averaged)
        write_json(root / 'cases' / f'{label}.json', {
            'improvements': sorted((r for r in averaged if r['mean_delta']['nDCG@10'] > 1e-12),
                                   key=lambda r:-r['mean_delta']['nDCG@10'])[:5],
            'regressions': sorted((r for r in averaged if r['mean_delta']['nDCG@10'] < -1e-12),
                                  key=lambda r:r['mean_delta']['nDCG@10'])[:5],
            'review_status': 'pending_original_page_review', 'reviewer_type': 'assistant', 'human': 'not_checked'})
    result = {'scope': m['scope'], 'queries': m['queries'], 'query_executions': m['queries'] * len(values),
        'seed_summary': seed_summary, 'models': {k:{a:b for a,b in v.items() if a != 'results'} for k,v in values.items()},
        'comparisons': comparisons, 'cross_seed_consistency': consistency, 'training_resources': resources,
        'runtime_resources': runtime,
        'labels': 'upstream qrels, including upstream human ESG annotations; not project assistant labels',
        'human': 'not_checked',
        'status': 'measured_pending_page_case_review_and_adoption', 'new_api_calls': 0}
    write_json(root / 'comparison.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'prepare-data', 'audit', 'commands', 'run', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--suite', type=Path, default=Path('configs/teacher-holdout-v2.json'))
    parser.add_argument('--training-data', type=Path, default=Path('data/vdr-stage2-control/filtered'))
    parser.add_argument('--stage', choices=('train', 'index', 'evaluate'))
    parser.add_argument('--model')
    parser.add_argument('--task')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.suite, args.training_data)
    elif args.command == 'prepare-data':
        result = prepare_data(args.root)
    elif args.command == 'audit':
        result = audit_data(args.root)
    elif args.command == 'report':
        result = report(args.root)
    else:
        if args.stage is None:
            parser.error('commands/run需要--stage')
        result = commands(args.root, args.stage, args.model, args.task, args.resume)
        if args.command == 'run':
            read_json(args.root / 'data-preparation.json')
            for cmd in result:
                subprocess.run(cmd, check=True)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
