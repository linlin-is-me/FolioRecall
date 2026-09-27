"""Read-only A5 quality and cost report; no candidate selection or judge API."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_runtime import ledger_snapshot, can_recover
from foliorecall.qa_evaluation import summarize_attempts, RATINGS
from scripts.prepare_qa_development import budget_snapshot
from scripts.prepare_qa_acceptance import validate_root, CAP


def reconcile(root):
    root = Path(root)
    config = read_json(root/'agent.json')
    manifests = []
    for name in ('translations', 'runs'):
        if (root/name/'manifest.json').exists():
            manifests.append((name, read_json(root/name/'manifest.json')))
    merged, executions = {}, []
    for name, manifest in manifests:
        ledger = ledger_snapshot([config], manifest['execution_id'])
        byid = {c['id']: c for c in ledger['calls']}
        if len(byid) != len(ledger['calls']) or set(byid) & set(merged):
            raise ValueError('账目ID重复或两个执行身份混用')
        linked = set()
        if name == 'runs':
            sources = [read_json(p) for p in sorted((root/'runs/attempts').glob('*/attempt-*.json'))]
            entries = [(a['key'], a['attempt'], a.get('result'), a['state']) for a in sources]
        else:
            records = read_json(root/'translations/translations.json')
            entries = [({'task_id': k}, 1, v, 'finished') for k, v in records.items()]
        unknown = []
        for key, attempt, result, state in entries:
            if state != 'finished':
                unknown.append(key)
                continue
            for call in result.get('api_calls', []):
                if call.get('cost_status') == 'not_sent':
                    if call.get('ledger_id') or call.get('cost_cny', 0):
                        raise ValueError('未发送调用出现支出')
                    continue
                lid = call['ledger_id']
                if lid not in byid or lid in linked:
                    raise ValueError('调用账目缺失或重复')
                row = byid[lid]
                ctx = json.loads(row['context'])
                if ctx.get('execution_id') != manifest['execution_id'] or ctx.get('attempt') != attempt or any(ctx.get(k) != v for k, v in key.items()):
                    raise ValueError('逐调用执行身份或运行键不匹配')
                known = call.get('usage') is not None
                expected = call['cost_cny'] if known else call['reservation_cny']
                if row['status'] != ('estimated_from_usage' if known else 'reserved_unknown_charge') or abs(row['cny']-expected) > 1e-8:
                    raise ValueError('逐调用费用或未知预留不匹配')
                linked.add(lid)
        unlinked = set(byid)-linked
        if unlinked and not unknown:
            raise ValueError('存在未归属终态记录的账目；暂停核对')
        merged.update(byid)
        executions.append({'kind': name, 'execution_id': manifest['execution_id'], 'identity_digest': manifest['identity_digest'],
                           'ledger_ids': sorted(byid), 'unknown_tasks': unknown, 'unlinked_ledger_ids': sorted(unlinked)})
    shared = budget_snapshot(config['ledger'])
    if config['phase_budget_cny'] != CAP:
        raise ValueError('固定费用上限变化')
    return {'executions': executions, 'calls': list(merged.values()), 'ledger_ids': sorted(merged),
        'known_cny': sum(c['cny'] for c in merged.values() if c['status'] == 'estimated_from_usage'),
        'unknown_reserved_cny': sum(c['cny'] for c in merged.values() if c['status'] == 'reserved_unknown_charge'),
        'occupied_cny': sum(c['cny'] for c in merged.values()),
        'shared': {'occupied_cny': shared['occupied_cny'], 'cap_cny': CAP, 'remaining_cny': CAP-shared['occupied_cny'],
                   'known_cny': sum(c['cny'] for c in shared['calls'] if c['status'] == 'estimated_from_usage'),
                   'unknown_reserved_cny': sum(c['cny'] for c in shared['calls'] if c['status'] == 'reserved_unknown_charge')}}


def grouped(rows, review_map):
    terminal = [a for a in rows if a['state'] == 'finished']
    scored = [review_map[(digest(a['key']), a['attempt'])] for a in terminal if (digest(a['key']), a['attempt']) in review_map]
    out = {'attempted': len(rows), 'terminal': len(terminal), 'unknown': len(rows)-len(terminal),
        'structurally_valid': sum(a['result']['completed'] for a in terminal), 'reviewed': len(scored),
        'missing_review': len(terminal)-len(scored), 'empty_answer': sum(not a['result']['answer']['claims'] for a in terminal),
        'termination_reasons': dict(Counter(a['result']['termination_reason'] for a in terminal)),
        'metrics': {m: {'denominator': sum(r[m] is not None for r in scored),
                        'not_applicable': sum(r[m] is None for r in scored),
                        'mean': mean([r[m] for r in scored if r[m] is not None])} for m in RATINGS}}
    language = [r for r in scored if not r.get('language_output_is_system_fallback') and r.get('language_ok') is not None]
    out['language'] = {'assessable': len(language), 'correct': sum(r['language_ok'] is True for r in language)}
    out['resources'] = {'actual_requests': sum(a['result']['actual_model_requests'] for a in terminal),
        'known_cny': sum(a['result']['estimated_cost_cny'] for a in terminal),
        'searches': sum(a['result']['searches'] for a in terminal),
        'read_pages': sum(len(a['result']['evidence']) for a in terminal),
        'timing': {name: sum(a['result'].get('timing', {}).get(name, 0) for a in terminal)
                   for name in sorted({k for a in terminal for k in a['result'].get('timing', {})})},
        'loading_seconds': sum(a['result'].get('loading_seconds', 0) for a in terminal)}
    return out


def mean(values):
    return sum(values)/len(values) if values else None


def report(root, final=False):
    root = Path(root)
    units, identity, jobs, _ = validate_root(root)
    from scripts.run_qa_development_segment import cache_gate
    cache_gate(units, jobs)
    manifest = read_json(root/'runs/manifest.json')
    if manifest['identity_digest'] != digest(identity) or manifest['job_keys'] != [j['key'] for j in jobs]:
        raise ValueError('验收运行身份或完整清单不匹配')
    rows = [read_json(p) for p in sorted((root/'runs/attempts').glob('*/attempt-*.json'))]
    reviews = [read_json(p) for p in sorted((root/'scoring').glob('*.json'))]
    if any(r.get('reviewer_type') != 'assistant' or r.get('human') != 'not_checked' for r in reviews):
        raise ValueError('必须明确助手评审与人工未检查')
    ledger = reconcile(root)
    summary = summarize_attempts(rows, 240, reviews, manifest['job_keys'], ledger_records=ledger['calls'])
    rmap = {(digest(r['key']), r['attempt']): r for r in reviews}
    first = [a for a in rows if a['attempt'] == 1]
    terminal = [a for a in first if a['state'] == 'finished']
    complete = len(terminal) == 240 and len(rows) == len(reviews) and all(a['state'] == 'finished' for a in rows)
    bykey = defaultdict(list)
    for a in rows:
        bykey[digest(a['key'])].append(a)
    recoverable_pending = []
    for j in jobs:
        saved = bykey[digest(j['key'])]
        if len(saved) == 1 and saved[0]['state'] == 'finished' and can_recover(saved[0]['result'], j, units[j['unit']]['agent']):
            recoverable_pending.append(j['key'])
    complete = complete and not recoverable_pending
    if final and not complete:
        raise ValueError('完整验收尚未完成；不得发布完整标记')
    cache = read_json(root/'translations/translations.json')
    for a in terminal:
        r = a['result']
        if not r['budget_state']['budget_snapshot_complete']:
            raise ValueError('终态缺完整预算快照')
        if a['key']['language'] == 'zh':
            if r['translation_status'] != 'cached_quality_only':
                raise ValueError('中文未全部使用冻结转换')
            # The runtime retains the original history fields, not a new conversion.
            expected = cache[a['key']['task_id']]
            history = r['translation_history']
            if any(history.get(k) != expected.get(k) for k in history):
                raise ValueError('转换历史与冻结缓存不符')
    groups = defaultdict(list)
    for a in first:
        k = a['key']
        groups['all'].append(a)
        for name, value in [('condition', k['condition']), ('language', k['language']), ('category', a['task']['category']),
                            ('condition/language/category', '/'.join((k['condition'], k['language'], a['task']['category'])))]:
            groups[f'{name}/{value}'].append(a)
    cells = defaultdict(dict)
    for a in first:
        cells[(a['key']['task_id'], a['key']['language'])][a['key']['condition']] = a
    pairs = []
    for key, conditions in sorted(cells.items()):
        for base, variant in [('visual-rag', 'hybrid-rag'), ('hybrid-rag', 'hybrid-agent'), ('visual-rag', 'hybrid-agent')]:
            aa, bb = conditions.get(base), conditions.get(variant)
            ra = rmap.get((digest(aa['key']), 1)) if aa else None
            rb = rmap.get((digest(bb['key']), 1)) if bb else None
            pairs.append({'task_id': key[0], 'language': key[1], 'base': base, 'variant': variant,
                'paired_reviewed': ra is not None and rb is not None,
                'delta': {m: rb[m]-ra[m] if ra and rb and ra[m] is not None and rb[m] is not None else None for m in RATINGS}})
    comparisons = {}
    for base, variant in [('visual-rag', 'hybrid-rag'), ('hybrid-rag', 'hybrid-agent'), ('visual-rag', 'hybrid-agent')]:
        selected = [p for p in pairs if (p['base'], p['variant']) == (base, variant)]
        comparisons[f'{variant}-minus-{base}'] = {m: {'common_denominator': len(v), 'mean_delta': mean(v),
            'improved': sum(x > 0 for x in v), 'regressed': sum(x < 0 for x in v), 'tied': sum(x == 0 for x in v)}
            for m in RATINGS for v in [[p['delta'][m] for p in selected if p['delta'][m] is not None]]}
    return {'status': 'complete_assistant_reviewed' if complete else 'partial', 'execution_id': manifest['execution_id'],
        'identity_digest': manifest['identity_digest'], 'summary': summary, 'groups': {k: grouped(v, rmap) for k, v in groups.items()},
        'recoveries_separate': grouped([a for a in rows if a['attempt'] == 2], rmap), 'recoverable_pending': recoverable_pending,
        'condition_pairs': pairs, 'paired_summary': comparisons, 'ledger': ledger,
        'review_digest': digest(reviews), 'cached_chinese_first': sum(a['key']['language'] == 'zh' for a in terminal),
        'translation_seconds': sum(v['seconds'] for v in cache.values()), 'human': 'not_checked',
        'scope': ['One frozen independent acceptance round; no post-hoc method or task selection.',
                  'Assistant reference and review, not human ground truth; 40 logical tasks, not 240 independent samples.',
                  'Cached query timing is not full online latency; loading, cooldown and manual waits remain separate.',
                  'First attempts include failures; recovery never replaces a first attempt.',
                  'Known charges are usage estimates; unknown reservations and historical charges stay in the shared cap.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--final', action='store_true')
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError('报告已存在，不覆盖')
    result = report(a.root, a.final)
    atomic_json(a.output, result)
    print({'status': result['status'], 'first': result['groups'].get('all'), 'ledger_occupied': result['ledger']['occupied_cny']})
