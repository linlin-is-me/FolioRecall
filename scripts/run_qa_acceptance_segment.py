"""A5 frozen batch boundaries; reuse the established QA runtime unchanged."""
import argparse
from collections import defaultdict
import contextlib
import math
import os
from pathlib import Path
import time

from foliorecall.io import read_json, provenance
from foliorecall.qa_storage import atomic_json, digest, output_lock
from foliorecall.qa_runtime import execute_plan, provider_factory, can_recover, latest_service_wait, wait_for_cooldown
from foliorecall.qa_evaluation import summarize_attempts
from scripts.prepare_qa_acceptance import DEVELOPMENT, validate_root
from scripts.run_qa_development_segment import cache_gate, segment_boundary, check_budget


def attempts(root):
    return [read_json(p) for p in sorted((Path(root)/'runs/attempts').glob('*/attempt-*.json'))]


def reviews(root):
    return [read_json(p) for p in sorted((Path(root)/'scoring').glob('*.json'))]


def attempt_digest(rows):
    return digest(sorted(rows, key=lambda a: (digest(a['key']), a['attempt'])))


def batch_boundary(root, identity, jobs, units, resume, retry_one=False):
    boundary = segment_boundary(Path(root)/'runs', identity, jobs, units, resume, retry_one)
    if not boundary['stop_after']:
        return dict(boundary, batch=None)
    task = int(boundary['next_key']['task_id'].split('-')[1])
    batch = (task-1)//8+1
    saved = boundary['saved']
    old = [a for a in saved if int(a['key']['task_id'].split('-')[1]) <= 8*(batch-1)]
    old_reviews = [r for r in reviews(root) if int(r['key']['task_id'].split('-')[1]) <= 8*(batch-1)]
    old_keys = [j['key'] for j in jobs[:48*(batch-1)]]
    summarize_attempts(old, len(old_keys), old_reviews, old_keys)
    if len(old_reviews) != len(old) or any(r.get('reviewer_type') != 'assistant' or r.get('human') != 'not_checked' for r in old_reviews):
        raise ValueError('前一批全部实际尝试尚未完成助手评审')
    bykey = defaultdict(list)
    for a in saved:
        bykey[digest(a['key'])].append(a)
    pending = []
    for j in jobs:
        rows = bykey[digest(j['key'])]
        if not rows or (len(rows) == 1 and can_recover(rows[0]['result'], j, units[j['unit']]['agent'])):
            pending.append(j)
    remaining = sum((int(j['key']['task_id'].split('-')[1])-1)//8+1 == batch for j in pending)
    limit = 1 if retry_one else min(boundary['stop_after'], remaining)
    first = [a for a in saved if a['attempt'] == 1]
    if batch == 1 and not retry_one:
        if len(first) < 6:
            limit = min(limit, 6-len(first))
        elif not (Path(root)/'checkpoints/first-six.json').exists():
            raise ValueError('首六项尚未核对；先保存只读检查点')
        else:
            checkpoint = read_json(Path(root)/'checkpoints/first-six.json')
            first_keys = {digest(j['key']) for j in jobs[:6]}
            if checkpoint['attempts_digest'] != attempt_digest([a for a in saved if digest(a['key']) in first_keys]):
                raise ValueError('首六项检查点来源变化')
    return dict(boundary, stop_after=limit, batch=batch)


def forecast(root, jobs=None, units=None):
    root = Path(root)
    from foliorecall.generation import NEW_MODEL, usage_cost, request_reservation
    config = read_json(root/'agent.json')
    migration = config['model'] == NEW_MODEL
    def priced(result):
        if not migration:
            return result['estimated_cost_cny']
        # Unknown reserves remain in the shared balance and fault allowance;
        # they are not measured token prices for future successful tasks.
        return sum(usage_cost(c['usage'], NEW_MODEL) if c.get('usage') else 0
                   for c in result.get('api_calls', []) if c.get('cost_status') != 'not_sent')
    baseline = [read_json(p) for p in (DEVELOPMENT/'baseline/attempts').glob('*/attempt-0001.json')]
    cells = defaultdict(list)
    for a in baseline:
        k = a['key']
        cells[(k['condition'], k['language'], k['domain'])].append(priced(a['result']))
    rates = defaultdict(float)
    for (condition, language, _), values in cells.items():
        rates[(condition, language)] = max(rates[(condition, language)], sum(values)/len(values))
    current = attempts(root)
    if any(a['state'] != 'finished' for a in current):
        raise ValueError('新增未知在途，禁止自动继续或外推为免费')
    observed = defaultdict(list)
    bykey = defaultdict(list)
    for a in current:
        bykey[digest(a['key'])].append(a)
        if a['attempt'] == 1:
            observed[(a['key']['condition'], a['key']['language'])].append(priced(a['result']))
    for key, values in observed.items():
        rates[key] = max(rates[key], sum(values)/len(values))
    suite = read_json(root/'suite.json')
    remaining = []
    jobmap = {digest(j['key']): j for j in jobs or []}
    for k in suite['job_order']:
        rows = bykey[digest(k)]
        if not rows:
            remaining.append(k)
        elif len(rows) == 1 and digest(k) in jobmap:
            j = jobmap[digest(k)]
            if can_recover(rows[0]['result'], j, units[j['unit']]['agent']):
                remaining.append(k)
    cache = read_json(root/'translations/translations.json') if (root/'translations/translations.json').exists() else {}
    conversion_rate = max(0.011989/40, sum(v['cost_cny'] for v in cache.values())/len(cache) if cache else 0)
    failure_allowance = 2.567539 if remaining else 0.
    probe_allowance = 0.
    if migration and remaining:
        frozen = read_json(root/'budget-before.json')
        from scripts.prepare_qa_development import budget_snapshot
        old_ids = {c['id'] for c in frozen['calls']}
        unknown = sum(c['status'] == 'reserved_unknown_charge' and c['id'] not in old_ids
                      for c in budget_snapshot(config['ledger'])['calls'])
        failure_allowance = max(0, 3-unknown)*request_reservation(NEW_MODEL, 2048)
        probe_allowance = 0. if (root/'probe/result.json').exists() else 1.
    estimate = sum(rates[(k['condition'], k['language'])] for k in remaining) + (40-len(cache))*conversion_rate + failure_allowance + probe_allowance
    return {'remaining_cny': estimate, 'evidence': {'baseline': str(DEVELOPMENT/'baseline'),
        'rates': {'/'.join(k): v for k, v in sorted(rates.items())}, 'remaining_attempts': len(remaining),
        'remaining_translations': 40-len(cache), 'failure_planning_allowance_cny': failure_allowance,
        'probe_planning_allowance_cny': probe_allowance, 'repriced_model': config['model'],
        'method': 'Maximum of baseline domain means and observed A5 condition/language means; full historical failure allowance retained while unfinished; not a guaranteed bound.'}}


def verify_first_six(root):
    root = Path(root)
    units, identity, jobs, _ = validate_root(root)
    cache_gate(units, jobs)
    keys = {digest(j['key']) for j in jobs[:6]}
    rows = [a for a in attempts(root) if digest(a['key']) in keys]
    if len([a for a in rows if a['attempt'] == 1 and a['state'] == 'finished']) != 6:
        raise ValueError('首六项尚未全部终态')
    for j in jobs[:6]:
        saved = [a for a in rows if a['key'] == j['key']]
        if len(saved) == 1 and can_recover(saved[0]['result'], j, units[j['unit']]['agent']):
            raise ValueError('首六项仍有可恢复网络故障待单独补跑')
    from scripts.report_qa_acceptance import reconcile
    ledger = reconcile(root)
    target = root/'checkpoints/first-six.json'
    if target.exists():
        raise ValueError('首六项检查点已存在')
    atomic_json(target, {'attempts_digest': attempt_digest(rows), 'first_terminal': 6, 'budget_reconciled': True,
                        'candidate_enabled': False, 'cache_hits_full_plan': 120, 'ledger': ledger})
    return {'checkpoint': str(target)}


def run(root, record, resume=False, retry_one=False, dry_run=False, max_seconds=14400):
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400:
        raise ValueError('运行段须在四小时以内')
    root = Path(root)
    started = time.monotonic()
    units, identity, jobs, _ = validate_root(root)
    gate = cache_gate(units, jobs)
    boundary = batch_boundary(root, identity, jobs, units, resume, retry_one)
    prediction = forecast(root, jobs, units)
    budget = check_budget(units[0]['agent'], read_json(root/'budget-before.json'), prediction['remaining_cny'])
    if dry_run:
        return {**gate, **{k: v for k, v in boundary.items() if k != 'saved'},
                'budget_passed': budget['passed'], 'remaining_cny': budget['remaining_cny']}
    record = Path(record)
    with output_lock(record):
        if record.exists():
            raise ValueError('段记录已存在，请用新名称')
        provenance(record, {'identity_digest': digest(identity), 'forecast': prediction,
            'boundary': {k: v for k, v in boundary.items() if k != 'saved'}, 'cache': gate})
        atomic_json(record/'budget.json', budget)
        if not budget['passed']:
            atomic_json(record/'result.json', {'stopped': 'forecast_budget_insufficient'})
            return {'stopped': 'forecast_budget_insufficient'}
        if not boundary['stop_after']:
            atomic_json(record/'result.json', {'stopped': None, 'already_complete': True})
            return {'stopped': None, 'already_complete': True}
        if not boundary['saved']:
            cache = read_json(root/'translations/translations.json')
            service, stop = latest_service_wait([{'result': cache[k]} for k in sorted(cache)])
            if not stop:
                waited, stop = wait_for_cooldown(time.monotonic(), max(65, service), started, max_seconds)
            else:
                waited = 0
            atomic_json(record/'initial-cooldown.json', {'seconds': waited, 'stopped': stop})
            if stop:
                atomic_json(record/'result.json', {'stopped': stop})
                return {'stopped': stop}
        print(f'A5 batch {boundary["batch"]}: at most {boundary["stop_after"]} attempts', flush=True)
        with (record/'run.log').open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = execute_plan(root/'runs', identity, jobs, lambda: provider_factory(units),
                [u['agent'] for u in units], resume, boundary['stop_after'], max_seconds, command_started=started)
        atomic_json(record/'result.json', result)
        return {k: result[k] for k in ('stopped', 'attempts_new', 'pending')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['forecast', 'translations', 'run', 'verify-first-six'])
    p.add_argument('--root', required=True)
    p.add_argument('--record')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-one', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-seconds', type=float, default=14400)
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_HOME='/mnt/d/foliorecall_cache/huggingface',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    root = Path(a.root)
    with output_lock(root/'scheduler'):
        if a.command == 'run':
            if not a.record and not a.dry_run:
                p.error('--record required')
            result = run(root, a.record, a.resume, a.retry_one, a.dry_run, a.max_seconds)
        elif a.command == 'verify-first-six':
            result = verify_first_six(root)
        else:
            units, identity, jobs, _ = validate_root(root, cached=False)
            prediction = forecast(root, jobs, units)
            budget = check_budget(units[0]['agent'], read_json(root/'budget-before.json'), prediction['remaining_cny'])
            if a.command == 'forecast':
                result = {'forecast': prediction, 'budget_passed': budget['passed'], 'remaining_cny': budget['remaining_cny']}
            else:
                if not budget['passed']:
                    raise ValueError('转换及剩余问答费用预测超过余额')
                if units[0]['agent'].get('translation_source_config'):
                    raise ValueError('迁移轮只复用冻结译文，不执行新的转换')
                from scripts.cache_qa_translations import execute_cache
                result = execute_cache(units, identity, root/'translations', resume=a.resume,
                                       dry_run=a.dry_run, cooldown_seconds=65)
        print(result, flush=True)
        return 75 if result.get('stopped') else 0


if __name__ == '__main__':
    raise SystemExit(main())
