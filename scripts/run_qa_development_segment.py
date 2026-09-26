"""A4 foreground segments: strict caches, service cooldown and domain boundaries."""
import argparse
import contextlib
import math
import os
from pathlib import Path
import time
from foliorecall.io import read_json, provenance
from foliorecall.qa_storage import atomic_json, digest, output_lock
from foliorecall.qa_runtime import prepare_suite, suite_jobs, execute_plan, provider_factory, can_recover, latest_service_wait, wait_for_cooldown
from foliorecall.qa_translation import validate_translation
from scripts.prepare_qa_development import budget_snapshot


def cache_gate(units, jobs, repeat=False):
    if repeat:
        if len(jobs) != 180 or any(j.get('translation') for j in jobs):
            raise ValueError('波动测量要求 180 项且中文实时转换')
        return {'planned': 180, 'cache_hits': 0, 'live_translations': 90}
    zh = [j for j in jobs if j['key']['language'] == 'zh']
    if len(jobs) != 240 or len(zh) != 120 or len({j['key']['task_id'] for j in zh}) != 40:
        raise ValueError('开发套件必须完整展开 240 项及 40 条译文')
    for j in zh:
        if not j.get('translation') or not validate_translation(j['translation'], j['question'], units[j['unit']]['agent']):
            raise ValueError('开发译文缺失或预算元数据不完整；禁止实时回退')
    return {'planned': 240, 'cache_hits': 120, 'translations': 40}


def segment_boundary(output, identity, jobs, units, resume, retry_one=False):
    output = Path(output)
    if output.exists():
        if not resume:
            raise ValueError('已有运行必须显式恢复')
        manifest = read_json(output/'manifest.json')
        if manifest['identity_digest'] != digest(identity):
            raise ValueError('恢复身份不一致；禁止加载模型')
    elif resume:
        raise ValueError('恢复目录不存在')
    pending, saved = [], []
    for j in jobs:
        rows = [read_json(p) for p in sorted((output/'attempts'/digest(j['key'])).glob('attempt-*.json'))]
        if len(rows) > 2 or any(a.get('key') != j['key'] or a.get('attempt') != i+1 for i,a in enumerate(rows)):
            raise ValueError('尝试清单损坏')
        for a in rows:
            if a.get('state') != 'finished' or a.get('answer_digest') != digest(a['result']['answer']):
                raise ValueError('未知在途或终态摘要损坏；必须先核对')
            saved.append(a)
        if not rows:
            pending.append((j, 1))
        elif len(rows) == 1 and can_recover(rows[0]['result'], j, units[j['unit']]['agent']):
            pending.append((j, 2))
    if not pending:
        return {'domain': None, 'stop_after': 0, 'saved': saved}
    if pending[0][1] == 2:
        if not retry_one:
            raise ValueError('网络故障补跑必须单独使用 --retry-one')
        count = 1
    else:
        if retry_one:
            raise ValueError('下一项不是允许补跑的请求')
        domain = pending[0][0]['key']['domain']
        count = 0
        for job, attempt in pending:
            if job['key']['domain'] != domain or attempt != 1:
                break
            count += 1
    return {'domain': pending[0][0]['key']['domain'], 'stop_after': count, 'saved': saved,
            'next_key': pending[0][0]['key'], 'attempt': pending[0][1]}


def check_budget(config, frozen, estimated_remaining):
    if config['ledger'] != frozen['ledger'] or config['phase_budget_cny'] != frozen['absolute_cap_cny']:
        raise ValueError('共享账本或固定费用上限变化')
    if not math.isfinite(estimated_remaining) or estimated_remaining < 0:
        raise ValueError('剩余费用预测必须有限且非负')
    snapshot = budget_snapshot(config['ledger'])
    remaining = frozen['absolute_cap_cny']-snapshot['occupied_cny']
    snapshot.update(remaining_cny=remaining, forecast_before_margin_cny=estimated_remaining,
                    forecast_with_margin_cny=1.25*estimated_remaining,
                    passed=remaining >= 1.25*estimated_remaining+1.06)
    return snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--record', required=True)
    p.add_argument('--budget', required=True)
    p.add_argument('--translation-cache')
    p.add_argument('--continuation', help='Explicit audited never-started subset; original run stays unchanged')
    p.add_argument('--forecast', required=True, help='Frozen JSON with remaining_cny and evidence; refreshed before each batch')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-one', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--max-seconds', type=float, default=14400)
    a = p.parse_args()
    with contextlib.ExitStack() as locks:
        descriptor = read_json(a.continuation) if a.continuation else None
        if descriptor:
            locks.enter_context(output_lock(descriptor['source_directory']))
        return run_segment(a, descriptor)


def initial_continuation_wait(source_attempts, saved, started, max_seconds):
    if saved:
        return {'seconds': 0., 'stopped': None, 'basis': 'runtime_resume_cooldown'}
    service, stop = latest_service_wait(source_attempts)
    if stop:
        return {'seconds': 0., 'stopped': stop}
    elapsed, stop = wait_for_cooldown(time.monotonic(), max(65., service), started, max_seconds)
    return {'seconds': elapsed, 'stopped': stop, 'basis': 'process_monotonic', 'service_wait_seconds': service}


def run_segment(a, descriptor=None):
    started = time.monotonic()
    if not 0 < a.max_seconds <= 14400:
        raise ValueError('运行段上限须在 0 至 14400 秒之间')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        HF_HOME='/mnt/d/foliorecall_cache/huggingface', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    units, identity = prepare_suite(a.suite, a.translation_cache)
    jobs = suite_jobs(units, identity['suite'])
    if identity['suite'].get('inter_task_cooldown_seconds') != 65 or any(u['student']['device'] != 'cpu' for u in units):
        raise ValueError('A4 必须使用 CPU 及 65 秒冷却')
    gate = cache_gate(units, jobs, identity['suite'].get('repeats') == 3)
    source_attempts = []
    if descriptor:
        from scripts.qa_continuation import derive
        identity, jobs, source_attempts = derive(descriptor, identity, jobs, a.output)
        gate = dict(gate, continuation_planned=len(jobs), declared_missing=len(descriptor['excluded']))
    boundary = segment_boundary(a.output, identity, jobs, units, a.resume, a.retry_one)
    forecast, frozen = read_json(a.forecast), read_json(a.budget)
    if not forecast.get('evidence'):
        raise ValueError('费用预测必须注明已有实测来源和剩余范围')
    if a.dry_run:
        budget = check_budget(units[0]['agent'], frozen, forecast['remaining_cny'])
        print({**gate, **{k:v for k,v in boundary.items() if k != 'saved'}, 'dry_run': True,
               'budget_passed': budget['passed'], 'remaining_cny': budget['remaining_cny']})
        return 0
    record = Path(a.record)
    with output_lock(record):
        if record.exists():
            raise ValueError('运行段日志已存在；请使用新的 record 目录')
        budget = check_budget(units[0]['agent'], frozen, forecast['remaining_cny'])
        provenance(record, {'identity_digest': digest(identity), 'forecast': forecast,
                           'boundary': {k:v for k,v in boundary.items() if k != 'saved'}, 'cache': gate})
        atomic_json(record/'budget.json', budget)
        if not budget['passed']:
            atomic_json(record/'result.json', {'stopped': 'forecast_budget_insufficient'})
            return 75
        if not boundary['stop_after']:
            atomic_json(record/'result.json', {'stopped': None, 'already_complete': True})
            return 0
        if descriptor:
            initial = initial_continuation_wait(source_attempts, boundary['saved'], started, a.max_seconds)
            atomic_json(record/'initial-cooldown.json', initial)
            if initial['stopped']:
                atomic_json(record/'result.json', initial)
                return 75
        # No model or provider initialization until every read-only gate passed.
        print(f'Running {boundary["domain"]}: at most {boundary["stop_after"]} attempts', flush=True)
        with (record/'run.log').open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = execute_plan(a.output, identity, jobs, lambda: provider_factory(units),
                [u['agent'] for u in units], a.resume, boundary['stop_after'], a.max_seconds,
                command_started=started)
        atomic_json(record/'result.json', result)
        print({'stopped': result['stopped'], 'attempts_new': result['attempts_new'], 'pending': result['pending']}, flush=True)
        return 0 if result['stopped'] is None else 75


if __name__ == '__main__':
    raise SystemExit(main())
