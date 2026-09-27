"""Prepare durable quality-only conversions; API costs unless --dry-run."""
import argparse
import math
import time
import uuid
from pathlib import Path
from foliorecall.generation import Bailian
from foliorecall.io import provenance, read_json
from foliorecall.qa_runtime import prepare_suite
from foliorecall.qa_storage import atomic_json, digest, output_lock
from foliorecall.qa_translation import cache_translation, validate_translation


def translation_jobs(units):
    jobs = {}
    for unit in units:
        for job in unit['jobs']:
            if job['key']['language'] != 'zh':
                continue
            item = {'question': job['question'], 'config': unit['agent']}
            task_id = job['key']['task_id']
            if task_id in jobs and jobs[task_id] != item:
                raise ValueError('同题译文配置或问题不一致')
            jobs[task_id] = item
    return jobs


def check_record(record, item):
    if not validate_translation(record, item['question'], item['config']):
        raise ValueError('译文缺少完整终态预算；不能推测或静默重译')


def execute_cache(units, identity, output, resume=False, seed_cache=None,
                  dry_run=False, provider_factory=Bailian, cooldown_seconds=0):
    if type(cooldown_seconds) not in (int, float) or not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
        raise ValueError('转换冷却必须为非负有限秒数')
    jobs = translation_jobs(units)
    seeds = read_json(seed_cache) if seed_cache else {}
    if not isinstance(seeds, dict) or not set(seeds) <= set(jobs):
        raise ValueError('种子缓存含不属于套件的任务')
    for task_id, record in seeds.items():
        check_record(record, jobs[task_id])
    binding = {'suite': identity, 'jobs': jobs, 'seed_records': seeds,
               'seed_source': str(Path(seed_cache).resolve()) if seed_cache else None}
    if cooldown_seconds:
        binding['cooldown_seconds'] = cooldown_seconds
    if dry_run:
        return {'dry_run': True, 'queries': len(jobs), 'seeded': len(seeds),
                'new_queries': len(jobs)-len(seeds)}
    output = Path(output)
    with output_lock(output):
        if output.exists():
            if not resume:
                raise ValueError('译文目录已存在；只能 --resume 接续，不覆盖')
            manifest = read_json(output/'manifest.json')
            if manifest['identity_digest'] != digest(binding):
                raise ValueError('译文恢复身份不匹配；任务、配置、种子或预算改变')
        else:
            if resume:
                raise ValueError('译文恢复目录不存在')
            provenance(output, binding)
            manifest = {'identity_digest': digest(binding), 'identity': binding,
                        'execution_id': uuid.uuid4().hex}
            atomic_json(output/'manifest.json', manifest)
            atomic_json(output/'execution.json', {'execution_id': manifest['execution_id']})
        records = {}
        expected = {digest(task_id): task_id for task_id in jobs}
        for path in sorted((output/'records').glob('*.json')):
            if path.stem not in expected:
                raise ValueError('译文目录含未知任务记录')
            task_id = expected[path.stem]
            record = read_json(path)
            check_record(record, jobs[task_id])
            if task_id in seeds and record != seeds[task_id]:
                raise ValueError('已导入种子发生变化')
            records[task_id] = record
        current = read_json(output/'current.json') if (output/'current.json').exists() else None
        # Complete atomic records survive a stale summary; an unrecorded request
        # is never inferred to be free and never retried automatically.
        if current and current['task_id'] not in records:
            raise ValueError('存在失败或未知在途转换；先核对账本，禁止自动补跑')
        for task_id, record in seeds.items():
            if task_id not in records:
                atomic_json(output/'records'/f'{digest(task_id)}.json', record)
                records[task_id] = record
        atomic_json(output/'translations.json', records)
        provider = None
        # A fresh process waits the full interval; no persisted wall clock is trusted.
        last_finished = time.monotonic()
        waited = []
        cooldown_record = output/'cooldowns'/f'{time.time_ns()}.json'
        for task_id, item in jobs.items():
            if task_id in records:
                continue
            if cooldown_seconds:
                from foliorecall.qa_runtime import latest_service_wait, wait_for_cooldown
                service, stop = latest_service_wait([{'result': records[k]} for k in jobs if k in records])
                if stop:
                    raise ValueError('转换服务等待无法解释；暂停核对')
                elapsed, stop = wait_for_cooldown(last_finished, max(cooldown_seconds, service), time.monotonic(), None)
                waited.append({'task_id': task_id, 'seconds': elapsed, 'service_seconds': service})
                atomic_json(cooldown_record, {'resume': resume, 'waits': waited})
            if provider is None:
                provider = provider_factory(item['config'])
            provider.context = {'suite': identity['suite']['suite_id'], 'task_id': task_id,
                'operation': 'translation_cache', 'execution_id': manifest['execution_id'],
                'attempt': 1, 'request_sequence': 1}
            atomic_json(output/'current.json', {'task_id': task_id, 'state': 'running'})
            try:
                record = cache_translation(item['question'], item['config'], provider)
                check_record(record, item)
            except Exception as exc:
                atomic_json(output/'failure.json', {'task_id': task_id, 'type': type(exc).__name__,
                    'call': getattr(exc, 'record', {}), 'scope': 'Stopped; reconcile ledger before continuation'})
                raise
            last_finished = time.monotonic()
            atomic_json(output/'records'/f'{digest(task_id)}.json', record)
            records[task_id] = record
            atomic_json(output/'translations.json', records)
            atomic_json(output/'current.json', {'task_id': task_id, 'state': 'finished'})
            print(f'Translations {len(records)}/{len(jobs)}: {task_id}', flush=True)
        summary = {'completed': len(records), 'planned': len(jobs), 'seeded': len(seeds),
                   'execution_id': manifest['execution_id']}
        atomic_json(output/'summary.json', summary)
        return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seed-cache')
    p.add_argument('--cooldown-seconds', type=float, default=0)
    a=p.parse_args()
    units,identity=prepare_suite(a.suite)
    print(execute_cache(units,identity,a.output,a.resume,a.seed_cache,a.dry_run,cooldown_seconds=a.cooldown_seconds))


if __name__ == '__main__':
    main()
