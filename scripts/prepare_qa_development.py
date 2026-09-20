"""Freeze A4 suites and a single incremental budget; never calls a model/API."""
import argparse
import copy
import sqlite3
from pathlib import Path
from foliorecall.generation import read_agent_config
from foliorecall.io import read_json, provenance
from foliorecall.qa_runtime import prepare_suite, suite_jobs
from foliorecall.qa_storage import atomic_json, output_lock


def budget_snapshot(ledger):
    ledger = Path(ledger).resolve()
    if not ledger.is_file():
        raise ValueError('原共享账本不存在；不新建账本')
    with sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT id,cny,status FROM calls ORDER BY id')]
    return {'ledger': str(ledger), 'calls': rows,
            'occupied_cny': sum(r['cny'] for r in rows)}


def prepare(output):
    output = Path(output).resolve()
    with output_lock(output):
        if output.exists():
            raise ValueError('A4 目录已存在；不得重设费用上限')
        source = read_json('configs/qa-development-suite.json')
        agent = read_agent_config(source['common']['agent_config'])
        before = budget_snapshot(agent['ledger'])
        agent['ledger'] = before['ledger']
        agent['phase_budget_cny'] = round(min(150, before['occupied_cny']+60), 6)
        before.update(absolute_cap_cny=agent['phase_budget_cny'], new_spend_limit_cny=60,
                      do_not_rebase_on_resume=True, minimum_request_space_cny=1.06)
        provenance(output/'preparation', {'scope': 'A4 CPU only; baseline, repeats, one candidate',
                    'baseline_source': 'A3 repair-v3; keep Finance mistranslation',
                    'sequence': ['baseline240', 'review_and_freeze_candidate', 'repeat180', 'candidate']})
        atomic_json(output/'budget-before.json', before)
        atomic_json(output/'agent.json', agent)
        reports = {}
        for name, path, count in [('baseline', 'configs/qa-development-suite.json', 240),
                                  ('repeat', 'configs/qa-repeat-suite.json', 180)]:
            suite = copy.deepcopy(read_json(path))
            suite['suite_id'] = f'qa-development-v1-{name}'
            suite['common']['agent_config'] = str(output/'agent.json')
            suite['inter_task_cooldown_seconds'] = 65
            target = output/f'{name}-suite.json'
            atomic_json(target, suite)
            units, identity = prepare_suite(target)
            jobs = suite_jobs(units, suite)
            if len(jobs) != count or any(u['student']['device'] != 'cpu' for u in units):
                raise ValueError('A4 数量或 CPU 配置不符合计划')
            suite['job_order'] = [j['key'] for j in jobs]
            atomic_json(target, suite)
            reports[name] = {'planned': count, 'job_keys': suite['job_order'],
                             'complete_candidate_libraries': True}
        atomic_json(output/'preflight.json', reports)
        return {'output': str(output), 'absolute_cap_cny': agent['phase_budget_cny'],
                'baseline': 240, 'repeat': 180}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    print(prepare(p.parse_args().output))
