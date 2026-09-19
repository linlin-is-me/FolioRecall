"""Freeze the two-task A3 pilot and its incremental cost ceiling; no API/model calls."""
import argparse
import copy
import sqlite3
from pathlib import Path

from foliorecall.generation import read_agent_config
from foliorecall.io import read_json, provenance
from foliorecall.qa_runtime import prepare_suite
from foliorecall.qa_storage import atomic_json


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('小试目录已存在；不能重新计算预算或覆盖记录')
    source = read_json('configs/qa-development-suite.json')
    agent = read_agent_config(source['common']['agent_config'])
    ledger = Path(agent['ledger'])
    if not ledger.is_file():
        raise ValueError('原阶段 A 账本不存在；拒绝另建空账本')
    with sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT id,cny,status FROM calls ORDER BY id')]
    occupied = sum(r['cny'] for r in rows)
    agent['phase_budget_cny'] = min(150, occupied + 5)
    suite = copy.deepcopy(source)
    suite['suite_id'] = 'qa-pilot-v1'
    suite['common']['agent_config'] = str(output/'agent.json')
    selection = {'hr':'hr:0', 'finance_en':'finance_en:1'}
    suite['domains'] = [d for d in suite['domains'] if d['name'] in selection]
    for domain in suite['domains']:
        domain['task_ids'] = [selection[domain['name']]]
    provenance(output/'preparation', {'source_suite':'configs/qa-development-suite.json',
        'selection':selection, 'new_spend_limit_cny':5,
        'scope':'A3 only; CPU; no A4, GPU, training or index building'})
    atomic_json(output/'budget-before.json', {'ledger':str(ledger), 'calls':rows,
        'occupied_cny':occupied,'absolute_cap_cny':agent['phase_budget_cny'],
        'new_spend_limit_cny':5,'do_not_rebase_on_resume':True})
    atomic_json(output/'agent.json',agent)
    atomic_json(output/'suite.json',suite)
    units, identity = prepare_suite(output/'suite.json')
    jobs = [j for unit in units for j in unit['jobs']]
    assert len(jobs) == 12 and len({j['key']['task_id'] for j in jobs}) == 2
    assert [j['key']['language'] for j in jobs] == ['en','zh']*6
    assert [(d['name'],d['task_ids']) for d in suite['domains']] == [('hr',['hr:0']),('finance_en',['finance_en:1'])]
    assert all(u['agent']['ledger'] == str(ledger) for u in units)
    assert all(u['student']['device'] == 'cpu' for u in units)
    atomic_json(output/'preflight.json',{'passed':True,'planned':12,
        'identity':identity,'job_keys':[j['key'] for j in jobs]})
    atomic_json(output/'selection.json', {'selected_before_system_answers':True,
        'reference_version':'dev-frozen-v2','reviewer':'assistant','human':'not_checked',
        'tasks':[
            {'task_id':'hr:0','reason':'Single-page direct evidence; years, populations and percentage points.',
             'corpus_pages':1110,'evidence_page_ids':['485'],'required_facts':2},
            {'task_id':'finance_en:1','reason':'Four-page evidence on dividend mechanisms and voting rights across two issuers.',
             'corpus_pages':2942,'evidence_page_ids':['2734','2686','1566','1570'],'required_facts':4}],
        'scoring_source':'reference_facts, fact_evidence and rubric; upstream reference_answer is historical only',
        'agent_behavior':'Additional search is optional when evidence is already sufficient.'})
    return {'passed':True,'planned':12,'translation_queries':2,'absolute_cap_cny':agent['phase_budget_cny']}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    print(prepare(p.parse_args().output))
