"""Prepare the same A3 pilot under repaired code without rebasing its cost ceiling."""
import argparse
import copy
from pathlib import Path
import sqlite3

from foliorecall.agent import ANSWER_RULES, answer_contract
from foliorecall.generation import read_agent_config
from foliorecall.io import read_json, provenance
from foliorecall.qa_runtime import prepare_suite
from foliorecall.qa_storage import atomic_json


def repair_configuration(source, output, calls, suite_id='qa-pilot-repair-v1'):
    source, output = Path(source), Path(output)
    before = read_json(source/'budget-before.json')
    agent = read_agent_config(source/'agent.json')
    if agent['phase_budget_cny'] != before['absolute_cap_cny']:
        raise ValueError('原配置与已冻结上限不一致')
    if Path(agent['ledger']).resolve() != Path(before['ledger']).resolve():
        raise ValueError('原共享账本路径不一致')
    suite = copy.deepcopy(read_json(source/'suite.json'))
    if not isinstance(suite_id,str) or not suite_id.strip():
        raise ValueError('需要非空套件 ID')
    suite['suite_id'] = suite_id
    suite['common']['agent_config'] = str(output/'agent.json')
    snapshot = {'ledger':agent['ledger'],'calls':calls,
        'occupied_cny':sum(r['cny'] for r in calls), 'absolute_cap_cny':before['absolute_cap_cny'],
        'inherited_cap_from':str(source/'budget-before.json'),'do_not_rebase_on_resume':True}
    return agent,suite,snapshot


def prepare(source, output, suite_id='qa-pilot-repair-v1', previous_run=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('修复目录已存在，拒绝覆盖或重置预算')
    cache = source/'translations/translations.json'
    if not cache.is_file():
        raise ValueError('原始来源缺少译文缓存；请使用原 pilot，比较来源通过 --previous-run 指定')
    if previous_run is not None and not (Path(previous_run)/'qa-result.json').is_file():
        raise ValueError('比较来源缺少 qa-result.json')
    original = read_agent_config(source/'agent.json')
    ledger = Path(original['ledger'])
    if not ledger.is_file():
        raise ValueError('共享账本不存在')
    with sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory = sqlite3.Row
        calls = [dict(r) for r in db.execute('SELECT * FROM calls ORDER BY at,id')]
    agent,suite,snapshot = repair_configuration(source,output,calls,suite_id)
    provenance(output/'preparation',{'source':str(source),'scope':'A3 repair, same 12 jobs; no new translation or GPU',
        'translation_cache':str(cache),'absolute_cap_cny':agent['phase_budget_cny']})
    atomic_json(output/'agent.json',agent)
    atomic_json(output/'suite.json',suite)
    atomic_json(output/'budget-before.json',snapshot)
    # Execution metadata points to the original translation, not a new API run.
    atomic_json(output/'translations/execution.json',read_json(source/'translations/execution.json')|
                {'reused_from':str(source/'translations/execution.json'),'new_translation_calls':0})
    atomic_json(output/'review-protocol.json',read_json(source/'review-protocol.json'))
    atomic_json(output/'repair.json',{'source_pilot':str(source),'translation_cache':str(cache),
        'previous_run':str(previous_run) if previous_run is not None else None,
        'answer_finalization':'separate_tool_decision_and_final_answer',
        'translation_cost_scope':'historical allocation only; excluded from new spend',
        'answer_rules':ANSWER_RULES,'contracts':{s:answer_contract(q)[1] for s,q in [('en','Question'),('zh','问题')]},
        'selection':'same previously observed tasks and ordering; no new unseen evaluation',
        'changes':['one no-tools final answer within existing budget','supported claims separate from unresolved',
                   'request phase and final transition records'],
        'A4_started':False,'GPU':False})
    units,identity = prepare_suite(output/'suite.json',cache)
    jobs = [j for u in units for j in u['jobs']]
    assert len(jobs)==12 and [j['key']['language'] for j in jobs]==['en','zh']*6
    assert [(d['name'],d['task_ids']) for d in suite['domains']]==[('hr',['hr:0']),('finance_en',['finance_en:1'])]
    assert all(u['student']['device']=='cpu' and u['agent']['ledger']==str(ledger) for u in units)
    assert all(j['translation'] for j in jobs if j['key']['language']=='zh')
    result={'passed':True,'planned':12,'cache_hits':6,'absolute_cap_cny':agent['phase_budget_cny'],
            'occupied_cny':snapshot['occupied_cny'],'new_translation_calls':0,'identity':identity}
    atomic_json(output/'preflight.json',result)
    return {k:v for k,v in result.items() if k!='identity'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='outputs/extensions/A/qa-pilot-v1')
    p.add_argument('--output',required=True)
    p.add_argument('--suite-id',default='qa-pilot-repair-v1')
    p.add_argument('--previous-run')
    a=p.parse_args()
    print(prepare(a.source,a.output,a.suite_id,a.previous_run))
