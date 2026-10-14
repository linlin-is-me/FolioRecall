"""Small paired Agent experiment; reuse runtime, budgets, reviews and frozen corpora."""
import argparse
import copy
import json
import os
from pathlib import Path
import sqlite3

from foliorecall.io import read_json, read_rows
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_evaluation import RATINGS, summarize_attempts
from foliorecall.qa_runtime import (execute_plan, provider_factory, execution_code, can_recover,
                                   transient, ledger_snapshot)

ROOT = Path('outputs/extensions/A/evidence-coverage-v1')
A5 = Path('outputs/extensions/A/qa-acceptance-v2')
DEMO = Path('outputs/releases/v0.1.0-candidate/demo-bundle')
JRC = Path('outputs/extensions/A/data/jrc-corpus-v1')
TASKS = Path('outputs/extensions/A/data/jrc-tasks-v2/tasks.jsonl')


def attempts(root):
    return [read_json(p) for p in sorted((Path(root)/'runs/attempts').glob('*/attempt-*.json'))]


def budget(config):
    with sqlite3.connect(Path(config['ledger']).resolve().as_uri()+'?mode=ro', uri=True) as db:
        occupied = db.execute('select coalesce(sum(cny),0) from calls').fetchone()[0]
        unknown = db.execute("select coalesce(sum(cny),0) from calls where status='reserved_unknown_charge'").fetchone()[0]
    return {'cap_cny': config['phase_budget_cny'], 'occupied_cny': occupied,
            'unknown_reserved_cny': unknown, 'remaining_cny': config['phase_budget_cny']-occupied}


def task_plan():
    original = {r['task_id']: r for r in read_rows(TASKS)}
    examples = read_json(DEMO/'qa-examples.json')
    population, employment = '7e690e3c38695927a52757a7fbd5b8fd', 'd6f073c8f7ce55d59beb3dfeed585bdf'
    rows = []
    def add(number, source, language, category, session=None, question=None, facts=None, evidence=None, domain='jrc'):
        base = copy.deepcopy(original[source]) if source else {
            'answerable': True, 'rubric': 'Assess supported facts, coverage, citations and insufficiency.',
            'review': {'assistant': 'source_checked', 'human': 'not_checked'}}
        base.update(task_id=f'cov-{number:02d}', source_task_id=source, domain=domain,
                    language=language, category=category, session=session or f'S{number}',
                    question=question or base['question_'+language])
        if facts is not None: base['reference_facts'] = facts
        if evidence is not None: base['evidence_page_ids'] = evidence
        rows.append(base)
    add(1, None, 'en', 'cross_document', 'D1', examples[2]['question'], [
        'Population report projects a decline in EU working-age population in 2022–2070; migration assumptions affect its scale.',
        'Employment report describes recent EU employment as high and still growing, with slower growth; distinguish observations from long-term projections.'
    ], [population+':9', employment+':6'], 'demo42')
    add(2, None, 'zh', 'cross_document', 'D1', examples[3]['question'], [
        'Population projections cover 2022–2070, supported by population report physical pages 9–10.',
        'Employment report Part I uses data available by 1 April 2025, including 2024 Q4 and February 2025; cite the employment report itself.'
    ], [population+':9', population+':10', employment+':3', employment+':6'], 'demo42')
    add(3, 'jrc-21', 'en', 'cross_document', 'D2')
    add(4, 'jrc-21', 'zh', 'cross_document', 'D2',
        '分别列出这两份报告证据收集方法的页码；其中调查于何时进行、后续研讨会何时举行，有什么样本代表性局限？',
        original['jrc-21']['reference_facts'] + original['jrc-16']['reference_facts'])
    for n, source, language, category in [
        (5,'jrc-14','en','multi_page'), (6,'jrc-17','zh','conditions'),
        (7,'jrc-29','en','chart'), (8,'jrc-35','zh','calculation'),
        (9,'jrc-02','en','single_page'), (10,'jrc-08','zh','single_page')]:
        add(n, source, language, category)
    add(11, None, 'en', 'insufficient', question=examples[4]['question'], facts=[], evidence=[], domain='demo42')
    rows[-1]['answerable'] = False
    add(12,'jrc-38','zh','insufficient')
    confirmation = []
    for n, source, language in [(1,'jrc-22','en'), (2,'jrc-22','zh'), (3,'jrc-11','en'),
                                (4,'jrc-30','zh'), (5,'jrc-37','en'), (6,'jrc-39','zh')]:
        base = copy.deepcopy(original[source])
        base.update(task_id=f'confirm-{n:02d}', source_task_id=source, domain='jrc', language=language,
                    session='C1' if n <= 2 else f'C{n}', question=base['question_'+language])
        if n == 2:
            base['question'] = '请分别给出这两份报告中基础设施建议与部署障碍的原文证据页，并说明为何技术成熟仍不能保证顺利部署。'
        confirmation.append(base)
    return rows, confirmation


def ordered_jobs(tasks):
    blocks = []
    for task in tasks:
        if not blocks or blocks[-1][0]['session'] != task['session']:
            blocks.append([])
        blocks[-1].append(task)
    jobs = []
    for i, block in enumerate(blocks):
        for arm in (('baseline','candidate') if i%2 == 0 else ('candidate','baseline')):
            for task in block:
                key = dict(suite='evidence-coverage-'+arm, condition='hybrid-agent', domain=task['domain'],
                           task_id=task['task_id'], language=task['language'], repeat=0)
                jobs.append({'key':key, 'question':task['question'], 'task':task, 'arm':arm,
                             'session':task['session'], 'unit': (0 if task['domain']=='demo42' else 2) + int(arm=='candidate')})
    return jobs


def prepare(root):
    root = Path(root)
    if root.exists(): raise ValueError('实验目录已存在，不覆盖')
    tasks, confirm = task_plan()
    config = read_json(A5/'agent.json')
    configs = {arm: dict(config, evidence_coverage_search=arm=='candidate') for arm in ('baseline','candidate')}
    root.mkdir(parents=True)
    atomic_json(root/'tasks.json', tasks)
    atomic_json(root/'confirmation-tasks.json', confirm)
    for arm, value in configs.items(): atomic_json(root/f'{arm}-agent.json', value)
    cache = read_json(A5/'translations/translations.json')
    atomic_json(root/'translations.json', {t['source_task_id']:cache[t['source_task_id']] for t in tasks
        if t['language']=='zh' and t['session'] not in ('D1','D2')})
    atomic_json(root/'budget-before.json', budget(config))
    # Inputs and reference pages are recorded once before any system answer.
    atomic_json(root/'selection.json', {'status':'prepared_not_run','development_turns':12,
        'development_first_attempts':24,'confirmation_turns':6,'human':'not_checked',
        'scope':'Previously seen development sources, not independent acceptance.',
        'reference_check':'42-page population physical 9–10, employment 3 and 5–7; employment 23 is references; JRC original frozen references.'})
    return {'root':str(root), 'planned':len(ordered_jobs(tasks)), 'budget':budget(config)}


def load_plan(root):
    from foliorecall.agent import load_public_corpus
    from foliorecall.generation import read_agent_config
    from foliorecall.query import validate_query_config
    from foliorecall.search import load_index
    from foliorecall.retrieval import read_retrieval_config, load_retrieval
    from foliorecall.qa_translation import validate_translation
    root = Path(root)
    teacher, student = read_json('configs/baseline.json'), read_json('configs/student-ml-cpu.json')
    validate_query_config(student, teacher)
    units = []
    for domain, folder, public, hybrid in [
        ('demo42', DEMO, DEMO/'public-corpus.json', DEMO/'hybrid.json'),
        ('jrc', Path('outputs/extensions/A/indexes/jrc-original'), JRC/'public-corpus.json', JRC/'hybrid.json')]:
        _, pages = load_index(folder, teacher)
        texts = load_public_corpus(public, pages)
        retrieval = read_retrieval_config(hybrid)
        load_retrieval(retrieval,pages)
        for arm in ('baseline','candidate'):
            spec = {'index':str(folder.resolve()),'mode':'agent'}
            units.append({'spec':spec,'teacher':teacher,'student':student,'retrieval_config':retrieval,
                'texts':texts,'agent':read_agent_config(root/f'{arm}-agent.json')})
    tasks = read_json(root/'tasks.json')
    jobs = ordered_jobs(tasks)
    cache = read_json(root/'translations.json')
    for job in jobs:
        task = job['task']
        if task['language']=='zh' and task['session'] not in ('D1','D2'):
            job['translation'] = cache[task['source_task_id']]
            if not validate_translation(job['translation'], job['question'], units[job['unit']]['agent']):
                raise ValueError('中文首问缺少完整原缓存预算')
    identity = {'suite':{'suite_id':'evidence-coverage-v1','inter_task_cooldown_seconds':65,
                        'job_order':[j['key'] for j in jobs]},
                'tasks':tasks,'units':units,'translations':cache,'code':execution_code(),
                'experiment_code':digest(Path(__file__).read_text(encoding='utf-8'))}
    return units, identity, jobs


def conversation_context(root, jobs, job):
    history, sources = [], []
    for previous in jobs:
        if previous['key'] == job['key']: break
        if previous['arm'] != job['arm'] or previous['session'] != job['session']: continue
        folder = Path(root)/'runs/attempts'/digest(previous['key'])
        saved = [read_json(p) for p in sorted(folder.glob('attempt-*.json'))]
        successful = [a for a in saved if a['state']=='finished' and a['result']['completed']]
        if successful:
            a = successful[0]
            history.append(a['result'])
            sources.append({'key':a['key'],'attempt':a['attempt'],'answer_digest':a['answer_digest']})
    return history[-3:], sources[-3:]


def run(root, resume=False, retry_one=False, stop_after=None, dry_run=False):
    root = Path(root)
    units, identity, jobs = load_plan(root)
    saved = attempts(root)
    recoverable = [a for a in saved if a['attempt']==1 and a['state']=='finished'
        and transient(a['result']['termination_reason'])
        and not any(b['key']==a['key'] and b['attempt']==2 for b in saved)
        and can_recover(a['result'],next(j for j in jobs if j['key']==a['key']),
                        units[next(j['unit'] for j in jobs if j['key']==a['key'])]['agent'])]
    if retry_one and not recoverable: raise ValueError('没有可合法独立补跑的任务')
    if recoverable and not retry_one and not dry_run: raise ValueError('请先单独使用 --resume --retry-one')
    def factory():
        execute = provider_factory(units)
        def call(job, resources, attempt):
            history, sources = conversation_context(root,jobs,job)
            events = []
            result = execute(dict(job,conversation=history,on_event=events.append), resources, attempt)
            result.update(context_sources=sources, context_missing=job['session'] in ('D1','D2') and job['task']['language']=='zh' and not history,
                          conversation_id=job['arm']+'-'+job['session'], turn_index=len(history)+1, progress=events)
            print(json.dumps({'task':job['key'],'attempt':attempt,'completed':result['completed'],
                'reason':result['termination_reason'],'cost_cny':result['estimated_cost_cny']},ensure_ascii=False),flush=True)
            return result
        return call
    return execute_plan(root/'runs',identity,jobs,factory,[u['agent'] for u in units],resume,
                        1 if retry_one else stop_after,14400,dry_run)


def report(root):
    from scripts.analyze_qa_development import paired_candidates
    root = Path(root)
    rows = attempts(root)
    reviews = [read_json(p) for p in sorted((root/'scoring').glob('*.json'))]
    manifest = read_json(root/'runs/manifest.json')
    configs = [read_json(root/f'{arm}-agent.json') for arm in ('baseline','candidate')]
    ledger = ledger_snapshot(configs,manifest['execution_id'])
    summary = summarize_attempts(rows,24,reviews,manifest['job_keys'],ledger['calls'])
    groups = {}
    for arm in ('baseline','candidate'):
        chosen = [a for a in rows if a['key']['suite']=='evidence-coverage-'+arm]
        ratings = [r for r in reviews if r['key']['suite']=='evidence-coverage-'+arm]
        first = [a for a in chosen if a['attempt']==1 and a['state']=='finished']
        first_ratings = [r for r in ratings if r['attempt']==1]
        groups[arm] = {'first_terminal':len(first),'valid':sum(a['result']['completed'] for a in first),
            'reviewed':len(ratings),'metrics':{m:{'denominator':sum(r[m] is not None for r in first_ratings),
                'mean':sum(r[m] for r in first_ratings if r[m] is not None)/sum(r[m] is not None for r in first_ratings)
                    if any(r[m] is not None for r in first_ratings) else None} for m in RATINGS},
            'actual_searches':sum(a['result']['actual_searches'] for a in chosen if a['state']=='finished'),
            'actual_requests':sum(a['result']['actual_model_requests'] for a in chosen if a['state']=='finished'),
            'known_cost_cny':sum(a['result']['estimated_cost_cny'] for a in chosen if a['state']=='finished'),
            'source_access':[{ 'key':a['key'], 'retrieved_doc_ids':sorted({p['page_id'].rsplit(':',1)[0] for t in a['result']['trace']
                if t.get('tool')=='search_pages' and t.get('status')=='ok' for p in t['result']}),
                'read_doc_ids':sorted({p['page_id'].rsplit(':',1)[0] for p in a['result']['evidence']})} for a in first]}
    select = lambda values,arm:[a for a in values if a['key']['suite']=='evidence-coverage-'+arm]
    pairs = paired_candidates(select(rows,'baseline'),select(rows,'candidate'),select(reviews,'baseline'),select(reviews,'candidate'))
    return {'status':'complete_assistant_reviewed' if len(rows)==len(reviews) and summary['primary']['finished']==24
            and not summary['primary']['incomplete_budget'] and not summary['recovery']['incomplete_budget'] else 'partial',
        'summary':summary,'by_arm':groups,'pairs':pairs,'ledger':ledger,'budget':budget(configs[0]),
        'human':'not_checked','scope':'Development sources already seen; conversation turns and conditions are paired, not independent samples.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','review','report'))
    p.add_argument('--root',default=str(ROOT)); p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-one',action='store_true'); p.add_argument('--stop-after',type=int)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--reviews'); p.add_argument('--output')
    a = p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      HF_HOME='/mnt/d/foliorecall_cache/stage4_cpu/huggingface',
                      OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if a.command=='prepare': value=prepare(a.root)
    elif a.command=='run': value=run(a.root,a.resume,a.retry_one,a.stop_after,a.dry_run)
    elif a.command=='review':
        from foliorecall.qa_reviews import save_judgments
        value=save_judgments(attempts(a.root),read_json(a.reviews),Path(a.root)/'scoring')
    else:
        value=report(a.root)
        if a.output:
            if Path(a.output).exists(): raise ValueError('报告已存在，不覆盖')
            atomic_json(a.output,value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
