"""Research-v1 experiment adapter; serial runtime, reviews and browser stay shared."""
import json
from pathlib import Path
import statistics
from collections import Counter

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_runtime import execute_plan, provider_factory, execution_code, ledger_snapshot
from foliorecall.qa_evaluation import RATINGS, summarize_attempts
from scripts import run_multi_agent_experiment as legacy
from scripts.run_evidence_coverage import budget

ARMS = ('S','M1','M2','N')
SPECIAL = {'cov-03':('M1','N','M2','S'),'cov-05':('M2','N','S','M1'),
           'cov-08':('N','M2','S','M1'),'confirm-01':('M1','M2','N','S')}


def expand(tasks, confirmation):
    blocks = []
    for task in tasks+confirmation:
        if not blocks or blocks[-1][-1]['session'] != task['session']:
            blocks.append([])
        blocks[-1].append(task)
    jobs = []
    for i,block in enumerate(blocks):
        order = SPECIAL.get(block[0]['task_id'],('S','M1','M2')[i%3:]+('S','M1','M2')[:i%3])
        for arm in order:
            for task in block:
                stage = 'confirmation' if task['task_id'].startswith('confirm-') else 'development'
                key = dict(suite=f'multi-agent-research-v1-{stage}-{arm}',condition='hybrid-agent',
                           domain=task['domain'],task_id=task['task_id'],language=task['language'],repeat=0)
                jobs.append({'key':key,'task':task,'question':task['question'],'stage':stage,
                    'stratum':'ablation' if arm=='N' else 'main','arm':arm,'session':task['session'],
                    'mode':'agent' if arm=='S' else 'multi-agent',
                    'unit':(0 if task['domain']=='demo42' else 4)+ARMS.index(arm)})
    return jobs


def prepare(root):
    root = Path(root)
    if root.exists():
        raise ValueError('实验目录已存在，不覆盖')
    tasks,confirm = read_json(legacy.SOURCE/'tasks.json'),read_json(legacy.SOURCE/'confirmation-tasks.json')
    base = read_json(legacy.A5/'agent.json')
    configs = {}
    for arm in ARMS:
        configs[arm] = dict(base,max_requests=16,max_searches=6,max_pages=10,max_output_tokens=16000,
            task_timeout=300,max_parallel_workers=2,evidence_coverage_search=False,final_answer_evidence_check=False,
            collaboration_workflow='research' if arm in ('M2','N') else 'basic',verification_rework=arm!='N')
    cache = read_json(legacy.A5/'translations/translations.json')
    translations = {t['source_task_id']:cache[t['source_task_id']] for t in tasks+confirm
                    if t['language']=='zh' and t['session'] not in ('D1','D2','C1')}
    jobs = expand(tasks,confirm)
    root.mkdir(parents=True)
    for name,value in [('tasks.json',tasks),('confirmation-tasks.json',confirm),('translations.json',translations),
                       ('query.json',read_json(legacy.QUERY)),('plan.json',{'main':jobs}),
                       ('budget-before.json',budget(base)),
                       ('selection.json',{'experiment':'research-v1','main':54,'ablation':6,'planned':60,
                        'main_turns':18,'main_sessions':15,'expected_cached_chinese':19,'expected_followups':11,
                        'scope':'Previously seen development tasks, paired arms and conversation turns are not independent samples.',
                        'human':'not_checked'})]:
        atomic_json(root/name,value)
    for arm,value in configs.items():
        atomic_json(root/f'agent-{arm}.json',value)
    return {'planned':len(jobs),'main':54,'ablation':6,'budget':budget(base)}


def load_plan(root):
    from foliorecall.agent import load_public_corpus
    from foliorecall.generation import read_agent_config
    from foliorecall.query import validate_query_config
    from foliorecall.search import load_index
    from foliorecall.retrieval import read_retrieval_config, load_retrieval
    from foliorecall.qa_translation import validate_translation
    root = Path(root)
    teacher,student = read_json('configs/baseline.json'),read_json(root/'query.json')
    validate_query_config(student,teacher)
    units = []
    for folder,public,hybrid,count in [(legacy.DEMO,legacy.DEMO/'public-corpus.json',legacy.DEMO/'hybrid.json',42),
        (Path('outputs/extensions/A/indexes/jrc-original'),legacy.JRC/'public-corpus.json',legacy.JRC/'hybrid.json',256)]:
        _,pages = load_index(folder,teacher)
        if len(pages)!=count:
            raise ValueError('必须使用完整42页与256页库')
        texts = load_public_corpus(public,pages)
        retrieval = read_retrieval_config(hybrid)
        load_retrieval(retrieval,pages)
        for arm in ARMS:
            units.append({'spec':{'index':str(folder.resolve()),'mode':'agent' if arm=='S' else 'multi-agent'},
                'teacher':teacher,'student':student,'retrieval_config':retrieval,'texts':texts,
                'agent':read_agent_config(root/f'agent-{arm}.json')})
    tasks,confirm = read_json(root/'tasks.json'),read_json(root/'confirmation-tasks.json')
    jobs = expand(tasks,confirm)
    if jobs != read_json(root/'plan.json')['main']:
        raise ValueError('任务清单或顺序变化')
    cache = read_json(root/'translations.json')
    for job in jobs:
        t = job['task']
        if t['language']=='zh' and t['session'] not in ('D1','D2','C1'):
            job['translation'] = cache[t['source_task_id']]
            if not validate_translation(job['translation'],job['question'],units[job['unit']]['agent']):
                raise ValueError('中文首问缓存或原预算无效')
    identity = {'suite':{'suite_id':'multi-agent-research-v1','inter_task_cooldown_seconds':65,
                         'job_order':[j['key'] for j in jobs]},
                'tasks':tasks,'confirmation':confirm,'translations':cache,'units':units,'code':execution_code(),
                'experiment_code':{p.name:digest(p.read_text(encoding='utf-8')) for p in
                                   (Path(__file__),Path(legacy.__file__))}}
    return units,identity,jobs


def run(root,phase='development',resume=False,retry_one=False,stop_after=None,dry_run=False):
    root = Path(root)
    if phase not in ('development','confirmation'):
        raise ValueError('research-v1不新增并发诊断')
    units,identity,jobs = load_plan(root)
    legacy.check_identity(root,identity,False)
    rows = legacy.attempts(root,False)
    queue = legacy.pending(jobs,rows,units)
    recovering = [j for j in queue if any(a['key']==j['key'] and a['state']=='finished' for a in rows)]
    if retry_one and (not recovering or queue[0]['key']!=recovering[0]['key']):
        raise ValueError('没有下一项合法补跑')
    if recovering and not retry_one and not dry_run:
        raise ValueError('先用 --resume --retry-one 保留首次结果独立补跑')
    limit = 0
    if retry_one:
        limit = 1
    else:
        for job in queue:
            if job['stage']!=phase or (phase=='confirmation' and job['arm']=='M2'):
                break
            limit += 1
        first_done = sum(a['attempt']==1 and a['state']=='finished' for a in rows)
        if phase=='development':
            if first_done < 14:
                limit = min(limit,14-first_done)
            elif not dry_run:
                legacy.require_reviews(root,[a for a in rows if a['key']['task_id'] in ('cov-01','cov-02','cov-03','cov-04')])
        elif not dry_run:
            legacy.require_reviews(root,[a for a in rows if a['key']['task_id'].startswith('cov-')])
    if stop_after is not None:
        limit = min(limit,stop_after)
    status = {'phase':phase,'next':queue[0]['key'] if queue else None,'automatic_tasks':limit,
        'browser_required':bool(queue and queue[0]['stage']=='confirmation' and queue[0]['arm']=='M2'),
        'budget':budget(units[0]['agent'])}
    if dry_run or not limit:
        return status
    def factory():
        execute = provider_factory(units)
        def call(job,resources,attempt):
            history,sources = legacy.history_for(legacy.attempts(root,False),jobs,job)
            events = []
            result = execute(dict(job,conversation=history,on_event=events.append),resources,attempt)
            result.update(context_sources=sources,context_missing=job['task']['language']=='zh' and
                          job['session'] in ('D1','D2','C1') and not history,
                conversation_id=job['arm']+'-'+job['session'],turn_index=len(history)+1,progress=events,
                experiment_arm=job['arm'],experiment_stage=job['stage'],experiment_stratum=job['stratum'])
            print(json.dumps({'task':job['key'],'attempt':attempt,'completed':result['completed'],
                'reason':result['termination_reason'],'cost_cny':result['estimated_cost_cny']},ensure_ascii=False),flush=True)
            return result
        return call
    return execute_plan(root/'main',identity,jobs,factory,[u['agent'] for u in units],resume,limit,14400)


def serve(root,port):
    from foliorecall.demo import serve_demo
    root = Path(root)
    units,identity,jobs = load_plan(root)
    legacy.check_identity(root,identity,False)
    queue = legacy.pending(jobs,legacy.attempts(root,False),units)
    if not queue or queue[0]['stage']!='confirmation' or queue[0]['arm']!='M2':
        raise ValueError('下一项不是浏览器研究模式确认')
    job = queue[0]
    print(json.dumps({'browser_question':job['question'],'session':job['session'],'port':port},ensure_ascii=False),flush=True)
    unit = units[job['unit']]
    output = root/'browser'/f'{job["task"]["task_id"]}-{__import__("time").time_ns()}'
    return serve_demo(unit['spec']['index'],unit['teacher'],unit['student'],port=port,output=output,
        retrieval_config=unit['retrieval_config'],agent_config_path=root/'agent-S.json',
        public_corpus=legacy.JRC/'public-corpus.json',multi_agent_config_path=root/'agent-M2.json',
        _chat_executor=legacy.BrowserExecutor(root,units,identity,jobs,browser_arm='M2'))


def report(root,final=False):
    from scripts.analyze_qa_development import paired_candidates
    root = Path(root)
    jobs = read_json(root/'plan.json')['main']
    by_key = {digest(j['key']):j for j in jobs}
    rows = legacy.attempts(root,False)
    reviews = [read_json(p) for p in sorted((root/'scoring').glob('*.json'))]
    configs = [read_json(root/f'agent-{a}.json') for a in ARMS]
    manifest = read_json(root/'main/manifest.json') if (root/'main/manifest.json').exists() else None
    ledgers = ledger_snapshot(configs,manifest['execution_id']) if manifest else {'calls':[]}
    summary = summarize_attempts(rows,60,reviews,[j['key'] for j in jobs],ledgers['calls'])
    # The legacy /3 field is an allocation diagnostic, not a new expense with a fourth arm.
    summary.pop('translation_allocated_cost_cny',None)
    summary['translation_cost_scope']='Historical source requests deduplicated by ledger_id, never charged again.'
    first = [a for a in rows if a['state']=='finished' and a['attempt']==1]
    rating_lookup = {(digest(r['key']),r['attempt']):r for r in reviews}
    def select(values,arm):
        return [a for a in values if a['attempt']==1 and by_key[digest(a['key'])]['arm']==arm]
    groups = []
    for arm in ARMS:
      for stage in ('all','development','confirmation'):
       for language,category in [('all','all'),('en','all'),('zh','all')]+[
               ('all',c) for c in sorted({j['task']['category'] for j in jobs})]:
        chosen = [a for a in select(first,arm) if (stage=='all' or by_key[digest(a['key'])]['stage']==stage)
                  and (language=='all' or a['key']['language']==language)
                  and (category=='all' or a['task']['category']==category)]
        if not chosen:
            continue
        rs = [rating_lookup[(digest(a['key']),1)] for a in chosen if (digest(a['key']),1) in rating_lookup]
        groups.append({'arm':arm,'stage':stage,'language':language,'category':category,'first_terminal':len(chosen),
            'completed':sum(a['result']['completed'] for a in chosen),'reviewed':len(rs),
            'model_answer_structure_valid':sum(a['result']['completed'] for a in chosen),
            'draft_structure_valid':sum(bool(a['result'].get('research',{}).get('drafts')) for a in chosen),
            'verification_complete':sum(a['result'].get('research',{}).get('verification_status') in ('passed','limited') for a in chosen),
            'limited_answers':sum(a['result']['completed'] and a['result']['answer']['insufficient_evidence'] for a in chosen),
            'termination':dict(Counter(a['result']['termination_reason'] for a in chosen)),
            'metrics':{m:{'denominator':sum(r[m] is not None for r in rs),
                'mean':statistics.mean([r[m] for r in rs if r[m] is not None]) if any(r[m] is not None for r in rs) else None} for m in RATINGS},
            'actual_requests':sum(a['result']['actual_model_requests'] for a in chosen),
            'actual_searches':sum(a['result']['actual_searches'] for a in chosen),
            'cost_cny':sum(a['result']['estimated_cost_cny'] for a in chosen),
            'active_seconds_median':statistics.median(a['result']['timing']['task_seconds'] for a in chosen)})
    pair_rows = {f'{c}-vs-{b}':paired_candidates(select(rows,b),select(rows,c),select(reviews,b),select(reviews,c))
                 for b,c in [('S','M1'),('M1','M2'),('S','M2'),('N','M2')]}
    # Compare the six ablation cells only, not unmatched full-workflow tasks.
    n_cells = {(j['key']['task_id'],j['key']['language']) for j in jobs if j['arm']=='N'}
    pair_rows['M2-vs-N'] = [v for v in pair_rows['M2-vs-N'] if
                            (v['cell'][2],v['cell'][3]) in n_cells]
    pairs = {name:{'rows':values,'metrics':{m:{'denominator':len(ds),
                    'mean_delta':statistics.mean(ds) if ds else None,'improved':sum(d>0 for d in ds),
                    'degraded':sum(d<0 for d in ds),'tied':sum(d==0 for d in ds)} for m in RATINGS
                for ds in [[v['score_delta'][m] for v in values if v['paired_reviewed'] and v['score_delta'][m] is not None]]}}
             for name,values in pair_rows.items()}
    research = [{'key':a['key'], 'research':a['result']['research']} for a in first if 'research' in a['result']]
    browser = [a for a in first if by_key[digest(a['key'])]['stage']=='confirmation' and by_key[digest(a['key'])]['arm']=='M2']
    complete = len(first)==60 and all(a['state']=='finished' for a in rows) and len(rows)==len(reviews)
    complete = complete and len(browser)==6 and all(a['result'].get('browser_confirmed') for a in browser)
    if complete:
        units,identity,actual_jobs = load_plan(root)
        legacy.check_identity(root,identity,False)
        complete = not legacy.pending(actual_jobs,rows,units)
    if final and not complete:
        raise ValueError('尚未完成60项、全部评审、合法补跑或六次浏览器确认')
    cache_calls = {c['ledger_id']:c for t in read_json(root/'translations.json').values() for c in t['api_calls']}
    return {'status':'complete_assistant_reviewed' if complete else 'partial','summary':summary,
        'groups':groups,'pairs':pairs,'research_records':research,'ledger':ledgers,
        'execution_id':manifest['execution_id'] if manifest else None,
        'historical_translation_cost_cny':sum(c.get('cost_cny',0) for c in cache_calls.values()),
        'historical_translation_ledger_ids':sorted(cache_calls),'budget':budget(configs[0]),
        'human':'not_checked','scope':read_json(root/'selection.json')['scope']}
