"""Frozen three-arm document Agent comparison, using the existing QA runtime."""
import argparse
import copy
import json
import os
from pathlib import Path

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_runtime import (execute_plan, provider_factory, execution_code,
                                   can_recover, transient, ledger_snapshot)
from foliorecall.qa_evaluation import RATINGS, summarize_attempts
from scripts.run_evidence_coverage import budget

ROOT = Path('outputs/extensions/A/multi-agent-v1/live-v1')
SOURCE = Path('outputs/extensions/A/evidence-coverage-v1')
A5 = Path('outputs/extensions/A/qa-acceptance-v2')
DEMO = Path('outputs/releases/v0.1.0-final-candidate/demo-bundle')
JRC = Path('outputs/extensions/A/data/jrc-corpus-v1')
QUERY = Path('outputs/extensions/D/onnx-int8-v1/configs/onnx-fp32.json')
ARMS = ('A', 'B', 'C', 'serial')


def expand(tasks, confirmation, diagnostic=False):
    if diagnostic:
        selected = [next(t for t in tasks if t['task_id'] == f'cov-{n:02d}') for n in (1, 3, 5, 9)]
        blocks = [([t], ('serial','C') if i % 2 == 0 else ('C','serial')) for i,t in enumerate(selected)]
    else:
        blocks = []
        for task in tasks + confirmation:
            if not blocks or blocks[-1][0][-1]['session'] != task['session']:
                i = len(blocks)
                blocks.append(([], ('A','B','C')[i % 3:] + ('A','B','C')[:i % 3]))
            blocks[-1][0].append(task)
    jobs = []
    for block, arms in blocks:
        for arm in arms:
            for task in block:
                stage = 'diagnostic' if diagnostic else 'confirmation' if task['task_id'].startswith('confirm-') else 'development'
                key = dict(suite=f'multi-agent-v1-{stage}-{arm}', condition='hybrid-agent',
                           domain=task['domain'], task_id=task['task_id'], language=task['language'], repeat=0)
                mode = 'multi-agent' if arm in ('C','serial') else 'agent'
                jobs.append({'key':key,'task':task,'question':task['question'],'arm':arm,'stage':stage,
                             'session':task['session'], 'mode':mode,
                             'unit':(0 if task['domain']=='demo42' else 4) + ARMS.index(arm)})
    return jobs


def prepare(root):
    root = Path(root)
    if root.exists():
        raise ValueError('实验目录已存在，不覆盖')
    tasks, confirmation = read_json(SOURCE/'tasks.json'), read_json(SOURCE/'confirmation-tasks.json')
    base = read_json(A5/'agent.json')
    configs = {}
    for arm in ARMS:
        value = dict(base, evidence_coverage_search=False, final_answer_evidence_check=False)
        if arm != 'A':
            value.update(max_requests=12,max_searches=4,max_pages=8,max_output_tokens=12000,task_timeout=240)
        value['max_parallel_workers'] = 1 if arm == 'serial' else 2
        configs[arm] = value
    cache = read_json(A5/'translations/translations.json')
    translations = {t['source_task_id']:cache[t['source_task_id']] for t in tasks+confirmation
                    if t['language']=='zh' and t['session'] not in ('D1','D2','C1')}
    root.mkdir(parents=True)
    for name,value in [('tasks.json',tasks),('confirmation-tasks.json',confirmation),
                       ('translations.json',translations),('query.json',read_json(QUERY)),
                       ('budget-before.json',budget(base))]:
        atomic_json(root/name,value)
    for arm,value in configs.items():
        atomic_json(root/f'agent-{arm}.json',value)
    plan = {'main':expand(tasks,confirmation),'diagnostic':expand(tasks,confirmation,True)}
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'selection.json',{'main':54,'diagnostic':8,'main_unique_turns':18,
        'main_sessions':15,'expected_cached_chinese_first_attempts':18,
        'expected_chinese_followups':9,'translation_sources':6,'human':'not_checked',
        'scope':'Previously seen development tasks; paired arms and conversation turns are not independent samples.'})
    return {'planned':{k:len(v) for k,v in plan.items()},'budget':budget(base)}


def load_plan(root, diagnostic=False):
    from foliorecall.agent import load_public_corpus
    from foliorecall.generation import read_agent_config
    from foliorecall.query import validate_query_config
    from foliorecall.search import load_index
    from foliorecall.retrieval import read_retrieval_config, load_retrieval
    from foliorecall.qa_translation import validate_translation
    root = Path(root)
    teacher, student = read_json('configs/baseline.json'), read_json(root/'query.json')
    validate_query_config(student,teacher)
    units = []
    for folder,public,hybrid,count in [(DEMO,DEMO/'public-corpus.json',DEMO/'hybrid.json',42),
        (Path('outputs/extensions/A/indexes/jrc-original'),JRC/'public-corpus.json',JRC/'hybrid.json',256)]:
        _,pages = load_index(folder,teacher)
        if len(pages) != count:
            raise ValueError('实验必须使用完整42页或256页语料')
        texts = load_public_corpus(public,pages)
        retrieval = read_retrieval_config(hybrid)
        load_retrieval(retrieval,pages)
        for arm in ARMS:
            units.append({'spec':{'index':str(folder.resolve()),'mode':'multi-agent' if arm in ('C','serial') else 'agent'},
                          'teacher':teacher,'student':student,'retrieval_config':retrieval,'texts':texts,
                          'agent':read_agent_config(root/f'agent-{arm}.json')})
    tasks,confirmation = read_json(root/'tasks.json'),read_json(root/'confirmation-tasks.json')
    jobs = expand(tasks,confirmation,diagnostic)
    stored = read_json(root/'plan.json')['diagnostic' if diagnostic else 'main']
    if jobs != stored:
        raise ValueError('任务顺序与准备记录不一致')
    cache = read_json(root/'translations.json')
    for job in jobs:
        t = job['task']
        if t['language']=='zh' and t['session'] not in ('D1','D2','C1'):
            job['translation'] = cache[t['source_task_id']]
            if not validate_translation(job['translation'],job['question'],units[job['unit']]['agent']):
                raise ValueError('中文首问缺少有效冻结缓存及原预算')
    identity = {'suite':{'suite_id':'multi-agent-v1-diagnostic' if diagnostic else 'multi-agent-v1-main',
                        'inter_task_cooldown_seconds':65,'job_order':[j['key'] for j in jobs]},
                'tasks':tasks,'confirmation':confirmation,'translations':cache,'units':units,
                'code':execution_code(),'experiment_code':digest(Path(__file__).read_text(encoding='utf-8'))}
    return units,identity,jobs


def attempts(root, diagnostic=None):
    root = Path(root)
    names = ('diagnostic',) if diagnostic is True else ('main',) if diagnostic is False else ('main','diagnostic')
    return [read_json(p) for name in names for p in sorted((root/name/'attempts').glob('*/attempt-*.json'))]


def pending(jobs, rows, units):
    result = []
    for job in jobs:
        saved = [a for a in rows if a['key']==job['key']]
        if not saved or saved[-1]['state']=='running':
            result.append(job)
        elif len(saved)==1 and transient(saved[0]['result']['termination_reason']) and can_recover(
                saved[0]['result'],job,units[job['unit']]['agent']):
            result.append(job)
    return result


def history_for(rows, jobs, job):
    history,sources = [],[]
    for previous in jobs:
        if previous['key']==job['key']:
            break
        if previous['arm']!=job['arm'] or previous['session']!=job['session']:
            continue
        valid = [a for a in rows if a['key']==previous['key'] and a['state']=='finished' and a['result']['completed']]
        if valid:
            a = valid[-1]
            history.append(a['result'])
            sources.append({'key':a['key'],'attempt':a['attempt'],'answer_digest':a['answer_digest']})
    return history[-3:],sources[-3:]


def check_identity(root,identity,diagnostic):
    manifest = Path(root)/('diagnostic' if diagnostic else 'main')/'manifest.json'
    if manifest.exists() and read_json(manifest)['identity_digest'] != digest(identity):
        raise ValueError('实验身份变化，不能混合恢复')


def require_reviews(root, rows):
    reviewed = {(digest(r['key']),r['attempt']) for r in
                [read_json(p) for p in (Path(root)/'scoring').glob('*.json')]}
    missing = [a for a in rows if a['state']=='finished' and (digest(a['key']),a['attempt']) not in reviewed]
    if missing:
        raise ValueError(f'段后评审尚缺{len(missing)}份')


def run(root, phase='development', resume=False, retry_one=False, stop_after=None, dry_run=False):
    root = Path(root)
    diagnostic = phase=='diagnostic'
    units,identity,jobs = load_plan(root,diagnostic)
    check_identity(root,identity,diagnostic)
    rows = attempts(root,diagnostic)
    queue = pending(jobs,rows,units)
    recovering = [j for j in queue if any(a['key']==j['key'] and a['state']=='finished' for a in rows)]
    if retry_one and not recovering:
        raise ValueError('没有原规则允许的独立补跑')
    if recovering and not retry_one and not dry_run:
        raise ValueError('先使用 --resume --retry-one 完成合法补跑')
    if retry_one:
        if queue[0]['key']!=recovering[0]['key']:
            raise ValueError('补跑顺序不一致')
        limit = 1
    else:
        limit = 0
        for job in queue:
            if job['stage']!=phase or (phase=='confirmation' and job['arm']=='C'):
                break
            limit += 1
        if phase=='development':
            first_done = sum(a['attempt']==1 and a['state']=='finished' for a in rows)
            if first_done < 12:
                limit = min(limit,12-first_done)
            elif not dry_run:
                require_reviews(root,[a for a in rows if a['key']['task_id'] in ('cov-01','cov-02','cov-03','cov-04')])
    if phase=='confirmation' and not dry_run:
        require_reviews(root,[a for a in rows if a['key']['task_id'].startswith('cov-')])
    if phase=='diagnostic' and not dry_run:
        require_reviews(root,attempts(root,False))
        if any(a['state']=='running' for a in attempts(root,False)) or len({digest(a['key']) for a in attempts(root,False)})!=54:
            raise ValueError('先完成主对照及评审')
    if stop_after is not None:
        limit = min(limit,stop_after)
    status = {'phase':phase,'next':queue[0]['key'] if queue else None,'automatic_tasks':limit,
              'browser_required':bool(queue and queue[0]['stage']=='confirmation' and queue[0]['arm']=='C'),
              'budget':budget(units[0]['agent'])}
    if dry_run or not limit:
        return status
    def factory():
        execute = provider_factory(units)
        def call(job,resources,attempt):
            history,sources = history_for(attempts(root,diagnostic),jobs,job)
            events = []
            result = execute(dict(job,conversation=history,on_event=events.append),resources,attempt)
            result.update(context_sources=sources,context_missing=job['task']['language']=='zh' and
                          job['session'] in ('D1','D2','C1') and not history,
                          conversation_id=job['arm']+'-'+job['session'],turn_index=len(history)+1,progress=events,
                          experiment_arm=job['arm'],experiment_stage=job['stage'])
            print(json.dumps({'task':job['key'],'attempt':attempt,'completed':result['completed'],
                              'reason':result['termination_reason'],'cost_cny':result['estimated_cost_cny']},ensure_ascii=False),flush=True)
            return result
        return call
    return execute_plan(root/('diagnostic' if diagnostic else 'main'),identity,jobs,factory,
                        [u['agent'] for u in units],resume,limit,14400)


def context_view(values):
    return [{k:v[k] for k in ('question','retrieval_query','answer','completed','termination_reason')} for v in values]


class BrowserExecutor:
    """The browser executes the next frozen C job; registration never generates answers."""
    def __init__(self,root,units,identity,jobs):
        self.root,self.units,self.identity,self.jobs = Path(root),units,identity,jobs

    def __call__(self,question,mode,context,event,state,resources):
        from foliorecall.multi_agent import ask_multi
        if mode!='multi-agent':
            raise ValueError('本确认入口仅接受固定多Agent任务')
        check_identity(self.root,self.identity,False)
        binding_path = self.root/'browser-bindings.json'
        bindings = read_json(binding_path) if binding_path.exists() else {}
        binding = state['conversation_id']+':'+str(len(state['turns'])+1)
        rows = attempts(self.root,False)
        if binding in bindings:
            prior = [a for a in rows if a['key']==bindings[binding] and a['state']=='finished']
            if prior:
                if question != prior[-1]['result']['question']:
                    raise ValueError('该浏览器轮次已经绑定其他问题，请新建会话')
                return prior[-1]['result']
        queue = pending(self.jobs,rows,self.units)
        if not queue or queue[0]['stage']!='confirmation' or queue[0]['arm']!='C':
            raise ValueError('下一项不是浏览器C组任务，请按冻结顺序运行命令行部分')
        job = queue[0]
        if question != job['question']:
            raise ValueError('请提交固定问题原文：'+job['question'])
        history,sources = history_for(rows,self.jobs,job)
        if context_view(context)!=context_view(history):
            raise ValueError('浏览器会话与本条件历史不一致；独立题请新建会话')
        require_reviews(self.root,[a for a in rows if a['key']['task_id'].startswith('cov-')])
        bindings[binding] = job['key']
        atomic_json(binding_path,bindings)
        captured = []
        def execute(current,budget_state,attempt):
            if current['key']!=job['key']:
                raise ValueError('浏览器任务顺序变化')
            provider = resources['provider']
            provider.context = dict(current['key'],attempt=attempt,execution_id=current['execution_id'])
            result = ask_multi(question,resources['model'],self.units[job['unit']]['student'],
                resources['index'],resources['pages'],self.units[job['unit']]['agent'],provider,current['graph_output'],
                resources['retrieval'],resources['texts'],translation=job.get('translation'),conversation=history,
                on_event=event,resume=(Path(current['graph_output'])/'manifest.json').exists(),
                query_lock=resources['query_lock'],cooldown_seconds=None,
                record_metadata={'conversation_id':state['conversation_id'],'turn_index':len(state['turns'])+1,
                                 'experiment_key':job['key'],'execution_id':current['execution_id']})
            result.update(context_sources=sources,context_missing=job['session']=='C1' and job['task']['language']=='zh' and not history,
                          experiment_arm='C',experiment_stage='confirmation',browser_confirmed=True,loading_seconds=0.)
            captured.append(result)
            return result
        execute_plan(self.root/'main',self.identity,self.jobs,lambda:execute,[u['agent'] for u in self.units],
                     resume=True,stop_after=1,max_seconds=14400)
        if not captured:
            raise ValueError('任务未发送；查看实验summary中的预算、服务等待或恢复原因')
        return captured[0]


def serve(root,port):
    from foliorecall.demo import serve_demo
    root = Path(root)
    units,identity,jobs = load_plan(root)
    check_identity(root,identity,False)
    queue = pending(jobs,attempts(root,False),units)
    if not queue or queue[0]['stage']!='confirmation' or queue[0]['arm']!='C':
        raise ValueError('先按顺序完成下一段命令行任务')
    job = queue[0]
    print(json.dumps({'browser_question':job['question'],'session':job['session'],'url':f'http://127.0.0.1:{port}'},ensure_ascii=False),flush=True)
    unit = units[job['unit']]
    output = root/'browser'/f'{job["task"]["task_id"]}-{__import__("time").time_ns()}'
    return serve_demo(unit['spec']['index'],unit['teacher'],unit['student'],port=port,output=output,
        retrieval_config=unit['retrieval_config'],agent_config_path=root/'agent-A.json',
        public_corpus=JRC/'public-corpus.json',multi_agent_config_path=root/'agent-C.json',
        _chat_executor=BrowserExecutor(root,units,identity,jobs))


def overlap(root,job):
    folder = Path(root)/'diagnostic/attempts'/digest(job['key'])/'graph/operations'
    intervals = []
    for p in folder.glob('*.json'):
        op = read_json(p)
        record = op.get('outcome',{}).get('record',{})
        if op['kind']=='request' and record.get('cost_status')!='not_sent' and record.get('seconds') is not None and 'finished_at' in op:
            intervals.append({'role':op['role'],'start':op['finished_at']-record['seconds'],'end':op['finished_at']})
    edges = sorted([(r['start'],1) for r in intervals]+[(r['end'],-1) for r in intervals])
    active,peak,previous,overlap_seconds = 0,0,None,0.
    for stamp,delta in edges:
        if previous is not None and active>=2:
            overlap_seconds += stamp-previous
        active += delta
        peak = max(peak,active)
        previous = stamp
    return {'key':job['key'],'client_intervals':intervals,'max_client_calls':peak,'overlap_seconds':overlap_seconds,
            'scope':'Client intervals reconstructed from recorded durations and completion times; includes response handling, not server GPU timing.'}


def report(root, final=False):
    from scripts.analyze_qa_development import paired_candidates
    root = Path(root)
    plan = read_json(root/'plan.json')
    rows = attempts(root)
    reviews = [read_json(p) for p in sorted((root/'scoring').glob('*.json'))]
    configs = [read_json(root/f'agent-{a}.json') for a in ARMS]
    ledgers = {}
    identities = []
    for name in ('main','diagnostic'):
        manifest = root/name/'manifest.json'
        if manifest.exists():
            m = read_json(manifest)
            identities.append({'stage':name,'execution_id':m['execution_id']})
            for c in ledger_snapshot(configs,m['execution_id'])['calls']:
                ledgers[c['id']] = c
    all_jobs = plan['main']+plan['diagnostic']
    summary = summarize_attempts(rows,62,reviews,[j['key'] for j in all_jobs],list(ledgers.values()))
    groups = {}
    for stage,arms in [('main',('A','B','C')),('diagnostic',('serial','C'))]:
        for arm in arms:
            keys = {digest(j['key']) for j in plan[stage] if j['arm']==arm}
            chosen = [a for a in rows if digest(a['key']) in keys]
            first = [a for a in chosen if a['state']=='finished' and a['attempt']==1]
            ratings = [r for r in reviews if digest(r['key']) in keys and r['attempt']==1]
            groups[stage+'-'+arm] = {'first_terminal':len(first),'structure_valid':sum(a['result']['completed'] for a in first),
                'reviewed_first':len(ratings),'attempts':len(chosen),
                'metrics':{m:{'denominator':sum(r[m] is not None for r in ratings),
                     'mean':sum(r[m] for r in ratings if r[m] is not None)/sum(r[m] is not None for r in ratings)
                     if any(r[m] is not None for r in ratings) else None} for m in RATINGS},
                'actual_requests':sum(a['result']['actual_model_requests'] for a in chosen if a['state']=='finished'),
                'actual_searches':sum(a['result']['actual_searches'] for a in chosen if a['state']=='finished'),
                'active_seconds':sum(a['result']['timing']['task_seconds'] for a in chosen if a['state']=='finished')}
    select = lambda values,arm:[a for a in values if a['key']['suite'] in (f'multi-agent-v1-development-{arm}',f'multi-agent-v1-confirmation-{arm}') and a['attempt']==1]
    pairs = {f'{c}-vs-{b}':paired_candidates(select(rows,b),select(rows,c),select(reviews,b),select(reviews,c))
             for b,c in [('B','C'),('A','B'),('A','C')]}
    unfinished = [a for a in rows if a['state']!='finished']
    complete = summary['primary']['finished']==62 and not unfinished and len(rows)==len(reviews)
    for diagnostic in (False,True):
        units,_,jobs = load_plan(root,diagnostic)
        if pending(jobs,attempts(root,diagnostic),units):
            complete = False
    browser_first = [a for a in rows if a['attempt']==1 and a['key']['suite']=='multi-agent-v1-confirmation-C']
    complete = complete and len(browser_first)==6 and all(a['result'].get('browser_confirmed') for a in browser_first)
    if final and not complete:
        raise ValueError('尚未完成62项、全部评审、合法补跑或六次浏览器确认')
    return {'status':'complete_assistant_reviewed' if complete else 'partial','summary':summary,'groups':groups,
        'pairs':pairs,'execution_ids':identities,'ledger':{'calls':list(ledgers.values()),
            'known_cost_cny':sum(c['cny'] for c in ledgers.values() if c['status']=='estimated_from_usage'),
            'unknown_reserved_cny':sum(c['cny'] for c in ledgers.values() if c['status']=='reserved_unknown_charge')},
        'historical_translation_cost_cny':sum(v['cost_cny'] for v in read_json(root/'translations.json').values()),
        'budget':budget(configs[0]),'diagnostic_overlap':[overlap(root,j) for j in plan['diagnostic']],
        'human':'not_checked','scope':read_json(root/'selection.json')['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','serve','review','report','status'))
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--phase',choices=('development','confirmation','diagnostic'),default='development')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-one',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--port',type=int,default=7872)
    parser.add_argument('--reviews',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--final',action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',HF_HOME='/mnt/d/foliorecall_cache/stage4_cpu/huggingface',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if args.command=='prepare':
        value = prepare(args.root)
    elif args.command in ('run','status'):
        value = run(args.root,args.phase,args.resume,args.retry_one,args.stop_after,args.dry_run or args.command=='status')
    elif args.command=='serve':
        return serve(args.root,args.port)
    elif args.command=='review':
        from foliorecall.qa_reviews import save_judgments
        value = save_judgments(attempts(args.root),read_json(args.reviews),args.root/'scoring')
    else:
        value = report(args.root,args.final)
    if args.output:
        if args.output.exists():
            raise ValueError('输出文件已存在，不覆盖')
        atomic_json(args.output,value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
