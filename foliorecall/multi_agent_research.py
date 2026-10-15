"""Role-specific document research, built on the existing collaboration runtime."""
import copy
import json
import math
import re
from typing import Annotated, TypedDict

from .agent import TOOLS, answer_contract, final_answer_contract, validate_answer
from .generation import TaskStopped
from .io import read_json
from .multi_agent import COLLABORATION_RULES, _merge, require_multiagent
from .qa_storage import atomic_json
from .qa_translation import convert_query, rewrite_followup_query, validate_translation


class ResearchState(TypedDict, total=False):
    query: str
    requirements: list
    tasks: list
    workers: Annotated[dict, _merge]
    phase: str
    version: int
    draft: dict
    verification: dict
    rework: dict
    answer: dict
    reason: str


def json_value(response):
    value = json.loads(response['content'])
    if not isinstance(value, dict):
        raise ValueError('角色输出须为JSON对象')
    return value


def text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 8000


def candidate_facts(worker, mapping):
    try:
        value = json.loads(worker.get('findings',''))
    except (TypeError, ValueError):
        return []
    facts = value.get('facts',[]) if isinstance(value,dict) else []
    if not isinstance(facts,list):
        return []
    records = []
    for i,fact in enumerate(facts):
        if not isinstance(fact,dict) or not text(fact.get('text')):
            continue
        ids = fact.get('evidence_page_ids',[])
        if not isinstance(ids,list) or not ids or any(p not in worker['read_ids'] for p in ids):
            continue
        records.append({**fact,'id':worker['task']['id']+f'-f{i+1}', 'task_id':worker['task']['id'],
            'status':'unverified','pages':[{k:mapping[p].get(k) for k in
                 ('page_id','doc_id','document_name','page_number')} for p in ids]})
    return records


def validate_plan(value):
    requirements, tasks = value.get('requirements'), value.get('tasks')
    if not isinstance(requirements, list) or not requirements or not isinstance(tasks, list) or not 1 <= len(tasks) <= 2:
        raise ValueError('规划须包含要求和1–2项取证任务')
    req_ids = []
    for item in requirements:
        if not isinstance(item, dict) or not text(item.get('id')) or not text(item.get('text')):
            raise ValueError('要求缺少编号或内容')
        req_ids.append(item['id'])
    if len(set(req_ids)) != len(req_ids):
        raise ValueError('要求编号重复')
    task_ids, owners, normalized = set(), set(), []
    for task in tasks:
        if not isinstance(task, dict) or any(not text(task.get(k)) for k in ('id', 'name', 'question', 'query')):
            raise ValueError('取证任务缺少编号、责任名、问题或检索词')
        if task['id'] in task_ids:
            raise ValueError('任务编号重复')
        task_ids.add(task['id'])
        ids, deps = task.get('requirement_ids'), task.get('depends_on', [])
        if (not isinstance(ids, list) or not ids or any(i not in req_ids for i in ids)
                or not isinstance(deps, list) or any(not text(i) for i in deps)):
            raise ValueError('任务责任或依赖错误')
        normalized.append(dict(task, requirement_ids=list(dict.fromkeys(ids)), depends_on=list(dict.fromkeys(deps))))
    for task in normalized:
        if task['id'] in task['depends_on'] or any(i not in task_ids for i in task['depends_on']):
            raise ValueError('依赖不存在或自我依赖')
    if len(normalized) == 2 and all(t['depends_on'] for t in normalized):
        raise ValueError('任务依赖成环')
    # Only remove exact duplicate responsibility/scope, never guess semantic equivalence.
    merged = []
    redirects = {}
    for task in normalized:
        signature = (tuple(sorted(task['requirement_ids'])), task.get('scope', ''), task['question'], task['query'])
        prior = next((p for p in merged if p['_signature'] == signature), None)
        if prior:
            redirects[task['id']] = prior['id']
        else:
            merged.append(dict(task, _signature=signature))
    for task in merged:
        task.pop('_signature')
        task['depends_on'] = list(dict.fromkeys(redirects.get(i, i) for i in task['depends_on']))
        if task['id'] in task['depends_on']:
            raise ValueError('合并后任务自我依赖')
        owners.update(task['requirement_ids'])
    if owners != set(req_ids):
        raise ValueError('问题要求缺少取证负责人')
    return requirements, merged


def calculation_checks(calculations, allowed):
    if not isinstance(calculations, list):
        raise ValueError('calculations须为数组')
    checks = []
    for item in calculations:
        if not isinstance(item, dict):
            raise ValueError('计算步骤须为对象')
        a, b, claimed = item.get('a'), item.get('b'), item.get('result')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a, b, claimed)):
            raise ValueError('操作数和结果须为有限数值')
        op = item.get('operation')
        if op not in ('add', 'subtract', 'multiply', 'divide', 'percent'):
            raise ValueError('仅支持四则运算及比例')
        ids = item.get('evidence_page_ids', [])
        if not isinstance(ids, list) or not ids or any(i not in allowed for i in ids):
            raise ValueError('计算操作数缺少本轮原页')
        actual = None if op in ('divide','percent') and b == 0 else (
            a+b if op == 'add' else a-b if op == 'subtract' else a*b if op == 'multiply'
            else a/b*(100 if op == 'percent' else 1))
        checks.append({**item, 'computed_result':actual,
                       'arithmetic_ok':actual is not None and math.isclose(actual,claimed,rel_tol=1e-4,abs_tol=1e-4),
                       'scope':'Arithmetic only; source operands, denominator and units require page verification.'})
    return checks


def validate_verification(value, draft, requirements, tasks):
    decisions, issues = value.get('claims'), value.get('issues')
    expected = {c['id'] for c in draft['numbered_claims']}
    if not isinstance(decisions,list) or not isinstance(issues,list):
        raise ValueError('核验须逐条判断声明并列出问题')
    ids = []
    for row in decisions:
        if (not isinstance(row,dict) or row.get('id') not in expected
                or type(row.get('supported')) is not bool or not text(row.get('reason'))):
            raise ValueError('声明核验格式错误')
        ids.append(row['id'])
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError('核验遗漏或重复声明')
    req_ids, task_ids = {r['id'] for r in requirements}, {t['id'] for t in tasks}
    for issue in issues:
        if (not isinstance(issue,dict) or not text(issue.get('detail'))
                or issue.get('requirement_id') not in req_ids
                or type(issue.get('needs_retrieval')) is not bool
                or issue.get('task_id') not in task_ids):
            raise ValueError('核验问题缺少要求、负责人或处理方式')
        if issue['needs_retrieval'] and not text(issue.get('query')):
            raise ValueError('补查缺少定向检索词')
    return value


def build_research_graph(runtime, saver):
    r = runtime
    StateGraph, START, END, Send, _ = require_multiagent()
    path = r.root/'research.json'

    def save(state, stage, **extra):
        prior = read_json(path) if path.exists() else {}
        workers = {role:{k:v for k,v in w.items() if k not in ('messages','pending')}
                   for role,w in state.get('workers',{}).items()}
        value = {**prior, **{k:state[k] for k in ('requirements','tasks','draft','verification','rework','version') if k in state},
                 'workers':workers, 'stage':stage, **extra}
        atomic_json(path,value)

    def request(state, logical, role, messages, phase, max_tokens, reserve, tools=None):
        save(state, phase)
        return r.request(logical,role,messages,tools,phase,max_tokens,reserve=reserve)

    def normalize(state):
        history, question = r.execution.history, r.execution.question
        call = lambda m:request(state,'research-query','coordinator',m,'query_rewrite' if history else 'query_translation',512,(7,8704))
        if history:
            r.emit('query_rewrite','started',role='coordinator')
            query, status, debit = rewrite_followup_query(question,history,call), 'contextual_rewrite', {}
            r.emit('query_rewrite','completed',role='coordinator',query=query)
        elif re.search(r'[\u3400-\u9fff]',question):
            cached = r.execution.translation
            if cached is not None and validate_translation(cached,question,r.config):
                query, status = cached['query'], 'cached_quality_only'
                debit = {k:cached[k] for k in ('identity','requests','output_tokens','seconds')}
            else:
                query, status, debit = convert_query(question,call), 'live', {}
        else:
            query, status, debit = question, 'not_needed', {}
        r.translation = {'query':query,'status':status,'debit':debit}
        atomic_json(r.root/'translation.json',r.translation)
        r.check_time()
        return {'query':query,'version':0,'phase':'main'}

    def plan(state):
        r.emit('coordinate','started',role='coordinator')
        docs = {p['doc_id']:p.get('document_name',p['doc_id']) for p in r.execution.mapping.values()}
        value = json_value(request(state,'research-plan','coordinator',[
            {'role':'system','content':COLLABORATION_RULES+'\nPlan document research; do not answer. '
             'Return JSON {"requirements":[{"id":"r1","text":"required fact or comparison"}],'
             '"tasks":[{"id":"t1","name":"responsibility label","question":"assigned subquestion",'
             '"query":"English retrieval query","scope":"subject, source clues, period and conditions",'
             '"requirement_ids":["r1"],"depends_on":[]}]}. '
             'Create 1–2 complementary evidence tasks with distinct responsibilities. Assign every requirement. '
             'Use depends_on only when a task needs the other task result. Single-source questions may need one task. '
             'The document catalogue is metadata, not evidence. Never infer a conclusion from a title.'},
            {'role':'user','content':json.dumps({'question':r.execution.question,'query':state['query'],
                                               'history':r.execution.history,'documents':docs},ensure_ascii=False)}],
            'plan',1024,(6,7680)))
        requirements, tasks = validate_plan(value)
        roles = {t['id']:f'worker-{i}' for i,t in enumerate(tasks,1)}
        workers = {}
        for task in tasks:
            role = roles[task['id']]
            workers[role] = {'role':role,'task':task,'dependencies':[roles[i] for i in task['depends_on']],
                'step':0,'done':False,'failed':None,'read_ids':[], 'messages':[
                    {'role':'system','content':COLLABORATION_RULES+'\nGather evidence for your assigned requirements only. '
                     'Use search_pages and read_page. Search snippets are not evidence. Preserve source, period, conditions and units. '
                     'You have at most four model decisions; read relevant pages before spending the remaining decisions searching. '
                     'Several read_page calls in one response are allowed. Stop tools when enough evidence is available. '
                     'An optional final report may use JSON {"facts":[{"text":"candidate fact","evidence_page_ids":[], '
                     '"requirement_ids":[],"subject":"","period":"","conditions":"","unit":"","location":""}],"gaps":[]}.'},
                    {'role':'user','content':json.dumps(task,ensure_ascii=False)}]}
        used = sum(r.role_usage(role,'main')[0] for role in workers)
        tokens = sum(r.role_usage(role,'main')[2] for role in workers)
        r.allocate(list(workers),'main',limits=(max(0,8-used),max(0,5-r.resources()['searches_used']),max(0,6144-tokens)))
        update = {'requirements':requirements,'tasks':tasks,'workers':workers}
        save({**state,**update},'gather')
        r.emit('coordinate','completed',role='coordinator',tasks=tasks,requirements=requirements)
        return update

    def dispatch(state):
        workers = state['workers']
        ready = []
        for w in workers.values():
            if not w['done'] and all(workers[d]['done'] and not workers[d].get('failed') for d in w['dependencies']):
                ready.append(Send('worker_model',{'worker':w,'phase':state['phase']}))
        return ready or 'gather_end'

    def decision(state):
        w, phase = copy.deepcopy(state['worker']), state['phase']
        w['step'] += 1
        if w['step'] > (2 if phase == 'rework' else 4):
            w['done'] = True
            return {'workers':{w['role']:w}}
        # Preserve source pages after a dependency; reports remain untrusted.
        if w['step'] == 1 and w['dependencies']:
            prior = read_json(path).get('workers',{})
            w['messages'].append({'role':'user','content':'UNTRUSTED PREDECESSOR RESULTS\n'+json.dumps(
                {d:prior.get(d,{}) for d in w['dependencies']},ensure_ascii=False)})
        try:
            response = r.request(f'research-{phase}-{w["role"]}-{w["step"]}',w['role'],
                r.materialize(w['messages']),TOOLS,phase,768,reserve=(2,3072) if phase=='rework' else (6,7680))
            w['pending'] = response.get('tool_calls') or []
            if not isinstance(w['pending'],list) or any(not isinstance(c,dict) or not isinstance(c.get('function'),dict) for c in w['pending']):
                raise ValueError('工具调用结构错误')
            w['messages'].append({'role':'assistant',**response})
            if not w['pending']:
                w.update(done=True,findings=response.get('content') or '')
        except TaskStopped as exc:
            if str(exc) in ('worker_request_limit','request_limit','output_token_limit'):
                w.update(done=True,findings='Use actual pages already received; no extra summary request.')
            else:
                w.update(done=True,failed=str(exc))
        return {'workers':{w['role']:w}}

    def tools(state):
        updates = {}
        phase = state['phase']
        for role, original in sorted(state['workers'].items()):
            w = copy.deepcopy(original)
            for i, call in enumerate(w.pop('pending',[])):
                f = call['function']
                try:
                    params = json.loads(f.get('arguments',''))
                except (TypeError,ValueError):
                    params = None
                result = r.tool(f'research-{phase}-{role}-{w["step"]}-tool{i}',role,f.get('name'),params,phase,
                                limits={'searches':6 if phase=='rework' else 5,'pages':10 if phase=='rework' else 8})
                w['messages'].append({'role':'tool','tool_call_id':call.get('id',''),'content':json.dumps(result,ensure_ascii=False)})
                if f.get('name')=='read_page' and result.get('status')=='read':
                    pid = result['page_id']
                    if pid not in w['read_ids']:
                        w['read_ids'].append(pid)
                        w['messages'].append({'evidence_page_id':pid})
            if w['step'] >= (2 if phase=='rework' else 4):
                w['done'] = True
            w['candidate_facts'] = candidate_facts(w,r.execution.mapping)
            updates[role] = w
        save({**state,'workers':updates},'gather' if phase=='main' else 'rework_gather')
        return {'workers':updates}

    def after_tools(state):
        if any(w.get('failed') for w in state['workers'].values()):
            return 'gather_end'
        return dispatch(state)

    def gather_end(state):
        updates, failure = {}, next((w['failed'] for w in state['workers'].values() if w.get('failed')),None)
        if failure:
            for role,w in state['workers'].items():
                if not w['done']:
                    updates[role] = dict(w,done=True,failed='blocked',blocked=True)
        save({**state,'workers':{**state['workers'],**updates}},'gather_failed' if failure else 'analyze')
        return {'workers':updates, **({'reason':failure} if failure else {})}

    def evidence(state):
        ids = sorted({pid for w in state['workers'].values() for pid in w['read_ids'] if pid in r.execution.read})
        return ids, r.materialize([{'evidence_page_id':pid} for pid in ids])

    def analyze(state):
        version = state.get('version',0)
        ids, pages = evidence(state)
        r.emit('research_analysis','started',role='analyst',version=version)
        _, contract = answer_contract(r.execution.question)
        messages = [{'role':'system','content':COLLABORATION_RULES+'\nAnalyze the actual pages, not merely worker summaries. '
            'Return JSON with exactly keys answer, coverage, calculations. answer uses the three-field answer format below. '
            'coverage is an array of {requirement_id, covered: boolean, note}. '
            'calculations is an array of {operation: add|subtract|multiply|divide|percent, a: number, b: number, '
            'result: number, unit: string, evidence_page_ids: string[]}; use [] when no numeric calculation is needed. '
            'Separate source subjects and periods, preserve conditions and exclusions. '
            'Do not convert forecasts to measured facts or a bibliography mention to substantive evidence.\n'
            +final_answer_contract(contract,ids)},
            {'role':'user','content':json.dumps({'question':r.execution.question,'requirements':state['requirements'],
                'tasks':state['tasks'],'worker_reports':[{'task':w['task'],'findings':w.get('findings',''),
                 'facts':w.get('candidate_facts',[]),'read_ids':w['read_ids']} for w in state['workers'].values()],
                'previous_draft':state.get('draft') if version else None,
                'verification_issues':state.get('verification') if version else None},ensure_ascii=False)},*pages]
        value = json_value(request(state,f'research-analysis-v{version}','analyst',messages,
                                  f'analysis_v{version}',2048,(1,1024) if version else (5,5632)))
        answer = validate_answer(value['answer'],set(ids))
        coverage = value.get('coverage')
        if not isinstance(coverage,list) or any(not isinstance(c,dict) or c.get('requirement_id') not in
                {x['id'] for x in state['requirements']} or type(c.get('covered')) is not bool for c in coverage):
            raise ValueError('草稿要求覆盖格式错误')
        draft = {'version':version,'answer':answer,'coverage':coverage,
                 'calculations':calculation_checks(value.get('calculations',[]),set(ids)),
                 'numbered_claims':[{'id':f'v{version}-c{i+1}',**c} for i,c in enumerate(answer['claims'])],
                 'page_ids':ids,'structure_valid':True}
        atomic_json(r.root/f'draft-v{version}.json',draft)
        save({**state,'draft':draft},f'verify_v{version}')
        r.emit('research_analysis','completed',role='analyst',version=version)
        return {'draft':draft}

    def verify(state):
        version = state.get('version',0)
        ids, pages = evidence(state)
        r.emit('research_verify','started',role='verifier',version=version)
        response = request(state,f'research-verify-v{version}','verifier',[
            {'role':'system','content':COLLABORATION_RULES+'\nIndependently verify the draft against the original question and actual pages. '
             'Check every numbered claim exactly once. Check requirements even if the draft coverage says complete. '
             'Return JSON {"claims":[{"id":"v0-c1","supported":true,"reason":"short page-grounded judgement"}],'
             '"issues":[{"requirement_id":"r1","task_id":"t1","kind":"missing_evidence|wrong_source|condition|calculation|unsupported",'
             '"detail":"specific missing or incorrect part","needs_retrieval":true,"query":"targeted English query"}]}. '
             'Use claims=[] when the draft has none. Use issues=[] if the answer appropriately explains actual evidence limits. '
             'Do not mark claims supported merely because citation IDs exist. Bibliographies, snippets, forecasts and opinions '
             'must not be mistaken for report findings or observations. If actual pages suffice for correction, set needs_retrieval=false. '
             'Keep reasons and issues concise. Do not rewrite the answer. This is model review, not human verification.'},
            {'role':'user','content':json.dumps({'question':r.execution.question,'requirements':state['requirements'],
                'tasks':state['tasks'],'draft':state['draft']},ensure_ascii=False)},*pages],
            f'verify_v{version}',1024,(0,0) if version else (4,4608))
        try:
            value = validate_verification(json_value(response),state['draft'],state['requirements'],state['tasks'])
        except (ValueError,KeyError,TypeError) as exc:
            raise TaskStopped('verification_invalid_output',{'error':str(exc)}) from exc
        atomic_json(r.root/f'verification-v{version}.json',value)
        save({**state,'verification':value},'verified')
        r.emit('research_verify','completed',role='verifier',version=version,issues=value['issues'])
        return {'verification':value}

    def needs_rework(state):
        value = state['verification']
        return (not state.get('version') and r.config.get('verification_rework',True)
                and (bool(value['issues']) or any(not c['supported'] for c in value['claims'])))

    def rework(state):
        issue = next((i for i in state['verification']['issues'] if i['needs_retrieval']),None)
        updates = {role:dict(w,done=True) for role,w in state['workers'].items()}
        if issue:
            role = next(role for role,w in updates.items() if w['task']['id']==issue['task_id'])
            w = copy.deepcopy(updates[role])
            w.update(step=0,done=False,failed=None,dependencies=[])
            w['messages'].append({'role':'user','content':'Targeted rework, at most TWO decisions, then analysis reads pages directly. '
                'Prefer relevant candidates already retrieved. If searching, read the relevant page next.\n'+json.dumps(issue,ensure_ascii=False)})
            updates[role] = w
            used, _, tokens = r.role_usage(role,'rework')
            r.allocate([role],'rework',limits=(max(0,2-used),max(0,6-r.resources()['searches_used']),max(0,1536-tokens)))
        record = {'triggered':True,'retrieval':bool(issue),'task_id':issue['task_id'] if issue else None}
        save({**state,'workers':updates,'version':1,'rework':record,'phase':'rework'},'rework')
        r.emit('research_rework','started',role='coordinator',**record)
        return {'workers':updates,'version':1,'rework':record,'phase':'rework'}

    def finish(state):
        draft, verification = state['draft'], state['verification']
        supported = {c['id'] for c in verification['claims'] if c['supported']}
        claims = [{k:c[k] for k in ('text','evidence_page_ids')} for c in draft['numbered_claims'] if c['id'] in supported]
        missing = list(draft['answer']['unresolved'])
        details = [i['detail'] for i in verification['issues']]
        rejected = [c['text'] for c in draft['numbered_claims'] if c['id'] not in supported]
        # System notices state the missing request, not an unverified replacement factual claim.
        prefix = '尚未核实：' if r.execution.language=='zh' else 'Unverified: '
        missing.extend(prefix+d for d in details+rejected)
        missing = list(dict.fromkeys(missing))
        if not claims and not missing:
            missing = ['缺少支持问题的证据。' if r.execution.language=='zh' else 'Evidence for the requested answer is missing.']
        answer = validate_answer({'claims':claims,'insufficient_evidence':bool(missing) or draft['answer']['insufficient_evidence'],
                                  'unresolved':missing},set(draft['page_ids']))
        save(state,'finished',verification_status='limited' if answer['insufficient_evidence'] else 'passed')
        atomic_json(r.root/'final-evidence.json',{'page_ids':draft['page_ids']})
        return {'answer':answer,'reason':'answered'}

    builder = StateGraph(ResearchState)
    for name,fn in [('normalize',normalize),('plan',plan),('worker_model',decision),('tools',tools),
                    ('gather_end',gather_end),('analyze',analyze),('verify',verify),('rework',rework),('finish',finish)]:
        builder.add_node(name,fn)
    builder.add_edge(START,'normalize')
    builder.add_edge('normalize','plan')
    builder.add_conditional_edges('plan',dispatch)
    builder.add_edge('worker_model','tools')
    builder.add_conditional_edges('tools',after_tools)
    builder.add_conditional_edges('gather_end',lambda s:END if s.get('reason') else 'analyze')
    builder.add_edge('analyze','verify')
    builder.add_conditional_edges('verify',lambda s:'rework' if needs_rework(s) else 'finish')
    builder.add_conditional_edges('rework',lambda s:dispatch(s) if s['rework']['retrieval'] else 'analyze')
    builder.add_edge('finish',END)
    return builder.compile(checkpointer=saver)


def research_result(runtime, result, state):
    path = runtime.root/'research.json'
    research = read_json(path) if path.exists() else {}
    research['verification_status'] = research.get('verification_status','incomplete') if result['completed'] else 'incomplete'
    research['failure_stage'] = None if result['completed'] else research.get('stage','normalize')
    research['drafts'] = [read_json(p) for p in sorted(runtime.root.glob('draft-v*.json'))]
    research['verifications'] = [read_json(p) for p in sorted(runtime.root.glob('verification-v*.json'))]
    records = []
    for pid,page in sorted(runtime.execution.read.items()):
        records.append({**page,'received_by':[role for role,w in state.get('workers',{}).items() if pid in w['read_ids']],
                        'analysis_versions':[d['version'] for d in research['drafts'] if pid in d['page_ids']]})
    research['pages'] = records
    stage = research.get('stage','normalize')
    if stage.startswith('verify'):
        need = {'requests':1,'output_tokens':1024}
    elif stage.startswith('analysis') or stage=='analyze':
        need = {'requests':2,'output_tokens':3072}
    else:
        need = {'requests':3,'output_tokens':3840}
    result.update(research=research,resume_requirements=need)
    result['scope'] = 'Research workflow; verification is model judgement, human=not_checked.'
