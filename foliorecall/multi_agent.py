"""Bounded document collaboration. LangGraph is an optional runtime dependency."""
import copy
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Annotated, TypedDict

from .agent import (AgentExecution, ANSWER_RULES, TOOLS, answer_contract,
                    final_answer_contract, validate_answer)
from .generation import TaskStopped, validate_agent_config
from .io import read_json
from .qa_storage import atomic_json, digest, output_lock
from .qa_translation import convert_query, rewrite_followup_query, validate_translation

# Intermediate roles have their own wire contracts, not the final answer schema.
COLLABORATION_RULES = ANSWER_RULES.partition('Return JSON with exactly:')[0]


def _merge(left, right):
    return {**left, **right}


class GraphState(TypedDict, total=False):
    query: str
    tasks: list
    workers: Annotated[dict, _merge]
    phase: str
    gap_used: bool
    answer: dict
    reason: str


def require_multiagent():
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.types import Send
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise ValueError('协作模式需要安装 foliorecall[multiagent]') from exc
    return StateGraph, START, END, Send, SqliteSaver


def can_resume_graph(output):
    """Only known local boundaries may relax the old running-attempt guard."""
    root = Path(output)
    if not (root/'manifest.json').exists():
        return False
    for path in (root/'operations').glob('*.json'):
        op = read_json(path)
        if op['kind']=='request' and 'outcome' not in op:
            return False
    return True


class ScopedProvider:
    """No worker mutates the parent provider's request context."""
    def __init__(self, provider, context):
        self.provider, self.fixed = provider, context
        self.context = {}
        self.key = getattr(provider, 'key', None)

    def complete(self, messages, tools, max_tokens, timeout):
        scoped = copy.copy(self.provider)
        scoped.context = {**getattr(self.provider, 'context', {}), **self.context, **self.fixed}
        return scoped.complete(messages, tools, max_tokens, timeout)


class Collaboration:
    def __init__(self, root, execution, attempt, query_lock=None):
        self.root, self.execution, self.attempt = Path(root), execution, attempt
        self.config = execution.agent_config
        self.on_event = execution.on_event
        self.lock = threading.RLock()
        self.query_lock = query_lock or threading.Lock()
        self.started = time.monotonic()
        self.ops = {p.stem: read_json(p) for p in sorted((self.root/'operations').glob('*.json'))}
        self.active_before = read_json(self.root/'elapsed.json')['seconds'] if (self.root/'elapsed.json').exists() else 0.
        clock = self.root/f'clock-{attempt}.json'
        if not clock.exists():
            atomic_json(clock,{'base_active_seconds':self.active_before})
        self.attempt_base = read_json(clock)['base_active_seconds']
        self.translation = read_json(self.root/'translation.json') if (self.root/'translation.json').exists() else {}
        self.quotas = read_json(self.root/f'quotas-{attempt}.json') if (self.root/f'quotas-{attempt}.json').exists() else {}
        self.events = []
        self.trace = []
        for op in self.ops.values():
            if op.get('kind') == 'tool' and op.get('outcome'):
                self.trace.extend(op['outcome'].get('trace', []))
        self.execution.searched = [row['arguments']['query'].strip().casefold() for row in self.trace
                                  if row.get('tool')=='search_pages' and row.get('status')=='ok']
        self._restore_pages()

    def _restore_pages(self):
        for op in self.ops.values():
            outcome = op.get('outcome', {})
            if op.get('tool') == 'read_page' and outcome.get('result', {}).get('status') == 'read':
                page = outcome['result']
                self.execution.read[page['page_id']] = page
                self.execution.cumulative_pages.add(page['page_id'])

    def elapsed(self):
        return self.active_before + time.monotonic()-self.started

    def save_op(self, key, value):
        self.ops[key] = value
        atomic_json(self.root/'operations'/f'{key}.json', value)
        atomic_json(self.root/'elapsed.json', {'seconds': self.elapsed()})

    def emit(self, event, status, **data):
        with self.lock:
            budget = self.resources()
            value = {'event': event, 'status': status, 'elapsed_seconds': self.elapsed(),
                     'sequence': len(self.events)+1, **data,
                     'model_requests':budget['requests_used'],'searches':budget['searches_used']}
            self.events.append(value)
            if self.on_event:
                self.on_event(value)

    def calls(self):
        return sorted((op for op in self.ops.values() if op['kind'] == 'request'),key=lambda op:op['sequence'])

    def resources(self):
        with self.lock:
            return self._resources()

    def _resources(self):
        calls = [op for op in self.calls() if op.get('outcome', {}).get('record', {}).get('cost_status') != 'not_sent']
        known, unknown = 0, 0
        for op in calls:
            usage = op.get('outcome', {}).get('record', {}).get('usage') or {}
            if type(usage.get('completion_tokens')) is int:
                known += usage['completion_tokens']
            else:
                unknown += op['max_tokens']
        debit = self.translation.get('debit', {})
        return {'requests_used': len(calls)+debit.get('requests', 0),
                'known_output_tokens': known+debit.get('output_tokens', 0),
                'unknown_output_tokens_reserved': unknown,
                'searches_used': sum(op.get('tool') == 'search_pages' for op in self.ops.values()),
                'distinct_page_ids': sorted(self.execution.read),
                'active_seconds_used': self.elapsed()+debit.get('seconds', 0),
                'translation_debit': debit.get('identity'),
                'budget_snapshot_complete': all('outcome' in op for op in calls)}

    def check_time(self):
        if self.resources()['active_seconds_used'] >= self.config['task_timeout']:
            raise TaskStopped('task_timeout')

    def _cached(self, logical):
        previous = sorted((op for op in self.ops.values() if op['logical'] == logical), key=lambda o: o['attempt'])
        for op in previous:
            if 'outcome' not in op:
                raise TaskStopped('unknown_inflight', {'operation_id': logical})
            if op['outcome'].get('error'):
                from .qa_runtime import transient
                if op['attempt'] == self.attempt or not transient(op['outcome']['error']):
                    raise TaskStopped(op['outcome']['error'], op['outcome'].get('record', {}))
            else:
                return op['outcome']
        return None

    def role_usage(self, role, phase):
        ops = [op for op in self.ops.values() if op.get('role') == role and op.get('phase') == phase]
        requests = [op for op in ops if op['kind'] == 'request' and op.get('outcome', {}).get('record', {}).get('cost_status') != 'not_sent']
        tokens = 0
        for op in requests:
            value = (op.get('outcome', {}).get('record', {}).get('usage') or {}).get('completion_tokens')
            tokens += value if type(value) is int else op.get('max_tokens',0)
        return len(requests), sum(op.get('tool') == 'search_pages' for op in ops), tokens

    def request(self, logical, role, messages, tools=None, phase='main', max_tokens=2048, reserve=None):
        with self.lock:
            cached = self._cached(logical)
            if cached is not None:
                return cached['message']
            # A provider wait applies across workers, not just to the failed role.
            for saved in reversed(self.calls()):
                record = saved.get('outcome',{}).get('record',{})
                if record.get('retry_after_status') in ('invalid','unresolved'):
                    raise TaskStopped('retry_after_unresolved',record)
                if saved['attempt'] == self.attempt and record.get('retry_after_seconds',0)>0:
                    raise TaskStopped(saved['outcome'].get('error') or 'retry_after_unresolved',record)
            self.check_time()
            budget = self.resources()
            reserve_requests = 0 if phase == 'final_answer' else 1 if phase == 'merge' else 2
            reserve_tokens = 1024*reserve_requests
            if reserve is not None:
                reserve_requests, reserve_tokens = reserve
            if budget['requests_used']+reserve_requests >= self.config['max_requests']:
                raise TaskStopped('request_limit')
            limit = min(max_tokens, self.config['max_output_tokens']-budget['known_output_tokens']-
                        budget['unknown_output_tokens_reserved']-reserve_tokens)
            if role.startswith('worker'):
                used, _, tokens = self.role_usage(role, phase)
                quota = self.quotas[role]
                if used >= quota['requests']:
                    raise TaskStopped('worker_request_limit')
                limit = min(limit, quota['tokens']-tokens)
            if limit <= 0:
                raise TaskStopped('output_token_limit')
            key = f'{logical}-attempt{self.attempt}'
            sequence = 1+max((op.get('sequence', 0) for op in self.calls()), default=0)
            op = {'logical': logical, 'kind': 'request', 'role': role, 'phase': phase,'started_at':time.time(),
                  'attempt': self.attempt, 'sequence': sequence, 'max_tokens': limit}
            self.save_op(key, op)  # Persist intent before any external side effect.
        context = {'task_id': getattr(self.execution.provider, 'context', {}).get('task_id', self.root.name),
                   'graph_task_id': self.root.name, 'agent_id': role, 'node': logical,
                   'request_sequence': sequence, 'phase': phase, 'attempt': self.attempt}
        provider = ScopedProvider(self.execution.provider, context)
        config = dict(self.config, max_requests=1, max_output_tokens=limit+(1024 if tools else 0))
        execution = AgentExecution(self.execution.question, 'agent', None, {}, None, [], config, provider,
                                   on_event=lambda e: self.emit(e['event'], e['status'], role=role, node=logical,
                                       **{k:v for k,v in e.items() if k not in ('event','status','elapsed_seconds')})
                                       if e['status']=='started' else None)
        execution.deadline = execution.started + max(0., self.config['task_timeout']-self.resources()['active_seconds_used'])
        outcome = {}
        try:
            outcome['message'] = execution.generate(messages, tools, phase)
        except TaskStopped as exc:
            outcome['error'] = str(exc)
        finally:
            # BaseException is intentionally not converted to a terminal response.
            if execution.calls:
                outcome['record'] = {**execution.calls[-1], **context}
            elif outcome.get('error'):
                outcome['record'] = {'usage':None, 'cost_status':'not_sent', **context}
            if 'message' in outcome or 'error' in outcome:
                with self.lock:
                    self.save_op(key, {**op, 'outcome': outcome,'finished_at':time.time()})
                self.emit('model_request','error' if outcome.get('error') else 'completed',role=role,node=logical,
                          phase=phase,cost_cny=outcome.get('record',{}).get('cost_cny'),
                          reason=outcome.get('error',''))
        if 'error' in outcome:
            raise TaskStopped(outcome['error'], outcome['record'])
        return outcome['message']

    def allocate(self, roles, phase='main', limits=None):
        budget = self.resources()
        reserve = 2 if phase == 'main' else 1
        remaining = [max(0, self.config['max_requests']-budget['requests_used']-reserve),
                     max(0, self.config['max_searches']-budget['searches_used']),
                     max(0, self.config['max_output_tokens']-budget['known_output_tokens']-
                         budget['unknown_output_tokens_reserved']-1024*reserve)]
        if limits is not None:
            remaining = [min(n, max(0, limit)) for n, limit in zip(remaining, limits)]
        quotas = {}
        for i, role in enumerate(roles):
            used = self.role_usage(role, phase)
            quotas[role] = dict(zip(('requests','searches','tokens'),
                [old+n//len(roles)+int(i<n%len(roles)) for old,n in zip(used,remaining)]))
        self.quotas = quotas
        atomic_json(self.root/f'quotas-{self.attempt}.json', quotas)

    def tool(self, logical, role, name, params, phase, limits=None):
        with self.lock:
            cached = self._cached(logical)
            if cached is not None:
                return cached['result']
            self.check_time()
            op = {'logical': logical, 'kind': 'tool', 'role': role, 'phase': phase,
                  'attempt': self.attempt, 'tool': name}
            key = f'{logical}-attempt{self.attempt}'
            # Tool operations are local and serialized; no paid call is made here.
            before = len(self.execution.trace)
            timing_before = dict(self.execution.timings)
            search_before = self.resources()['searches_used']
            self.execution.searches = search_before
            self.execution.on_event = lambda e:self.emit(e['event'],e['status'],role=role,
                **{k:v for k,v in e.items() if k not in ('event','status','elapsed_seconds')})
            self.execution.deadline = self.execution.started + self.config['task_timeout']-self.active_before-self.translation.get('debit',{}).get('seconds',0)
            try:
                if name == 'search_pages' and isinstance(params, dict) and set(params) == {'query'}:
                    if limits and search_before >= limits['searches']:
                        raise TaskStopped('search_limit')
                    _, used, _ = self.role_usage(role, phase)
                    if used >= self.quotas[role]['searches']:
                        raise TaskStopped('search_limit')
                    self.execution.searches = self.resources()['searches_used']
                    with self.query_lock:
                        result = self.execution.search_pages(params['query'])
                elif name == 'read_page' and isinstance(params, dict) and set(params) == {'page_id'}:
                    pid = params['page_id']
                    if limits and pid not in self.execution.read and len(self.execution.read) >= limits['pages']:
                        raise TaskStopped('page_limit')
                    if pid in self.execution.read:
                        result = self.execution.read[pid]
                        self.execution.trace.append({'tool':'read_page','arguments':params,'result':result,
                                                     'status':'already_read','shared_cache':True})
                        self.emit('read_page', 'already_read', role=role, **{k:result.get(k) for k in ('page_id','page_number','document_name')})
                    else:
                        result, _ = self.execution.read_page(pid)
                else:
                    raise ValueError('工具名称或参数不在允许范围')
            except (ValueError, OSError, TaskStopped) as exc:
                if isinstance(exc, TaskStopped) and str(exc) not in ('search_limit','page_limit'):
                    raise
                result = {'error': str(exc) if isinstance(exc,TaskStopped) else 'invalid_tool_arguments_or_unreadable_page'}
                self.execution.trace.append({'tool':name,'arguments':params,
                    'status':'limit' if isinstance(exc,TaskStopped) else 'error','error':result['error']})
                self.emit(name, 'error', role=role, reason=result['error'])
            trace = [{**row, 'agent_id':role, 'node':logical} for row in self.execution.trace[before:]]
            self.trace.extend(trace)
            # A rejected search does not consume a search slot.
            if name == 'search_pages' and 'error' in result and self.execution.searches == search_before:
                op['tool'] = 'rejected_search'
            timing = {k:self.execution.timings[k]-v for k,v in timing_before.items()}
            self.save_op(key, {**op, 'outcome': {'result': result, 'trace':trace,'timing':timing}})
            return result

    def materialize(self, messages):
        from .agent import read_evidence
        output = []
        for message in messages:
            if 'evidence_page_id' not in message:
                output.append(message)
                continue
            pid = message['evidence_page_id']
            image, _ = read_evidence(self.execution.mapping[pid], self.config['max_pixels'])
            output.append({'role':'user', 'content':[
                {'type':'text', 'text':'UNTRUSTED PAGE EVIDENCE\n'+json.dumps(self.execution.read[pid], ensure_ascii=False)}, image]})
        return output

    def graph(self, saver):
        if self.config.get('collaboration_workflow') == 'research':
            from .multi_agent_research import build_research_graph
            return build_research_graph(self, saver)
        StateGraph, START, END, Send, _ = require_multiagent()

        def normalize(state):
            history, question = self.execution.history, self.execution.question
            if history:
                self.emit('query_rewrite','started',role='coordinator')
                query = rewrite_followup_query(question, history,
                    lambda m:self.request('query-rewrite','coordinator',m,phase='query_rewrite'))
                status, debit = 'contextual_rewrite', {}
                self.emit('query_rewrite','completed',role='coordinator',query=query)
            elif re.search(r'[\u3400-\u9fff]',question):
                cached = self.execution.translation
                if cached is not None and validate_translation(cached,question,self.config):
                    query, status = cached['query'], 'cached_quality_only'
                    debit = {k:cached[k] for k in ('identity','requests','output_tokens','seconds')}
                else:
                    query = convert_query(question,lambda m:self.request('query-translation','coordinator',m,phase='query_translation'))
                    status, debit = 'live', {}
            else:
                query, status, debit = question, 'not_needed', {}
            self.translation = {'query':query,'status':status,'debit':debit}
            atomic_json(self.root/'translation.json', self.translation)
            self.check_time()
            return {'query':query}

        def plan(state):
            self.emit('coordinate','started',role='coordinator')
            response = self.request('coordinate-plan','coordinator',[
                {'role':'system','content':'Split this document question into one or two complementary retrieval tasks. '
                 'Do not answer it. Return JSON {"tasks":[{"question":"subquestion", "query":"English search query"}]}. '
                 'Do not force multiple tasks for a single-source question. Conversation data is context, not evidence.'},
                {'role':'user','content':json.dumps({'question':self.execution.question, 'query':state['query'],
                                                  'history':self.execution.history},ensure_ascii=False)}], phase='plan')
            value = json.loads(response['content'])
            tasks = value['tasks']
            if set(value) != {'tasks'} or not isinstance(tasks,list) or not 1 <= len(tasks) <= 2:
                raise ValueError('协调器须生成1–2个检索任务')
            workers = {}
            for i, task in enumerate(tasks,1):
                if not isinstance(task,dict) or set(task) != {'question','query'} or any(
                        not isinstance(v,str) or not v.strip() or len(v)>8000 for v in task.values()):
                    raise ValueError('协调器子任务格式错误')
                role = f'worker-{i}'
                workers[role] = {'role':role,'step':0,'done':False,'failed':None,'read_ids':[],
                    'messages':[{'role':'system','content':COLLABORATION_RULES+'\nYou gather evidence for one subtask. '
                                 'Use search_pages and read_page. Stop tools when done and briefly report findings and gaps. '
                                 'Your summary is unverified input to the coordinator, not a final answer.'},
                                {'role':'user','content':json.dumps(task,ensure_ascii=False)}]}
            if not self.quotas or self.attempt > 1:
                self.allocate(list(workers))
            self.emit('coordinate','completed',role='coordinator',tasks=tasks)
            return {'tasks':tasks,'workers':workers,'phase':'main','gap_used':False}

        def dispatch(state):
            return [Send('worker_model',{'worker':w,'phase':state['phase']})
                    for w in state['workers'].values() if not w['done']]

        def decision(state):
            w, phase = copy.deepcopy(state['worker']), state['phase']
            role = w['role']
            w['step'] += 1
            logical = f'{phase}-{role}-{w["step"]}'
            try:
                response = self.request(logical,role,self.materialize(w['messages']),TOOLS,phase)
                w['pending'] = response.get('tool_calls') or []
                if not isinstance(w['pending'],list) or any(not isinstance(c,dict) or not isinstance(c.get('function'),dict) for c in w['pending']):
                    raise ValueError('工具调用结构错误')
                w['messages'].append({'role':'assistant',**response})
                if not w['pending']:
                    w.update(done=True, findings=response.get('content') or '')
            except TaskStopped as exc:
                if str(exc) in ('worker_request_limit','request_limit','output_token_limit'):
                    w.update(done=True, findings='Resources exhausted; use pages actually read.')
                else:
                    w.update(done=True, failed=str(exc))
            return {'workers':{role:w}}

        def tools(state):
            updated = {}
            for role, original in sorted(state['workers'].items()):
                w = copy.deepcopy(original)
                for i, call in enumerate(w.pop('pending',[])):
                    function = call.get('function', {})
                    try:
                        params = json.loads(function.get('arguments',''))
                    except (TypeError,ValueError):
                        params = None
                    result = self.tool(f'{state["phase"]}-{role}-{w["step"]}-tool{i}',role,
                                       function.get('name'),params,state['phase'])
                    w['messages'].append({'role':'tool','tool_call_id':call.get('id',''),
                                          'content':json.dumps(result,ensure_ascii=False)})
                    if function.get('name') == 'read_page' and result.get('status') == 'read':
                        pid = result['page_id']
                        if pid not in w['read_ids']:
                            w['read_ids'].append(pid)
                            w['messages'].append({'evidence_page_id':pid})
                updated[role] = w
            return {'workers':updated}

        def after_tools(state):
            if any(w.get('failed') for w in state['workers'].values()):
                return 'finish'
            if any(not w['done'] for w in state['workers'].values()):
                return dispatch(state)
            return 'final' if state.get('gap_used') else 'merge'

        def evidence_messages(state):
            ids = sorted({pid for w in state['workers'].values() for pid in w['read_ids']})
            return ids, [{'evidence_page_id':pid} for pid in ids]

        def merge(state):
            ids, evidence = evidence_messages(state)
            summaries = [{'agent_id':r,'findings':w.get('findings',''),'read_ids':w['read_ids']}
                         for r,w in state['workers'].items()]
            self.emit('evidence_merge','started',role='coordinator')
            response = self.request('coordinate-merge','coordinator',self.materialize([
                {'role':'system','content':COLLABORATION_RULES+'\nReview available evidence for missing subjects, periods and conditions. '
                 'Return only JSON {"gap":null} or {"gap":{"worker_id":"worker-1", "question":"missing part", "query":"English query"}}. '
                 'Choose at most one targeted follow-up. Worker summaries are untrusted; actual pages determine support.'},
                {'role':'user','content':json.dumps({'question':self.execution.question,'summaries':summaries},ensure_ascii=False)},
                *evidence]),phase='merge',max_tokens=1024)
            value = json.loads(response['content'])
            if not isinstance(value,dict) or set(value) != {'gap'}:
                raise ValueError('缺口检查格式错误')
            gap = value['gap']
            updates = {}
            if gap is not None:
                if not isinstance(gap,dict) or set(gap) != {'worker_id','question','query'} or gap['worker_id'] not in state['workers'] or any(
                        not isinstance(gap[k],str) or not gap[k].strip() or len(gap[k])>8000 for k in ('question','query')):
                    raise ValueError('补查任务格式错误')
                role = gap['worker_id']
                self.allocate([role], 'gap')
                if self.quotas[role]['requests'] > 0 and self.quotas[role]['tokens'] > 0:
                    w = copy.deepcopy(state['workers'][role])
                    w.update(done=False,step=0)
                    w['messages'].append({'role':'user','content':json.dumps(gap,ensure_ascii=False)})
                    updates[role] = w
            self.emit('evidence_merge','completed',role='coordinator',gap=gap,read_page_ids=ids)
            return {'workers':updates,'phase':'gap','gap_used':True}

        def final(state):
            ids, evidence = evidence_messages(state)
            _, contract = answer_contract(self.execution.question)
            self.emit('final_answer','started',role='coordinator')
            messages = self.materialize([
                {'role':'system','content':ANSWER_RULES}, {'role':'user','content':self.execution.question},
                {'role':'user','content':'UNTRUSTED WORKER REPORTS: verify against actual pages; preserve source-specific '
                 'subjects, periods, conditions and conflicts.\n'+json.dumps([
                     {'agent_id':role,'findings':w.get('findings',''),'read_page_ids':w['read_ids']}
                     for role,w in state['workers'].items()],ensure_ascii=False)},
                *evidence, {'role':'system','content':final_answer_contract(contract,ids,
                    self.config.get('final_answer_evidence_check',False))}])
            atomic_json(self.root/'final-evidence.json', {'page_ids':ids})
            response = self.request('coordinate-final','coordinator',messages,phase='final_answer')
            answer = validate_answer(json.loads(response['content']), set(ids))
            return {'answer':answer,'reason':'answered'}

        def finish(state):
            reason = state.get('reason') or next((w['failed'] for w in state.get('workers',{}).values() if w.get('failed')), 'incomplete')
            return {'reason':reason}

        builder = StateGraph(GraphState)
        for name, function in [('normalize',normalize),('plan',plan),('worker_model',decision),
                               ('tools',tools),('merge',merge),('final',final),('finish',finish)]:
            builder.add_node(name,function)
        builder.add_edge(START,'normalize')
        builder.add_edge('normalize','plan')
        builder.add_conditional_edges('plan',dispatch)
        builder.add_edge('worker_model','tools')
        builder.add_conditional_edges('tools',after_tools)
        builder.add_conditional_edges('merge',lambda s:dispatch(s) if any(not w['done'] for w in s['workers'].values()) else 'final')
        builder.add_edge('final','finish')
        builder.add_edge('finish',END)
        return builder.compile(checkpointer=saver)

    def result(self, state, reason=None, failure=None):
        budget = self.resources()
        calls = [op['outcome']['record'] for op in self.calls() if op['attempt']==self.attempt and 'outcome' in op]
        reason = reason or state.get('reason','incomplete')
        answer = state.get('answer') if reason == 'answered' else None
        answer = answer or {'claims':[], 'insufficient_evidence':True,
            'unresolved':['任务未完成；已取得证据及缺失工作另行保留。' if self.execution.language=='zh' else 'Task incomplete; collected evidence and missing work are retained separately.']}
        tools = [op for op in self.ops.values() if op['kind']=='tool' and op['attempt']==self.attempt]
        result = {'question':self.execution.question,'retrieval_query':self.translation.get('query',self.execution.question),
            'mode':'multi-agent','answer':answer,'answer_language':self.execution.language,'completed':reason=='answered',
            'termination_reason':reason,'evidence':list(self.execution.read.values()),'trace':self.trace,'api_calls':calls,
            'model_requests':budget['requests_used'],'actual_model_requests':sum(c.get('cost_status')!='not_sent' for c in calls),
            'searches':budget['searches_used'],'actual_searches':sum(op.get('tool')=='search_pages' for op in tools),
            'distinct_pages':len(self.execution.read),'output_tokens':budget['known_output_tokens'],
            'budget_state':budget,'translation_status':self.translation.get('status','not_needed'),
            'translation_history':self.execution.translation if self.translation.get('debit') else None,
            'translation_cost_share_cny':(self.execution.translation or {}).get('cost_cny',0)/3 if self.translation.get('debit') else 0.,
            'context_turns':len(self.execution.history),'failure':failure or {},
            'estimated_cost_cny':sum(c.get('cost_cny',0) for c in calls),
            'cost_complete':budget['budget_snapshot_complete'] and all(c.get('usage') is not None for c in calls),
            'timing':{'task_seconds':self.elapsed()-self.attempt_base,
                      'recovered_active_seconds':self.active_before-self.attempt_base,
                      'prior_attempt_active_seconds':self.attempt_base,
                      'api_seconds':sum(c.get('seconds',0) for c in calls),
                      **{k:sum(op.get('outcome',{}).get('timing',{}).get(k,0.) for op in tools)
                         for k in ('retrieval_seconds','page_read_seconds')}},
            'graph':{'workers':{r:{k:v for k,v in w.items() if k not in ('messages','pending')} for r,w in state.get('workers',{}).items()},
                     'gap_used':state.get('gap_used',False),'attempt':self.attempt},
            'progress':self.events,'scope':'collaboration functional output; citation membership checked, semantic quality not automatically proven'}
        if self.config.get('collaboration_workflow') == 'research':
            from .multi_agent_research import research_result
            research_result(self, result, state)
        return result


def ask_multi(question, model, query_config, index, pages, agent_config, provider, output,
              retrieval=None, evidence_texts=None, translation=None, conversation=None, on_event=None,
              resume=False, query_lock=None, cooldown_seconds=65, record_metadata=None):
    """A task directory owns its graph, immutable attempts and durable call records."""
    _, _, _, _, SqliteSaver = require_multiagent()
    validate_agent_config(agent_config)
    root = Path(output)
    if resume and (root/'manifest.json').exists():
        saved_identity = read_json(root/'manifest.json')['identity']
        if conversation is None:
            conversation = saved_identity['conversation']
        if translation is None:
            translation = saved_identity['translation']
        if record_metadata is None:
            record_metadata = saved_identity.get('record_metadata')
    texts = evidence_texts if evidence_texts is not None else (retrieval or {}).get('texts',{})
    from .qa_runtime import execution_code, transient
    identity = {'question':question,'query_config':query_config,'agent_config':agent_config,
                'pages':[{k:p.get(k) for k in ('page_id','doc_id','page_number','preview','source')} for p in pages],
                'texts':digest(texts),'retrieval':(retrieval.get('config',retrieval) if retrieval else None),
                'conversation':conversation,'translation':translation,
                'record_metadata':record_metadata,'code':execution_code()}
    with output_lock(root):
        if (root/'manifest.json').exists():
            if not resume:
                raise ValueError('协作任务目录已存在；接续需 --resume')
            if read_json(root/'manifest.json')['identity'] != identity:
                raise ValueError('协作任务恢复身份不匹配')
        else:
            if resume:
                raise ValueError('协作任务恢复目录不存在')
            if root.exists() and any(root.iterdir()):
                raise ValueError('协作任务目录非空')
            root.mkdir(parents=True,exist_ok=True)
            atomic_json(root/'manifest.json',{'identity':identity})
        attempts = sorted(root.glob('attempt-*.json'))
        attempt = 1
        if attempts:
            previous = read_json(attempts[-1])
            if previous['completed'] or not transient(previous['termination_reason']) or len(attempts)>=2:
                return previous
            attempt = 2
            from .qa_runtime import latest_service_wait
            service_wait, stopped = latest_service_wait([{'result':previous}])
            if stopped:
                raise TaskStopped(stopped)
            wait = max(cooldown_seconds,service_wait) if cooldown_seconds is not None else 0
            if wait:
                time.sleep(wait)
        execution = AgentExecution(question,'agent',model,query_config,index,pages,agent_config,provider,
                                   retrieval,texts,translation=translation,conversation=conversation,on_event=None)
        execution.on_event = on_event
        runtime = Collaboration(root,execution,attempt,query_lock)
        state, failure, reason = {}, {}, None
        try:
            if any(op['kind']=='request' and 'outcome' not in op for op in runtime.ops.values()):
                raise TaskStopped('unknown_inflight')
            connection = sqlite3.connect(root/'checkpoints.sqlite3',check_same_thread=False)
            try:
                graph = runtime.graph(SqliteSaver(connection))
                config = {'configurable':{'thread_id':f'task-attempt-{attempt}'},
                          'max_concurrency':agent_config.get('max_parallel_workers',2), 'recursion_limit':80}
                snapshot = graph.get_state(config)
                try:
                    state = graph.invoke(None if snapshot.values else {'workers':{}},config,durability='sync')
                except BaseException:
                    state = graph.get_state(config).values
                    raise
            finally:
                connection.close()
        except TaskStopped as exc:
            reason, failure = str(exc), exc.record
        except (ValueError,KeyError,TypeError) as exc:
            reason, failure = 'invalid_answer_or_evidence', {'error':type(exc).__name__,'message':str(exc)}
            if hasattr(exc,'code'):
                failure['output_diagnostic'] = {'code':exc.code,'path':exc.path,'message':str(exc)}
        result = runtime.result(state,reason,failure)
        result.update(record_metadata or {})
        runtime.emit('terminal','completed' if result['completed'] else 'failed',termination_reason=result['termination_reason'])
        result['progress'] = runtime.events
        atomic_json(root/f'attempt-{attempt:04d}.json',result)
        atomic_json(root/'result.json',result)
        atomic_json(root/'elapsed.json',{'seconds':runtime.elapsed()})
        return result
