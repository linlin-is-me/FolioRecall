"""Serial, resumable QA execution. Attempts are immutable evidence, not a queue service."""
from pathlib import Path
import math
import re
import time
import uuid

from .io import read_json, read_rows, provenance
from .qa_storage import atomic_json, digest, output_lock, restart_gap


def transient(reason):
    return reason in ('api_timeout','api_transport_error','api_http_408','api_http_429') or bool(re.fullmatch(r'api_http_5\d\d',reason))


def pause_reason(reason):
    return reason.startswith('api_') or reason in ('missing_usage','phase_budget_exhausted','invalid_answer_or_evidence')


def execution_code():
    root = Path(__file__).parent
    # Only relevant small implementation files, never models, pages or data scans.
    names = ('agent.py','generation.py','qa_runtime.py','qa_translation.py','qa_storage.py',
             'qa_evaluation.py','query.py','retrieval.py','search.py','evaluation.py','text_baseline.py')
    return {name:digest((root/name).read_text(encoding='utf-8')) for name in names}


def metadata(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path':str(path), 'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns}


def prepare_unit(spec, repeats=1, task_ids=None, suite_id='standalone', condition=None,
                 domain=None, repeat_authorized=False, translation_cache=None):
    from .agent import load_public_corpus
    from .generation import read_agent_config
    from .query import validate_query_config
    from .search import load_index
    from .retrieval import read_retrieval_config, load_retrieval
    from .qa_evaluation import validate_tasks
    from .qa_translation import validate_translation
    teacher, student = read_json(spec['config']), read_json(spec['query_config'])
    validate_query_config(student,teacher)
    _, pages = load_index(spec['index'],teacher)
    texts = load_public_corpus(spec['public_corpus'],pages)
    tasks, status = validate_tasks(spec['tasks'],pages)
    public = read_json(spec['public_corpus'])
    if status.get('schema_version') == 2 and any(public.get(k)!=status.get(k) for k in ('task','dataset','revision')):
        raise ValueError('公开资料与冻结任务的领域或 revision 不匹配')
    if task_ids is not None:
        selected = set(task_ids)
        if len(selected) != len(task_ids) or not selected.issubset({t['task_id'] for t in tasks}):
            raise ValueError('套件选择包含重复或未知任务')
        tasks = [t for t in tasks if t['task_id'] in selected]
    if repeats not in (1,3) or (repeats == 3 and (status.get('split') != 'development' or (len(tasks) != 10 and not repeat_authorized))):
        raise ValueError('三轮测量只允许已验证的全局十条开发任务')
    if repeats == 3 and translation_cache:
        raise ValueError('波动测量必须实时转换，不允许译文缓存')
    agent = read_agent_config(spec['agent_config'])
    probe = read_json(spec['probe'])
    if probe.get('passed') is not True or probe.get('model') != agent['model'] or probe.get('region') != agent['region']:
        raise ValueError('缺少同一固定模型和地域的已通过探针')
    retrieval = read_retrieval_config(spec['retrieval_config']) if spec.get('retrieval_config') else None
    load_retrieval(retrieval,pages)  # Identity checks before any model/provider load.
    cache = read_json(translation_cache) if translation_cache else {}
    for task in tasks:
        if task['task_id'] in cache:
            validate_translation(cache[task['task_id']],task['question_zh'],agent)
    name = condition or ('hybrid-' if retrieval else 'visual-')+spec['mode']
    domain = domain or status.get('task') or tasks[0].get('task') or 'corpus'
    if status.get('task') and domain != status['task']:
        raise ValueError('套件领域与冻结任务不匹配')
    jobs = []
    for repeat in range(repeats):
        for task in tasks:
            for language in ('en','zh'):
                key = dict(suite=suite_id, condition=name, domain=domain, task_id=task['task_id'], language=language, repeat=repeat)
                jobs.append({'key':key, 'question':task['question_'+language], 'task':task,
                             'translation':cache.get(task['task_id']) if language == 'zh' else None})
    files = [metadata(p) for p in sorted(Path(spec['index']).iterdir())
             if p.is_file() and p.name in ('index.faiss','pages.jsonl','config.json')]
    public = read_json(spec['public_corpus'])
    if public.get('texts'):
        files.append(metadata(Path(spec['public_corpus']).resolve().parent/public['texts']))
    if retrieval:
        files.append(metadata(Path(retrieval['text_index'])/'index.json'))
    identity = {'spec':{k:str(v) for k,v in spec.items()}, 'teacher':teacher,'student':student,
                'agent':agent,'retrieval':retrieval,'tasks':tasks,'status':status,'public':public,
                'files':files,'cache':cache,'probe':probe,'code':execution_code(), 'jobs':[j['key'] for j in jobs]}
    return {'identity':identity,'jobs':jobs,'spec':spec,'teacher':teacher,'student':student,
            'agent':agent,'retrieval_config':retrieval,'texts':texts}


def suite_jobs(units, manifest):
    """Optional full permutation; retain unit IDs used by the resident model."""
    jobs = [j for u in units for j in u['jobs']]
    cooldown = manifest.get('inter_task_cooldown_seconds',0)
    if type(cooldown) not in (int,float) or not math.isfinite(cooldown) or cooldown < 0:
        raise ValueError('任务间冷却时间必须是非负有限秒数')
    if 'job_order' not in manifest:
        return jobs
    order = manifest['job_order']
    if not isinstance(order,list) or any(not isinstance(k,dict) for k in order):
        raise ValueError('job_order 必须为完整运行键数组')
    lookup = {digest(j['key']):j for j in jobs}
    keys = [digest(k) for k in order]
    if len(keys)!=len(jobs) or len(set(keys))!=len(keys) or set(keys)!=set(lookup):
        raise ValueError('job_order 必须无重复地覆盖全部运行键')
    return [lookup[k] for k in keys]


def prepare_suite(path, translation_cache=None):
    manifest = read_json(path)
    if manifest.get('conditions') != ['visual-rag','hybrid-rag','hybrid-agent']:
        raise ValueError('套件必须明确固定三个比较条件')
    domains = manifest['domains']
    if len({d['name'] for d in domains}) != len(domains):
        raise ValueError('套件领域重复')
    repeats = manifest.get('repeats',1)
    if repeats == 3:
        selected = [(d['name'],t) for d in domains for t in d.get('task_ids',[])]
        if len(domains)!=5 or len(selected)!=10 or len(set(selected))!=10 or any(len(d.get('task_ids',[]))!=2 for d in domains):
            raise ValueError('重复套件要求五域各两条，共十条开发任务')
        if not manifest.get('selection'):
            raise ValueError('重复套件缺少预先冻结的选题清单')
        selection = read_json(manifest['selection'])
        if selection.get('selection_before_system_answers') is not True or selected != [(t['domain'],t['task_id']) for t in selection['tasks']]:
            raise ValueError('重复任务与冻结的选题清单不匹配')
    else:
        selection = None
    units = []
    for d in domains:
        for condition in manifest['conditions']:
            spec = dict(manifest['common'], **d['spec'], mode='agent' if condition.endswith('agent') else 'rag')
            if condition.startswith('visual'):
                spec.pop('retrieval_config',None)
            unit = prepare_unit(spec,repeats,d.get('task_ids'),manifest['suite_id'],condition,d['name'],repeats==3,translation_cache)
            for job in unit['jobs']:
                job['unit'] = len(units)
            units.append(unit)
    if not units or any(u['student'] != units[0]['student'] or u['agent'] != units[0]['agent'] for u in units):
        raise ValueError('串行套件的查询模型与生成配置必须一致')
    suite_jobs(units,manifest)  # Validate the execution policy before loading models/providers.
    return units, {'suite':manifest,'selection':selection,'units':[u['identity'] for u in units]}


def provider_factory(units):
    """Load one query model per process; switch validated indexes between units."""
    from .generation import Bailian
    from .agent import ask
    from .query import load_search_session
    from .search import load_index
    from .retrieval import load_retrieval
    model, session, current, provider = None, None, None, None
    def execute(job, budget, attempt):
        nonlocal model, session, current, provider
        unit = units[job.get('unit',0)]
        if current != job.get('unit',0):
            started = time.monotonic()
            if model is None:
                model,index,pages,retrieval = load_search_session(unit['spec']['index'],unit['teacher'],unit['student'],retrieval_config=unit['retrieval_config'])
            else:
                if unit['student'] != units[0]['student']:
                    raise ValueError('同一串行套件只支持一个查询编码配置')
                index,pages = load_index(unit['spec']['index'],unit['teacher'])
                retrieval = load_retrieval(unit['retrieval_config'],pages)
            provider = Bailian(unit['agent'])
            session = index,pages,retrieval
            current = job.get('unit',0)
            loading = time.monotonic()-started
        else:
            loading = 0.
        provider.context = dict(job['key'],attempt=attempt,execution_id=job['execution_id'])
        index,pages,retrieval = session
        options = {'conversation': job['conversation']} if 'conversation' in job else {}
        if 'on_event' in job:
            options['on_event'] = job['on_event']
        result = ask(job['question'],unit['spec']['mode'],model,unit['student'],index,pages,
                     unit['agent'],provider,retrieval,unit['texts'],budget,job.get('translation'), **options)
        result['loading_seconds'] = loading
        return result
    return execute


def can_recover(result, job, config):
    budget = result.get('budget_state',{})
    if not budget.get('budget_snapshot_complete') or not transient(result.get('termination_reason','')):
        return False
    for field in ('requests_used','known_output_tokens','unknown_output_tokens_reserved','searches_used'):
        if type(budget.get(field)) is not int or budget[field] < 0:
            raise ValueError('恢复预算快照损坏，不能推测剩余额度')
    seconds, pages = budget.get('active_seconds_used'), budget.get('distinct_page_ids')
    if (type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0
            or not isinstance(pages,list) or any(not isinstance(p,str) for p in pages)
            or len(set(pages)) != len(pages)):
        raise ValueError('恢复时间或页面预算快照损坏')
    minimum_requests = 1 + int(bool(re.search(r'[\u3400-\u9fff]',job['question'])) and not budget.get('translation_debit'))
    return (budget['requests_used']+minimum_requests <= config['max_requests']
            and budget['known_output_tokens']+budget['unknown_output_tokens_reserved'] < config['max_output_tokens']
            and budget['searches_used'] < config['max_searches']
            and budget['active_seconds_used'] < config['task_timeout'])


def ledger_snapshot(configs, execution_id):
    """Read-only reconciliation includes reservations left by interrupted requests."""
    import json
    import sqlite3
    rows, phase = {}, {}
    for path in sorted({str(c['ledger']) for c in configs if c.get('ledger')}):
        target = Path(path).resolve()
        if not target.exists():
            continue
        with sqlite3.connect(target.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row
            columns={r[1] for r in db.execute('PRAGMA table_info(calls)')}
            if not columns:
                raise ValueError('现有账本缺少 calls 表')
            phase[path]=db.execute('SELECT COALESCE(SUM(cny),0) FROM calls').fetchone()[0]
            if 'context' not in columns: # historical entries still count toward phase cap
                continue
            for row in db.execute('SELECT * FROM calls'):
                context=json.loads(row['context'] or '{}')
                if context.get('execution_id')==execution_id:
                    rows[row['id']]=dict(row)
    return {'ledger_ids':sorted(rows),'known_cost_cny':sum(r['cny'] for r in rows.values() if r['status']=='estimated_from_usage'),
            'unknown_reservation_cny':sum(r['cny'] for r in rows.values() if r['status']=='reserved_unknown_charge'),
            'calls':list(rows.values()),'phase_total_including_other_runs':phase,
            'scope':'estimated usage and conservative reservations; not invoice charges'}


def wait_for_cooldown(last_finished, seconds, started, max_seconds):
    """last_finished is process-local monotonic time, never a persisted timestamp."""
    begin = time.monotonic()
    if not seconds or last_finished is None:
        return 0., None
    delay = max(0.,last_finished+seconds-begin)
    if max_seconds is not None and begin-started+delay >= max_seconds:
        return 0., 'max_seconds'
    while time.monotonic()-begin < delay:
        elapsed = time.monotonic()-begin
        if max_seconds is not None and time.monotonic()-started >= max_seconds:
            return elapsed, 'max_seconds'
        time.sleep(max(0.,min(1.,delay-elapsed)))
    elapsed = time.monotonic()-begin
    return elapsed, None


def latest_service_wait(attempts):
    """Attempts arrive in frozen job/attempt order; wall time cannot reorder calls."""
    for attempt in reversed(attempts):
        result = attempt.get('result', {})
        calls = [c for c in result.get('api_calls', []) if c.get('cost_status') != 'not_sent']
        call = calls[-1] if calls else result.get('failure')
        if call and call.get('cost_status') == 'not_sent':
            continue
        if not call:
            continue
        status = call.get('retry_after_status')
        if status == 'unresolved' or (status is None and 'retry_not_before' in call):
            return 0., 'retry_after_unresolved'
        seconds = call.get('retry_after_seconds', 0.)
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            return 0., 'retry_after_unresolved'
        return seconds, None
    return 0., None


def execute_plan(output, identity, jobs, factory, configs, resume=False, stop_after=None, max_seconds=None,
                 dry_run=False, command_started=None):
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop-after 必须为正整数')
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('max-seconds 必须为正数')
    if len({digest(j['key']) for j in jobs}) != len(jobs):
        raise ValueError('重复运行键')
    policy = identity.get('suite',{})
    cooldown = policy.get('inter_task_cooldown_seconds',0)
    if type(cooldown) not in (int,float) or not math.isfinite(cooldown) or cooldown < 0:
        raise ValueError('任务间冷却时间必须是非负有限秒数')
    if 'job_order' in policy and policy['job_order'] != [j['key'] for j in jobs]:
        raise ValueError('执行顺序与套件身份不一致')
    if dry_run:
        return {'dry_run':True,'planned':len(jobs),'keys':[j['key'] for j in jobs],'stopped':None}
    output = Path(output)
    started = command_started if command_started is not None else time.monotonic()
    signature = digest(identity)
    with output_lock(output):
        if output.exists():
            if not resume:
                raise ValueError('QA 输出目录已存在；接续需 --resume')
            saved = read_json(output/'manifest.json')
            if saved['identity_digest'] != signature:
                raise ValueError('QA 恢复身份不匹配，配置或关键代码变化需使用新目录')
        elif resume:
            raise ValueError('恢复目录不存在')
        else:
            output.mkdir()
            saved={'identity_digest':signature,'identity':identity,'execution_id':uuid.uuid4().hex,
                   'job_keys':[j['key'] for j in jobs]}
            atomic_json(output/'manifest.json',saved)
        # No provider is constructed until a task actually requires execution.
        all_attempts, pending = [], []
        for job in jobs:
            folder = output/'attempts'/digest(job['key'])
            attempts = [read_json(p) for p in sorted(folder.glob('attempt-*.json'))]
            if len(attempts)>2 or any(a.get('key')!=job['key'] or a.get('attempt')!=i+1
                                      or a.get('state') not in ('running','finished') for i,a in enumerate(attempts)):
                raise ValueError('尝试记录损坏、运行键错配或超过两次')
            for a in attempts:
                if a['state']=='finished' and a.get('answer_digest')!=digest(a['result']['answer']):
                    raise ValueError('已保存回答内容与摘要不匹配')
            all_attempts.extend(attempts)
            if not attempts:
                pending.append((job,folder,1,None))
            elif len(attempts)==1 and attempts[0].get('state')=='finished' and transient(attempts[0]['result']['termination_reason']):
                first = attempts[0]
                if can_recover(first['result'],job,configs[job.get('unit',0)]):
                    pending.append((job,folder,2,first))
                else:
                    atomic_json(folder/'recovery.json',{'status':'recovery_skipped_budget_insufficient'})
            elif attempts[-1].get('state')=='running':
                atomic_json(folder/'recovery.json',{'status':'recovery_ineligible_incomplete_budget'})
        segment = output/'segments'/uuid.uuid4().hex
        provenance(segment,{'identity_digest':signature,'resume':resume,'stop_after':stop_after,'max_seconds':max_seconds})
        new, stopped, executor, execution_error = 0, None, None, None
        cooldown_waits = []
        # A restarted process waits the full delay, irrespective of offline wall time.
        last_finished = time.monotonic() if all_attempts else None
        previous_wall = all_attempts[-1].get('finished_at') if all_attempts else None
        service_wait, service_stop = latest_service_wait(all_attempts)
        if any(a['state'] == 'running' for a in all_attempts):
            stopped, pending = 'incomplete_budget', []
        for job,folder,attempt,first in pending:
            if stop_after is not None and new >= stop_after:
                stopped = 'stop_after'; break
            if max_seconds is not None and time.monotonic()-started >= max_seconds:
                stopped = 'max_seconds'; break
            if service_stop:
                stopped = service_stop; break
            waited, stop = wait_for_cooldown(last_finished,max(cooldown,service_wait),started,max_seconds)
            cooldown_waits.append({'key':job['key'],'attempt':attempt,'seconds':waited,
                                   'previous_finished_at':previous_wall,'stopped':stop,
                                   'service_wait_seconds':service_wait, 'basis':'process_monotonic'})
            if stop:
                stopped = stop; break
            service_wait = 0.
            if executor is None:
                executor = factory()
            record = {'key':job['key'],'task':job.get('task',{}),'attempt':attempt,'state':'running',
                      'started_at':time.time(),'cooldown_seconds':waited}
            record.update(restart_gap(record,first))
            path = folder/f'attempt-{attempt:04d}.json'
            if path.exists():
                raise ValueError('拒绝覆盖已有尝试')
            atomic_json(path,record)
            try:
                result = executor(dict(job,execution_id=saved['execution_id']), first['result']['budget_state'] if first else None,attempt)
            except BaseException as exc:
                # Keep running state: consumed task resources are unknown, never guess a resume budget.
                record['process_error']=type(exc).__name__
                atomic_json(path,record)
                all_attempts.append(record)
                stopped, execution_error = 'incomplete_budget', exc
                break
            record.update(state='finished',finished_at=time.time(),result=result,
                          answer_digest=digest(result['answer']))
            last_finished = time.monotonic()
            previous_wall = record['finished_at']
            atomic_json(path,record)
            all_attempts.append(record)
            new += 1
            atomic_json(output/'state.json',{'last_key':job['key'],'last_attempt':attempt,'saved_attempts':len(all_attempts)})
            if pause_reason(result['termination_reason']):
                stopped = result['termination_reason']; break
        from .qa_evaluation import summarize_attempts
        ledger = ledger_snapshot(configs,saved['execution_id'])
        summary = summarize_attempts(all_attempts,len(jobs),planned_keys=saved['job_keys'],ledger_records=ledger['calls'])
        if summary['primary']['incomplete_budget'] or summary['recovery']['incomplete_budget']:
            stopped = stopped or 'incomplete_budget'
        summary['ledger'] = ledger
        summary.update(stopped=stopped, attempts_new=new, segment_seconds=time.monotonic()-started)
        summary.update(cooldown_seconds=sum(w['seconds'] for w in cooldown_waits),cooldown_waits=cooldown_waits)
        atomic_json(output/'summary.json',summary)
        atomic_json(segment/'result.json',summary)
        if execution_error is not None:
            raise execution_error
        return summary
