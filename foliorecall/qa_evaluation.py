"""Task-level execution records; semantic ratings require attributed reviews."""
from pathlib import Path
import math
from .io import read_json, read_rows
from .retrieval import page_identity

RATINGS = ('fact_correctness', 'answer_coverage', 'citation_support', 'insufficiency_handling')


def validate_tasks(path, pages):
    path = Path(path)
    status = read_json(path.parent/'status.json')
    if status.get('qa_reference_frozen') is not True or status.get('evaluation_allowed') is not True:
        raise ValueError('任务参考事实与评分规则尚未冻结，不能进行 QA 验收')
    if status.get('corpus_pages') != page_identity(pages):
        raise ValueError('QA 任务绑定的完整页面库不匹配；不同领域的相同页面 ID 不能混用')
    tasks = read_rows(path)
    if status.get('schema_version') == 2:
        from .qa_storage import digest
        if digest(tasks) != status.get('tasks_digest'):
            raise ValueError('冻结任务内容发生变化，需重新核验后使用新版本')
        source = read_json(status['source_record'])
        if any(source.get(k)!=status.get(k) for k in ('dataset','revision')) or source.get('task',{}).get('name') != status.get('task'):
            raise ValueError('冻结任务语料领域或 revision 不匹配')
        if any(t.get('task') != status['task'] for t in tasks):
            raise ValueError('任务领域不匹配')
    ids = [t['task_id'] for t in tasks]
    candidates = {p['page_id'] for p in pages}
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('任务 ID 为空或重复')
    for task in tasks:
        if any(not isinstance(task.get('question_'+language),str) or not task['question_'+language].strip() for language in ('en','zh')):
            raise ValueError('每条逻辑任务需要中英文问题')
        if type(task.get('answerable')) is not bool or not isinstance(task.get('reference_facts'),list) or not isinstance(task.get('evidence_page_ids'),list):
            raise ValueError('缺少任务答案状态、参考事实或证据页')
        if not set(task['evidence_page_ids']).issubset(candidates):
            raise ValueError('任务证据页不在完整页面库内')
        if task['answerable'] and (not task['reference_facts'] or not task['evidence_page_ids']):
            raise ValueError('可回答任务缺少事实或证据')
        if not task.get('rubric') or not task.get('review'):
            raise ValueError('缺少评分规则或核验来源')
    return tasks, status


def summarize_runs(rows, reviews=None):
    if not rows:
        raise ValueError('没有 QA 结果')
    keys=[(r['task_id'],r['language'],r['repeat']) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('重复任务/语言/轮次')
    by_language={}
    for language in sorted({r['language'] for r in rows}):
        selected=[r for r in rows if r['language']==language]
        by_language[language]={'runs':len(selected), 'completion_rate':sum(r['completed'] for r in selected)/len(selected),
            'mean_task_seconds':sum(r['timing']['task_seconds'] for r in selected)/len(selected),
            'estimated_known_cost_cny':sum(r['estimated_cost_cny'] for r in selected),
            'cost_complete':all(r['cost_complete'] for r in selected),
            'mean_model_requests':sum(r['model_requests'] for r in selected)/len(selected)}
    attributed={}
    seen=set()
    for review in reviews or []:
        key=(review['task_id'],review['language'],review['repeat'])
        actor=review['reviewer_type']
        if key not in keys or actor not in ('human','assistant','automatic') or (key,actor) in seen:
            raise ValueError('评审必须对应运行且来源明确，不能重复')
        seen.add((key,actor))
        for metric in RATINGS:
            value=review.get(metric)
            if not isinstance(value,(int,float)) or not 0<=value<=1:
                raise ValueError('评分应在 0–1 之间')
        attributed.setdefault(actor,[]).append(review)
    return {'logical_tasks':len({r['task_id'] for r in rows}), 'paired_languages_not_independent':True,
        'by_language':by_language, 'semantic_reviews':{actor:{'count':len(values),
            **{metric:sum(r[metric] for r in values)/len(values) for metric in RATINGS}} for actor,values in attributed.items()},
        'semantic_quality_status':'reviewed_subset_only' if reviews else 'pending_review',
        'scope':'completion and valid citation IDs do not establish answer correctness or citation support'}


def summarize_attempts(attempts, planned, reviews=None, planned_keys=None, ledger_records=None):
    """First attempts remain the primary sample; recovery is a separate stratum."""
    from .qa_storage import digest, restart_gap
    keys = [(digest(a['key']),a['attempt']) for a in attempts]
    if len(keys) != len(set(keys)):
        raise ValueError('重复运行键及尝试号')
    if planned_keys is not None:
        allowed = {digest(k) for k in planned_keys}
        if len(allowed) != planned or any(k not in allowed for k,a in keys):
            raise ValueError('尝试记录不属于固定运行清单')
    for a in attempts:
        if a['attempt'] not in (1,2) or a.get('state') not in ('running','finished'):
            raise ValueError('无效尝试号或状态')
        if a['state']=='finished' and a.get('answer_digest') != digest(a['result']['answer']):
            raise ValueError('回答内容与保存的摘要不匹配')
        if a['attempt']==2 and (digest(a['key']),1) not in keys:
            raise ValueError('补跑记录缺少首次尝试')
    lookup = {key:a for key,a in zip(keys,attempts)}
    # Recompute from preserved timestamps, including records written by older code.
    # A historical negative or missing gap must not become a zero-duration wait.
    gaps = {key:restart_gap(a,lookup.get((key[0],1))) for key,a in lookup.items()}
    seen, accepted = set(), []
    for review in reviews or []:
        key = (digest(review['key']),review['attempt'])
        actor = review['reviewer_type']
        if key not in lookup or actor not in ('human','assistant','automatic') or (key,actor) in seen:
            raise ValueError('评分运行键、尝试或来源不匹配/重复')
        row = lookup[key]
        if row.get('state') != 'finished' or review.get('answer_digest') != row.get('answer_digest'):
            raise ValueError('评分回答内容不匹配')
        for metric in RATINGS:
            value = review.get(metric)
            if value is None:
                if not review.get('not_applicable',{}).get(metric):
                    raise ValueError('不适用评分需要说明原因；未评审请勿导入记录')
            elif type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('评分必须为有限的 0–1 数值')
        task = row.get('task',{})
        if task.get('reference_facts') and review['answer_coverage'] is None:
            raise ValueError('存在必需事实时，覆盖度不能设为不适用')
        if task.get('reference_facts') and not row['result']['answer']['claims'] and review['answer_coverage'] != 0:
            raise ValueError('存在必需事实时，空答案覆盖度必须为零')
        if task.get('answerability')=='insufficient' and not task.get('reference_facts'):
            if review['answer_coverage'] is not None:
                raise ValueError('完全无答案任务没有必需事实，覆盖度应为 null')
            if not row['result']['answer']['claims'] and any(review[m] is not None for m in ('fact_correctness','citation_support')):
                raise ValueError('没有事实声明的拒答不自动获得事实或引用满分')
        seen.add((key,actor)); accepted.append(review)
    first = [a for a in attempts if a['attempt']==1]
    recovery = [a for a in attempts if a['attempt']==2]
    def aggregate(rows):
        finished = [a for a in rows if a.get('state')=='finished']
        success = sum(a['result']['completed'] for a in finished)
        intervals = [gaps[(digest(a['key']),a['attempt'])] for a in rows if a['attempt']==2]
        known_gap = sum(g['gap_seconds'] for g in intervals if g['gap_seconds'] is not None)
        unknown_gaps = sum(g['gap_seconds'] is None for g in intervals)
        return {'attempted':len(rows),'finished':len(finished),'completed':success,
                'completion_rate':success/len(rows) if rows else None,
                'incomplete_budget':len(rows)-len(finished),
                'mean_task_seconds':sum(a['result']['timing']['task_seconds'] for a in finished)/len(finished) if finished else None,
                'total_actual_task_seconds':sum(a['result']['timing']['task_seconds'] for a in finished),
                'manual_restart_gap_seconds':None if unknown_gaps else known_gap,
                'manual_restart_gap_known_seconds':known_gap,
                'manual_restart_gap_unknown_count':unknown_gaps,
                'manual_restart_gap_status_counts':{status:sum(g['gap_status']==status for g in intervals)
                    for status in sorted({g['gap_status'] for g in intervals})},
                'known_cost_cny':sum(a['result'].get('estimated_cost_cny',0) for a in finished)}
    groups = {}
    for a in first:
        k = a['key']
        name = '/'.join(k[x] for x in ('condition','domain','language'))
        groups.setdefault(name,[]).append(a)
    planned_groups = {}
    for k in planned_keys or []:
        name='/'.join(k[x] for x in ('condition','domain','language'))
        planned_groups[name]=planned_groups.get(name,0)+1
        groups.setdefault(name,[])
    review_groups = {}
    for r in accepted:
        k = r['key']
        name = '/'.join([k['condition'],k['domain'],k['language'],r['reviewer_type'],str(r['attempt'])])
        review_groups.setdefault(name,[]).append(r)
    semantic = {}
    for name, rows in review_groups.items():
        k = rows[0]['key']; attempt = rows[0]['attempt']
        denominator = sum(a['attempt']==attempt and all(a['key'][f]==k[f] for f in ('condition','domain','language')) for a in attempts)
        semantic[name] = {'reviewed':len(rows),'attempted':denominator,'coverage':len(rows)/denominator,
                         'metrics':{m:{'count':sum(r[m] is not None for r in rows),
                             'mean':sum(r[m] for r in rows if r[m] is not None)/sum(r[m] is not None for r in rows) if any(r[m] is not None for r in rows) else None} for m in RATINGS}}
    calls = {}
    for a in attempts:
        for i,c in enumerate(a.get('result',{}).get('api_calls',[])):
            calls[c.get('ledger_id') or f'{digest(a["key"])}:{a["attempt"]}:{i}'] = c
    for c in ledger_records or []:
        calls[c['id']]={'ledger_id':c['id'], 'cost_cny':c['cny'] if c['status']=='estimated_from_usage' else 0,
                        'reservation_cny':c['cny'] if c['status']=='reserved_unknown_charge' else 0,
                        'cost_status':c['status']}
    translations = {}
    for a in attempts:
        historical=a.get('result',{}).get('translation_history')
        if historical:
            translations[digest(a['key'])]=historical['cost_cny']/3
    pairs = {}
    for a in first:
        key={k:v for k,v in a['key'].items() if k!='language'}
        entry=pairs.setdefault(digest(key),{'key':key,'languages':{}})
        entry['languages'][a['key']['language']]={'state':a['state'],'completed':a.get('result',{}).get('completed'),
            'answer_digest':a.get('answer_digest')}
    return {'schema_version':2,'planned':planned,'pending':planned-len(first),
            'logical_tasks':len({(a['key']['domain'],a['key']['task_id']) for a in first}),
            'paired_languages_not_independent':True,'primary':aggregate(first),'recovery':aggregate(recovery),
            'by_condition_domain_language':{k:aggregate(v)|{'planned':planned_groups.get(k),
                  'pending':planned_groups[k]-len(v) if k in planned_groups else None} for k,v in groups.items()},
            'primary_language_pairs':list(pairs.values()),
            'recovery_intervals':[{'key':a['key'],'attempt':2,**gaps[(digest(a['key']),2)]}
                                  for a in recovery],
            'semantic_reviews':semantic,'semantic_quality_status':'reviewed_subset_only' if accepted else 'pending_review',
            'all_attempts_known_cost_cny':sum(c.get('cost_cny',0) for c in calls.values()),
            'unknown_reservation_cny':sum(c.get('reservation_cny',0) for c in calls.values() if c.get('cost_status')=='reserved_unknown_charge'),
            'translation_allocated_cost_cny':sum(translations.values()),
            'translation_cost_scope':'historical conversion cost / 3 per logical condition, not charged again on recovery',
            'ledger_scope':'API reservations after process crashes must also be read from the durable ledger; results alone are not the phase balance'}
