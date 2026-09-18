"""Task-level execution records; semantic ratings require attributed reviews."""
from pathlib import Path
import time
from .io import read_json, read_rows, write_json, write_rows, provenance
from .agent import ask

RATINGS = ('fact_correctness', 'answer_coverage', 'citation_support', 'insufficiency_handling')


def validate_tasks(path, pages):
    path = Path(path)
    status = read_json(path.parent/'status.json')
    if status.get('qa_reference_frozen') is not True or status.get('evaluation_allowed') is not True:
        raise ValueError('任务参考事实与评分规则尚未冻结，不能进行 QA 验收')
    tasks = read_rows(path)
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


def evaluate_qa(tasks_path, output, mode, model, query_config, index, pages, agent_config, provider,
                retrieval=None, evidence_texts=None, repeats=1):
    output=Path(output)
    if output.exists():
        raise ValueError('QA 结果目录已存在，请使用新目录')
    tasks,status=validate_tasks(tasks_path,pages)
    if repeats not in (1,3) or (repeats==3 and (status.get('split')!='development' or len(tasks)!=10)):
        raise ValueError('仅固定 10 条开发任务可做 3 轮波动测量')
    provenance(output,{'agent_config':agent_config,'query_config':query_config,'retrieval_config':retrieval['config'] if retrieval else None,
                       'tasks':str(tasks_path),'task_status':status,'mode':mode,'repeats':repeats})
    write_rows(output/'tasks.jsonl',tasks)
    rows=[]
    started=time.monotonic()
    for repeat in range(repeats):
        for task in tasks:
            for language in ('en','zh'):
                result=ask(task['question_'+language],mode,model,query_config,index,pages,agent_config,provider,retrieval,evidence_texts)
                result.update(task_id=task['task_id'],language=language,repeat=repeat,category=task.get('category'))
                rows.append(result)
                write_rows(output/'results.jsonl',rows)
                if result['termination_reason'] in ('phase_budget_exhausted','api_http_401','api_http_403','api_http_404','api_transport_or_json_error','missing_usage'):
                    summary=summarize_runs(rows) | {'stopped':result['termination_reason'],'seconds':time.monotonic()-started}
                    write_json(output/'summary.json',summary)
                    return summary
    summary=summarize_runs(rows) | {'stopped':None,'seconds':time.monotonic()-started}
    write_json(output/'summary.json',summary)
    return summary
