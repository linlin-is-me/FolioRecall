"""Export four historical research runs as a read-only, offline HTML showcase.

No model, index, network client or experiment executor is loaded. Only the
selected first attempts, their saved review and actually read page previews are
copied. Run from the repository: python -m scripts.prepare_agent_showcase.
"""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'outputs/extensions/A/multi-agent-research-v2'
DEFAULT_OUTPUT = ROOT / 'outputs/extensions/A/agent-showcase-v1'
CASES = (
    ('cov-01', '人口与就业：责任分工与并行取证', '两个角色分别取得人口预测和就业正文；核验仍遗漏了部分限定。'),
    ('confirm-04', '图表比例：任务依赖与共享原页', '先提取分项，后续角色依赖前项取得总数并计算；两位角色实际收到同一页。'),
    ('confirm-03', '城市政策：核验后的定向补查', '首次核验触发取证返工，增加正文并修订答案；最终仍有覆盖遗漏。'),
    ('cov-11', '信息不足：核验结构失败', '空声明草稿遇到不存在的声明编号；保存技术失败，不计为合法拒答或成功历史。'),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def fields(value, names):
    return {name: copy.deepcopy(value[name]) for name in names.split() if name in value}


def clean_text(value):
    """Final guard for accidentally embedded local paths in allowed text fields."""
    if isinstance(value, str):
        value = re.sub(r'(?i)\b[A-Z]:[\\/][^\s<>\"\']+', '[本机路径已移除]', value)
        return re.sub(r'/(?:mnt/[a-z]|home|root|Users)/[^\s<>\"\']+', '[本机路径已移除]', value)
    if isinstance(value, list):
        return [clean_text(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_text(item) for key, item in value.items()}
    return value


def answer_view(answer):
    value = fields(answer, 'insufficient_evidence unresolved')
    value['claims'] = [fields(row, 'text evidence_page_ids') for row in answer.get('claims', [])]
    return value


def task_view(task):
    return fields(task, 'id name question query scope requirement_ids depends_on')


def page_view(page):
    return fields(page, 'page_id doc_id document_name page_number text status received_by analysis_versions')


def worker_view(worker):
    value = fields(worker, 'role dependencies step done failed blocked read_ids findings')
    value['task'] = task_view(worker.get('task', {}))
    value['candidate_facts'] = [fields(row, 'text page_id page_ids subject period conditions unit task_id')
                                for row in worker.get('candidate_facts', []) if isinstance(row, dict)]
    return value


def verification_view(value, version=None):
    result = fields(value, 'version')
    if version is not None:
        result['version'] = version
    result['claims'] = [fields(row, 'id supported reason') for row in value.get('claims', [])]
    result['issues'] = [fields(row, 'kind type requirement_id claim_id task_id owner detail reason needs_retrieval query')
                        for row in value.get('issues', [])]
    return result


def draft_view(value):
    result = fields(value, 'version page_ids structure_valid')
    result['answer'] = answer_view(value.get('answer', {}))
    result['numbered_claims'] = [fields(row, 'id text evidence_page_ids') for row in value.get('numbered_claims', [])]
    result['coverage'] = [fields(row, 'requirement_id covered note') for row in value.get('coverage', [])]
    result['calculations'] = [fields(row, 'description expression operands operation a b result expected valid error unit '
                                         'evidence_page_ids computed_result arithmetic_ok scope')
                              for row in value.get('calculations', []) if isinstance(row, dict)]
    return result


def event_view(value):
    result = fields(value, 'event status state elapsed_seconds sequence role node phase version reason '
                          'termination_reason model_requests searches cost_cny page_id document_name '
                          'page_number query task_id depends_on read_ids kind detail triggered retrieval')
    if 'tasks' in value:
        result['tasks'] = [task_view(task) for task in value['tasks']]
    if 'requirements' in value:
        result['requirements'] = [fields(row, 'id text') for row in value['requirements']]
    if 'pages' in value:
        result['pages'] = [page_view(page) for page in value['pages']]
    if 'issues' in value:
        result['issues'] = verification_view(value)['issues']
    return result


def saved_verifications(attempt_path, result):
    """Bind to recorded version filenames; never infer versions from list order."""
    paths = sorted(attempt_path.parent.joinpath('graph').glob('verification-v*.json'))
    if paths:
        return [verification_view(read_json(path), int(re.fullmatch(r'verification-v(\d+)\.json', path.name)[1]))
                for path in paths]
    values = result.get('research', {}).get('verifications', [])
    completed = [event for event in result.get('progress', [])
                 if event.get('event') == 'research_verify' and event.get('status') == 'completed'
                 and isinstance(event.get('version'), int)]
    output = []
    for value in values:
        if isinstance(value.get('version'), int):
            output.append(verification_view(value))
        else:
            matches = [event for event in completed if event.get('issues', []) == value.get('issues', [])]
            output.append(verification_view(value, matches[0]['version'] if len(matches) == 1 else None))
    return output


def sanitize_result(result, attempt_path):
    value = fields(result, 'question retrieval_query mode answer_language completed termination_reason '
                           'actual_model_requests actual_searches distinct_pages output_tokens '
                           'estimated_cost_cny cost_complete timing loading_seconds context_turns '
                           'translation_status experiment_arm experiment_stage experiment_stratum')
    value['answer'] = answer_view(result.get('answer', {}))
    value['evidence'] = [page_view(page) for page in result.get('evidence', [])]
    value['progress'] = [event_view(event) for event in result.get('progress', [])]
    research = result.get('research', {})
    value['research'] = fields(research, 'version stage verification_status failure_stage')
    value['research'].update(
        requirements=[fields(row, 'id text') for row in research.get('requirements', [])],
        tasks=[task_view(task) for task in research.get('tasks', [])],
        workers={key: worker_view(worker) for key, worker in research.get('workers', {}).items()},
        drafts=[draft_view(draft) for draft in research.get('drafts', [])],
        verifications=saved_verifications(attempt_path, result),
        pages=[page_view(page) for page in research.get('pages', [])])
    if 'rework' in research:
        value['research']['rework'] = fields(research['rework'], 'triggered retrieval kind task_id worker role reason detail question query')
    # Old records keep final worker state in graph.workers as well.
    value['graph'] = {'workers': {key: worker_view(worker)
                                 for key, worker in result.get('graph', {}).get('workers', {}).items()}}
    return clean_text(value)


def select_attempts(source):
    selected = {}
    wanted = {case[0] for case in CASES}
    for path in sorted((source / 'main/attempts').glob('*/attempt-0001.json')):
        attempt = read_json(path)
        key = attempt.get('key', {})
        name = key.get('task_id')
        if name not in wanted or not key.get('suite', '').endswith('-M2'):
            continue
        if name in selected:
            raise ValueError(f'首次M2记录不唯一：{name}')
        if attempt.get('attempt') != 1 or attempt.get('state') != 'finished':
            raise ValueError(f'需要已保存的首次终态：{name}')
        selected[name] = path, attempt
    missing = wanted - selected.keys()
    if missing:
        raise ValueError('缺少指定M2首次终态：' + ', '.join(sorted(missing)))
    return selected


def local_preview(value, parent):
    # Frozen experiments ran in WSL; resolve only their explicit preview paths.
    if os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':' + value[6:]
    path = Path(value)
    return path if path.is_absolute() else parent / path


def load_pages(repository):
    paths = [repository/'outputs/releases/v0.1.0-final-candidate/demo-bundle/pages.jsonl',
             repository/'outputs/extensions/A/indexes/jrc-original/pages.jsonl']
    pages = {}
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            pages[row['page_id']] = dict(row, preview_file=local_preview(row['preview'], path.parent))
    return pages


def review_view(review, attempt):
    if (review.get('answer_digest') != attempt['answer_digest'] or review.get('attempt') != 1
            or review.get('key') != attempt['key']):
        raise ValueError('已有助手评审与首次答案不匹配')
    value = fields(review, 'reviewer_type human answer_digest fact_correctness citation_support '
                          'answer_coverage insufficiency_handling insufficiency_reason language_ok diagnosis')
    value['claim_reviews'] = [fields(row, 'claim_index fact citation reason') for row in review.get('claim_reviews', [])]
    value['required_fact_reviews'] = [fields(row, 'fact_index score reason') for row in review.get('required_fact_reviews', [])]
    return clean_text(value)


def escape(value):
    return html.escape(str(value), quote=True)


def beijing_time(value):
    """Keep raw timestamps in JSON; label the display timezone explicitly."""
    zone = timezone(timedelta(hours=8))
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, zone).strftime('%Y-%m-%d %H:%M:%S 北京时间')
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if timestamp.tzinfo is not None:
                return timestamp.astimezone(zone).strftime('%Y-%m-%d %H:%M:%S 北京时间')
        except ValueError:
            pass
    return str(value)


def answer_html(result):
    answer = result.get('answer', {})
    claims = []
    for claim in answer.get('claims', []):
        links = ' '.join(f'<a class="chip" href="#page-{quote(page_id, safe="")}">{escape(page_id)}</a>'
                         for page_id in claim.get('evidence_page_ids', []))
        claims.append(f'<li>{escape(claim.get("text", ""))}<div class="citation-links">{links}</div></li>')
    state = '成功终态' if result.get('completed') else '技术失败，未形成有效终态回答'
    out = f'<section class="card"><h2>保存的回答</h2><p class="muted">{state} · {escape(result.get("termination_reason", ""))}</p>'
    out += '<ol>' + ''.join(claims) + '</ol>' if claims else '<p>没有事实声明。</p>'
    unresolved = answer.get('unresolved', [])
    if unresolved:
        out += '<h3>未解决事项</h3><ul>' + ''.join(f'<li>{escape(text)}</li>' for text in unresolved) + '</ul>'
    return out + '</section>'


def render_case(case, all_cases):
    from foliorecall.research_display import research_html, research_css
    result, review = case['result'], case['assistant_review']
    navigation = '<a href="../index.html">全部案例</a> · ' + ' · '.join(
        f'<a href="{name}.html">{escape(title.split("：")[0])}</a>' for name, title, _ in all_cases)
    preview = []
    for page in result['evidence']:
        preview.append(f'<article id="page-{escape(page["page_id"])}" class="page-card">'
            f'<h3>{escape(page["document_name"])} · 物理第 {escape(page["page_number"])} 页</h3>'
            f'<p class="muted">{escape(page["page_id"])} · <a href="{escape(page["source"])}">公开来源</a></p>'
            f'<a href="../{escape(page["preview"])}" target="_blank" rel="noopener"><img loading="lazy" '
            f'src="../{escape(page["preview"])}" alt="{escape(page["document_name"])} 物理第 {escape(page["page_number"])} 页"></a></article>')
    diagnostics = f'<section class="card diagnosis"><h2>已有助手复核</h2><p>{escape(review.get("diagnosis", ""))}</p>'
    diagnostics += f'<p>{escape(review.get("insufficiency_reason", ""))}</p><p class="muted">assistant · human=not_checked。内部核验和助手复核都不是人工确认。</p></section>'
    body = (f'<nav>{navigation}</nav><header class="showcase-header"><p class="eyebrow">FolioRecall · 历史真实调用</p>'
        f'<h1>{escape(case["title"])}</h1><p>{escape(case["description"])}</p>'
        f'<p class="notice">只读展示；不执行、重试或恢复任务。以下耗时属于历史调用，当前打开页面不会生成回答。</p>'
        f'<p><a href="{case["task_id"]}.json" download>下载脱敏JSON</a> · <a href="../SOURCES.md">素材来源</a></p></header>'
        f'<section class="card"><h2>原问题</h2><p>{escape(result["question"])}</p>'
        f'<p class="muted">{escape(beijing_time(case["finished_at"]))} · 首次尝试 · M2 · 活动时间 '
        f'{result.get("timing", {}).get("task_seconds", 0):.2f} 秒 · '
        f'{result.get("actual_model_requests", 0)} 次请求 / {result.get("actual_searches", 0)} 次搜索</p></section>'
        + research_html(result=result, page_prefix='page-') + answer_html(result) + diagnostics
        + '<section class="card"><h2>本轮实际读取的原页</h2><p class="muted">点击图像打开原尺寸；页码为PDF物理页码。未附助手额外复核页。</p>'
        + ''.join(preview) + '</section>')
    return document(case['title'], body, research_css())


STYLE = '''
*{box-sizing:border-box}body{margin:0;background:#f5f7f6;color:#243a32;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;overflow-wrap:anywhere}
main{max-width:1120px;margin:auto;padding:28px 24px 64px}a{color:#176b50;text-underline-offset:3px}h1{font-size:30px;line-height:1.3}h2{font-size:21px}h3{font-size:17px}nav{font-size:14px}.showcase-header{margin:28px 0}.eyebrow{color:#417060;font-weight:650;letter-spacing:.08em}.muted{color:#62766c;font-size:14px}.card{padding:22px 26px;background:white;border:1px solid #dce6e0;border-radius:14px;margin:18px 0}.notice{padding:12px 16px;background:#eaf2ed;border-radius:8px}.citation-links{margin:5px 0 13px}.chip{display:inline-block;background:#eef4f0;padding:3px 8px;border-radius:5px;font-size:13px}.page-card{padding:18px 0;scroll-margin-top:12px}.page-card img{display:block;width:100%;max-width:750px;height:auto;border:1px solid #dce6e0}.case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}.case-grid .card{margin:0}.diagnosis{border-left:4px solid #ad8a4b}li{margin-bottom:10px}code{overflow-wrap:anywhere}@media(max-width:600px){main{padding:18px 12px}.card{padding:16px}h1{font-size:25px}}
'''


def document(title, body, extra_css=''):
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{escape(title)}</title><style>{STYLE}\n{extra_css}</style></head><body><main>{body}</main></body></html>')


def prepare(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, repository=ROOT):
    source, output, repository = Path(source), Path(output), Path(repository)
    if output.resolve() == source.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('展示目录不能覆盖或位于历史源记录目录内')
    attempts, pages = select_attempts(source), load_pages(repository)
    closeout = source/'closeout-summary.json'
    source_record = read_json(closeout) if closeout.exists() else {}
    demo_sources = (repository/'outputs/releases/v0.1.0-final-candidate/demo-bundle/SOURCES.md').read_text(encoding='utf-8')
    cases = []
    images = {}
    for name, title, description in CASES:
        path, attempt = attempts[name]
        result = sanitize_result(attempt['result'], path)
        for page in result['evidence']:
            original = pages[page['page_id']]
            if not original['preview_file'].is_file():
                raise ValueError(f'预览不存在：{page["page_id"]}')
            source_url = original.get('source', '')
            if not source_url.startswith('https://'):
                raise ValueError('展示页需要已有公开来源URL')
            asset = 'pages/' + re.sub(r'[^a-zA-Z0-9_.-]', '-', page['page_id']) + original['preview_file'].suffix.lower()
            page.update(source=source_url, preview=asset)
            images[asset] = original['preview_file']
        review_path = source/'scoring'/f'{path.parent.name}-attempt-0001.json'
        case = {'task_id': name, 'title': title, 'description': description,
                'record_type': 'historical_real_first_attempt', 'read_only': True,
                'attempt': 1, 'answer_digest': attempt['answer_digest'],
                'started_at': attempt['started_at'], 'finished_at': attempt['finished_at'],
                'source_key': fields(attempt['key'], 'suite condition domain task_id language repeat'),
                'result': result, 'assistant_review': review_view(read_json(review_path), attempt)}
        cases.append(case)
    # Render all records before creating an output folder, so data errors do not
    # leave a misleading partially complete showcase.
    rendered = {case['task_id']: render_case(case, CASES) for case in cases}
    output.mkdir(parents=True, exist_ok=True)
    (output/'cases').mkdir(exist_ok=True)
    (output/'pages').mkdir(exist_ok=True)
    for asset, source_file in images.items():
        shutil.copy2(source_file, output/asset)
    for case in cases:
        name = case['task_id']
        write_json(output/'cases'/f'{name}.json', case)
        (output/'cases'/f'{name}.html').write_text(rendered[name], encoding='utf-8')
    cards = ''.join(f'<article class="card"><h2><a href="cases/{case["task_id"]}.html">{escape(case["title"])}</a></h2>'
                   f'<p>{escape(case["description"])}</p></article>' for case in cases)
    (output/'index.html').write_text(document('FolioRecall 多Agent离线案例',
        '<header class="showcase-header"><p class="eyebrow">FolioRecall · 多Agent工程展示</p><h1>四份真实研究记录</h1>'
        '<p>查看责任分工、依赖执行、共享证据、核验返工及失败保存。</p>'
        '<p class="notice">历史真实调用的只读展示。直接打开即可使用，无需API、模型或索引，不会重新生成回答。</p>'
        '</header><div class="case-grid">'+cards+'</div><p class="muted">保留原答案、失败和助手诊断；'
        'human=not_checked。四例展示工程流程，不代表整体质量优于单Agent。</p><p><a href="SOURCES.md">素材来源与许可边界</a></p>'), encoding='utf-8')
    sources = ('# 离线案例的记录与页面来源\n\n只读摘录2026年10月15日既有M2首次尝试，'
        f'不包含原始API日志、账号信息、检查点、完整PDF、索引或模型。仅复制四例实际读取的{len(images)}张页面预览；'
        '未改写回答或重新评测。JSON中的answer_digest沿用已有记录，不新增哈希清单。\n\n'
        '## 42页示例既有素材说明\n\n'+demo_sources+'\n## JRC案例的选定原页\n\n'
        '- JRC141212：物理第7、68、70页。https://publications.jrc.ec.europa.eu/repository/handle/JRC141212\n'
        '- JRC143054：物理第46页。https://publications.jrc.ec.europa.eu/repository/handle/JRC143054\n\n'
        '© European Union。沿用既有资料清单：正文采用CC BY 4.0，单独署名的第三方素材除外；'
        '许可依据为各PDF物理第2页。本次复用已有PNG，仅选取上述页面，保留物理页码和来源。'
        '代码Apache-2.0许可不覆盖文档素材；本地候选不代表完整JRC库获得重新分发许可。\n')
    (output/'SOURCES.md').write_text(sources, encoding='utf-8')
    summary = {'cases': [case['task_id'] for case in cases], 'case_count': len(cases),
               'preview_count': len(images), 'new_model_calls': 0, 'historical_records_modified': False,
               'requires_network': False, 'human': 'not_checked',
               'render_kind': 'offline_display_of_historical_records',
               'render_note': '当前代码仅重绘保存的结果，不是历史生成代码版本。'}
    if source_record.get('runtime_commit'):
        summary['source_runtime_commit'] = source_record['runtime_commit']
        summary['source_commit_record'] = 'closeout-summary.json:runtime_commit'
    write_json(output/'manifest.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False, indent=2))
