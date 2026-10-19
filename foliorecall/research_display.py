"""Read-only collaboration views shared by Gradio and saved-case HTML."""
import html
import re
from urllib.parse import quote

from .demo_style import document_label


STATES = {'waiting': '等待执行', 'running': '执行中', 'completed': '取证结束',
          'failed': '执行失败', 'blocked': '前置失败，已阻塞', 'unknown': '状态未记录'}
REASONS = {'evidence_finished': '模型结束取证', 'decision_limit': '达到决策上限',
           'worker_request_limit': '达到角色请求上限', 'request_limit': '达到请求上限',
           'output_token_limit': '达到输出上限', 'blocked': '前置任务未成功',
           'ready': '等待调度', 'dependency_pending': '前置任务尚未完成',
           'decision_started': '正在选择下一步', 'rework_requested': '收到定向返工要求'}
STAGES = [('prepare', '问题准备'), ('plan', '规划分工'), ('evidence', '责任取证'),
          ('analysis', '证据分析'), ('verify', '原页核验'), ('rework', '定向返工'), ('end', '保存终态')]


def esc(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def is_research(result, events):
    return bool(result and 'research' in result) or any(
        e.get('event', '').startswith('research_') or e.get('node', '').startswith('research-')
        or (e.get('event') == 'coordinate' and 'requirements' in e) for e in events)


def _stage(event):
    name = event.get('event')
    return {'query_rewrite': 'prepare', 'coordinate': 'plan', 'research_task': 'evidence',
            'search_pages': 'evidence', 'read_page': 'evidence', 'research_analysis': 'analysis',
            'research_verify': 'verify', 'research_rework': 'rework', 'terminal': 'end'}.get(name)


def view_state(result=None, events=None):
    """Derive display state without updating the result or making execution decisions."""
    events = list(events if events is not None else (result or {}).get('progress', []))
    if not is_research(result, events):
        return None
    research = (result or {}).get('research', {})
    tasks, requirements = [], []
    states, phases, seen, received = {}, {}, set(), {}
    current, terminal, failed = 'prepare', False, False
    for event in events:
        if event.get('event') == 'coordinate' and event.get('tasks'):
            tasks, requirements = event['tasks'], event.get('requirements', [])
        if event.get('event') == 'research_task':
            key = event.get('task_id')
            states[key] = dict(event)
            phases[(key, event.get('phase', 'main'))] = dict(event)
        stage = _stage(event)
        if stage:
            current = stage
            seen.add(stage)
        if event.get('event') == 'read_page' and event.get('status') in ('completed', 'read', 'already_read'):
            pid = event.get('page_id')
            if pid:
                page = received.setdefault(pid, {'page_id': pid, 'document_name': event.get('document_name', pid),
                                                 'page_number': event.get('page_number'), 'received_by': []})
                role = event.get('role')
                if role and role not in page['received_by']:
                    page['received_by'].append(role)
        if event.get('event') == 'terminal':
            terminal, failed = True, event.get('status') not in ('completed', 'finished')
    if result is not None:
        tasks = research.get('tasks', tasks)
        requirements = research.get('requirements', requirements)
        terminal, failed, current = True, not result.get('completed', False), 'end'
        for worker in research.get('workers', {}).values():
            tid = worker.get('task', {}).get('id')
            if tid in states:
                continue
            state = ('blocked' if worker.get('blocked') else 'failed' if worker.get('failed') else
                     'completed' if worker.get('done') else 'unknown')
            states[tid] = {'state': state, 'reason': worker.get('failed', ''), 'phase': 'main'}
        if tasks:
            seen.add('plan')
        if research.get('pages'):
            seen.add('evidence')
        if research.get('drafts'):
            seen.add('analysis')
        if research.get('verifications'):
            seen.add('verify')
        if research.get('rework', {}).get('triggered'):
            seen.add('rework')
    return {'tasks': tasks, 'requirements': requirements, 'states': states, 'phases': phases,
            'seen': seen, 'current': current, 'terminal': terminal, 'failed': failed,
            'pages': research.get('pages', list(received.values())), 'events': events, 'research': research}


def _version(check, events, position, checks):
    if isinstance(check.get('version'), int):
        return check['version']
    versions = {int(match.group(1)) for claim in check.get('claims', [])
                if (match := re.fullmatch(r'v(\d+)-c\d+', claim.get('id', '')))}
    if len(versions) == 1:
        return versions.pop()
    completed = list(dict.fromkeys(e.get('version') for e in events
        if e.get('event') == 'research_verify' and e.get('status') == 'completed'
        and isinstance(e.get('version'), int)))
    return completed[position] if len(completed) == len(checks) else None


def _page_link(page, prefix):
    pid = page['page_id']
    name = page.get('document_name') or pid
    return (f'<a class="fr-evidence-link" href="#{esc(prefix)}{quote(pid, safe="")}" title="{esc(name)}">'
            f'{esc(document_label(name))} · 物理第{esc(page.get("page_number") or "?")}页</a>')


def _calculations(draft):
    rows = []
    signs = {'add': '+', 'subtract': '−', 'multiply': '×', 'divide': '÷'}
    for calc in draft.get('calculations', []):
        verdict = '算术复算一致' if calc.get('arithmetic_ok') is True else '算术复算不一致' if calc.get('arithmetic_ok') is False else '复算未记录'
        rows.append(f'<p class="fr-research-note">{esc(calc.get("a"))} '
                    f'{esc(signs.get(calc.get("operation"), calc.get("operation")))} {esc(calc.get("b"))}'
                    f'；草稿值 {esc(calc.get("result"))}，复算值 {esc(calc.get("computed_result"))}'
                    f' · {verdict} · {esc(calc.get("unit"))}</p>')
    return ('<div class="fr-calculations"><strong>计算记录</strong>' + ''.join(rows) +
            '<p class="fr-research-note">仅核对算术；操作数、分母和单位仍需查看原页。</p></div>') if rows else ''


def research_html(result=None, events=None, page_prefix='qa-page-'):
    state = view_state(result, events)
    if state is None:
        return ''
    research = state['research']
    status = '任务未完成' if state['failed'] else '执行已结束' if state['terminal'] else '正在协作'
    chunks = ['<section class="fr-research" aria-label="研究协作过程">',
              f'<header class="fr-research-head"><div><span class="fr-eyebrow">协作过程</span>'
              f'<h3>分工、证据与核验</h3></div><span class="fr-state {"failed" if state["failed"] else ""}">{status}</span></header>',
              '<ol class="fr-stages" aria-label="执行阶段">']
    for key, label in STAGES:
        active = key == state['current']
        style = 'failed' if active and state['failed'] else 'active' if active else 'seen' if key in state['seen'] else ''
        suffix = ' · 可选' if key == 'rework' and key not in state['seen'] else ''
        chunks.append(f'<li class="{style}">{label}{suffix}</li>')
    chunks.append('</ol>')
    if state['failed']:
        chunks.append(f'<p class="fr-research-warning">失败阶段：{esc(research.get("failure_stage") or "未记录")}'
                      f' · {esc((result or {}).get("termination_reason", "执行异常"))}。草稿不代表完成答案。</p>')
    elif not state['terminal']:
        chunks.append('<p class="fr-research-note">状态来自实际执行事件；取证结束不代表所有要求已满足。</p>')
    tasks = state['tasks']
    task_names = {t['id']: t.get('name') or t['id'] for t in tasks}
    roles = {f'worker-{i+1}': task_names[t['id']] for i, t in enumerate(tasks)}
    reqs = {r['id']: r.get('text', r['id']) for r in state['requirements']}
    if tasks:
        chunks.append('<div class="fr-task-grid">')
        for i, task in enumerate(tasks):
            tid = task['id']
            item = state['states'].get(tid, {})
            code = item.get('state', 'unknown')
            reason = item.get('reason') or ''
            deps = '、'.join(task_names.get(d, d) for d in task.get('depends_on', []))
            chunks.append(f'<article class="fr-task"><header><span class="fr-task-number">{i+1:02}</span>'
                          f'<h4>{esc(task_names[tid])}</h4><span class="fr-state {esc(code)}">{STATES.get(code, "状态未记录")}</span></header>'
                          f'<p>{esc(task.get("question", ""))}</p><p class="fr-task-dependency">'
                          f'{"等待完成：" + esc(deps) if deps and code == "waiting" else "前置任务：" + esc(deps) if deps else "独立责任 · 可并行调度"}</p>')
            if item.get('phase') == 'rework':
                chunks.append('<p class="fr-research-note">当前显示定向返工阶段；首次取证见完整日志。</p>')
            if reason:
                chunks.append(f'<p class="fr-research-note">状态说明：{esc(REASONS.get(reason, reason))}</p>')
            if task.get('scope') or task.get('requirement_ids'):
                chunks.append('<details class="fr-task-scope"><summary>责任范围与要求</summary>')
                if task.get('scope'):
                    chunks.append(f'<p>{esc(task["scope"])}</p>')
                for rid in task.get('requirement_ids', []):
                    chunks.append(f'<p>{esc(rid)} · {esc(reqs.get(rid, "要求内容未记录"))}</p>')
                chunks.append('</details>')
            chunks.append('</article>')
        chunks.append('</div>')
    else:
        chunks.append('<p class="fr-research-note">规划完成后显示取证责任和依赖。</p>')
    pages = state['pages']
    if pages:
        cited = {pid for c in (result or {}).get('answer', {}).get('claims', []) for pid in c.get('evidence_page_ids', [])}
        chunks.append('<details class="fr-research-details"><summary>证据交接 · '+str(len(pages))+'张原页</summary>'
                      '<p class="fr-research-note">按工具记录列出成功读页的取证任务，不代表取证模型随后已看到页面图像。'
                      '分析接收版本和最终引用单独记录，引用不等于事实得到支持。</p>'
                      '<div class="fr-table-wrap"><table><thead><tr><th>原页</th><th>成功读页的取证任务</th><th>分析收到（版本）</th><th>最终引用</th></tr></thead><tbody>')
        for page in pages:
            receivers = '、'.join(roles.get(r, r) for r in page.get('received_by', [])) or '未记录'
            versions = '、'.join('v'+str(v) for v in page.get('analysis_versions', [])) or '未记录'
            link = _page_link(page, page_prefix) if state['terminal'] else esc(document_label(page.get('document_name') or page['page_id'])) + f' · 第{esc(page.get("page_number"))}页'
            chunks.append(f'<tr><td>{link}</td><td>{esc(receivers)}</td><td>{esc(versions)}</td>'
                          f'<td>{"已引用" if page["page_id"] in cited else "未引用" if state["terminal"] else "等待回答"}</td></tr>')
        chunks.append('</tbody></table></div></details>')
    if result is not None:
        rework = research.get('rework', {})
        if rework.get('triggered'):
            owner = task_names.get(rework.get('task_id'), rework.get('task_id') or '分析Agent')
            chunks.append(f'<aside class="fr-rework"><strong>已执行一次定向返工</strong><p>'
                          f'{"补充取证" if rework.get("retrieval") else "修订分析"} · 负责人：{esc(owner)}</p></aside>')
            drafts_by_version = sorted(research.get('drafts', []), key=lambda d: d.get('version', 0))
            if len(drafts_by_version) > 1:
                added = set(drafts_by_version[-1].get('page_ids', [])) - set(drafts_by_version[0].get('page_ids', []))
                added_links = [_page_link(p, page_prefix) for p in pages if p['page_id'] in added]
                if added_links:
                    chunks.append('<p class="fr-research-note">修订分析新增收到：' + '、'.join(added_links) + '</p>')
        checks, drafts = research.get('verifications', []), research.get('drafts', [])
        chunks.append('<details class="fr-research-details"><summary>草稿与模型核验'
                      f' · {len(drafts)}版草稿 / {len(checks)}次完成核验</summary>'
                      '<p class="fr-research-note">核验属于模型判断，未经独立人工确认。不同版本的同号声明不代表同一条修改。</p>')
        verification_label = {'passed':'模型核验完成', 'limited':'保留有限回答', 'incomplete':'核验未完成'}.get(
            research.get('verification_status'), '核验状态未记录')
        chunks.append(f'<p class="fr-research-note">{verification_label}；语义正确性仍需查看原页。</p>')
        used_versions = set()
        ordered_checks = [(_version(check, state['events'], position, checks), position, check)
                          for position, check in enumerate(checks)]
        ordered_checks.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else item[1]))
        for position, (version, _, check) in enumerate(ordered_checks):
            if version is not None:
                used_versions.add(version)
            draft = next((d for d in drafts if d.get('version') == version), {}) if version is not None else {}
            label = ('首次核验（返工前）' if position == 0 else '末次核验') if len(checks) > 1 else '首次核验'
            chunks.append(f'<section class="fr-draft"><h4>{label} · {"v"+str(version) if version is not None else "版本未记录"}</h4>')
            judgments = {c['id']: c for c in check.get('claims', [])}
            for claim in draft.get('numbered_claims', []):
                judgment = judgments.get(claim['id'], {})
                label = '模型判为支持' if judgment.get('supported') is True else '模型判为不支持' if judgment.get('supported') is False else '未记录判断'
                chunks.append(f'<article class="fr-review-claim"><strong>{esc(claim["id"])} · {label}</strong>'
                              f'<p>{esc(claim.get("text"))}</p><small>{esc(judgment.get("reason", ""))}</small></article>')
            if draft and not draft.get('numbered_claims'):
                chunks.append('<p class="fr-research-note">本版没有事实声明。</p>')
            chunks.append(_calculations(draft))
            for text in draft.get('answer', {}).get('unresolved', []):
                chunks.append(f'<p class="fr-research-note">草稿未解决：{esc(text)}</p>')
            for issue in check.get('issues', []):
                chunks.append(f'<p class="fr-review-issue">{esc(issue.get("detail", ""))}</p>')
            if not check.get('issues'):
                chunks.append('<p class="fr-research-note">未报告问题。</p>')
            chunks.append('</section>')
        for draft in drafts:
            if draft.get('version') in used_versions:
                continue
            chunks.append(f'<section class="fr-draft"><h4>草稿 v{esc(draft.get("version", "?"))} · 核验未完成或版本未对应</h4>')
            for c in draft.get('numbered_claims', []):
                chunks.append(f'<p>{esc(c.get("id"))} · {esc(c.get("text"))}</p>')
            chunks.append(_calculations(draft))
            for text in draft.get('answer', {}).get('unresolved', []):
                chunks.append(f'<p>{esc(text)}</p>')
            chunks.append('</section>')
        if not drafts and not checks:
            chunks.append('<p class="fr-research-note">尚无完整草稿与核验记录。</p>')
        chunks.append('</details>')
    chunks.append('</section>')
    return '\n'.join(chunks)


def research_css():
    return '''
.fr-research { color:#203732; background:#fff; border:1px solid #dde6e2; border-radius:16px; padding:22px; margin-bottom:16px; font:14px/1.65 system-ui, "Segoe UI", sans-serif; }
.fr-research * { box-sizing:border-box; }
.fr-research-head { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.fr-research-head h3 { margin:3px 0 14px; font-size:20px; font-weight:600; }
.fr-research .fr-eyebrow { color:#176a58; font-size:12px; }
.fr-state { padding:3px 8px; border-radius:6px; background:#f0f4f2; color:#50665e; font-size:12px; white-space:nowrap; }
.fr-state.running { background:#e4f3ec; color:#176a58; }
.fr-state.failed,.fr-state.blocked { background:#fff0e4; color:#8b5121; }
.fr-stages { display:flex; flex-wrap:wrap; list-style:none; gap:6px; padding:0!important; margin:0 0 14px!important; }
.fr-stages li { list-style:none; padding:5px 9px; color:#6a7e75; font-size:12px; background:#f5f7f6; border:1px solid transparent; border-radius:6px; }
.fr-stages li.active { color:#176a58; border-color:#176a58; background:#edf5f1; }
.fr-stages li.seen { color:#345d4e; background:#edf5f1; }
.fr-stages li.failed { color:#8b5121; border-color:#bc7e45; background:#fff5ea; }
.fr-task-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
.fr-task { border:1px solid #dde6e2; border-radius:10px; padding:15px; overflow-wrap:anywhere; }
.fr-task header { display:flex; flex-wrap:wrap; gap:6px; align-items:baseline; }
.fr-task h4 { margin:0; font-size:15px; flex:1; min-width:100px; }
.fr-task-number { color:#176a58; font-size:12px; font-weight:650; }
.fr-task p { margin:8px 0; font-size:13px; line-height:1.7; }
.fr-task-dependency { color:#176a58; }
.fr-research-note { color:#5c706a; font-size:12px; line-height:1.7; margin:8px 0; }
.fr-task-scope { border-top:1px solid #e7eeea; padding-top:8px; }
.fr-task-scope summary { color:#5c706a; font-size:12px; cursor:pointer; }
.fr-research-details { border-top:1px solid #dde6e2; margin-top:16px; padding-top:12px; }
.fr-research-details>summary { cursor:pointer; font-weight:600; color:#345d4e; }
.fr-research-details summary:focus-visible { outline:2px solid #176a58; outline-offset:3px; }
.fr-table-wrap { overflow-x:auto; }
.fr-research table { width:100%; border-collapse:collapse; font-size:12px; }
.fr-research th,.fr-research td { padding:10px 7px; text-align:left; vertical-align:top; border-bottom:1px solid #e7eeea; overflow-wrap:anywhere; }
.fr-research th { color:#5c706a; font-weight:500; }
.fr-evidence-link { color:#176a58!important; }
.fr-rework,.fr-research-warning { padding:12px 15px; background:#fff8ef; border-left:3px solid #ba844b; border-radius:4px; margin:14px 0; }
.fr-rework p { margin:4px 0 0; font-size:13px; }
.fr-draft { padding:12px 0; }
.fr-draft h4 { margin:6px 0 12px; font-size:15px; }
.fr-review-claim { padding:12px 14px; border:1px solid #dde6e2; border-radius:8px; margin:8px 0; }
.fr-review-claim strong { font-size:12px; color:#345d4e; }
.fr-review-claim p,.fr-review-issue { white-space:pre-wrap; overflow-wrap:anywhere; }
.fr-review-claim small { color:#5c706a; }
.fr-review-issue { border-left:2px solid #ba844b; padding-left:12px; }
@media(max-width:760px) { .fr-task-grid { grid-template-columns:1fr; } .fr-research { padding:16px; } .fr-research-head { align-items:flex-start; } }
'''
