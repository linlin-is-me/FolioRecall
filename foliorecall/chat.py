"""Browser-session conversation and progress delivery for the document Agent."""
import copy
import html
from pathlib import Path
import queue
import threading
import time
import uuid

from .io import write_json
from .demo_style import document_label


def new_session():
    return {'conversation_id': uuid.uuid4().hex, 'turns': []}


def recent_context(session):
    return [turn for turn in session['turns'] if turn['completed']][-3:]


def answer_text(result):
    answer = result['answer']
    # The adjacent answer view renders full citation links; keep the transcript readable.
    text = '\n\n'.join(c['text'] for c in answer['claims'])
    if answer['unresolved']:
        text += '\n\n' + '\n'.join(answer['unresolved'])
    if not result['completed']:
        text = '任务未完成：' + result['termination_reason'] + '\n\n' + text
    return text.strip()


def messages_for(session):
    messages = []
    for turn in session['turns']:
        messages.extend([{'role': 'user', 'content': turn['question']},
                         {'role': 'assistant', 'content': answer_text(turn)}])
    return messages


def progress_html(events):
    labels = {'queued': '等待执行', 'model_request': '模型请求', 'query_rewrite': '追问改写',
              'coordinate': '任务分工', 'evidence_merge': '证据汇合',
              'search_pages': '搜索页面', 'read_page': '读取页面', 'source_access_status': '来源读取状态',
              'final_answer': '最终回答', 'terminal': '任务结束'}
    phases = {'query_translation': '中文转换', 'query_rewrite': '追问改写',
              'tool_decision': '选择工具', 'final_answer': '最终回答'}
    statuses = {'started': '执行中', 'completed': '完成', 'already_read': '本轮已读',
                'error': '错误', 'failed': '未完成', 'limit': '达到限额'}
    rows = []
    for event in events:
        full_name = ''
        name = labels.get(event['event'], event['event'])
        if event.get('role'):
            name = ('协调器' if event['role']=='coordinator' else event['role'].replace('worker-','检索Agent ')) + ' · ' + name
        detail = phases.get(event.get('phase'), '')
        if event.get('query'):
            detail = event['query']
        if event.get('page_id'):
            full_name = event.get('document_name') or event['page_id']
            detail = f"{document_label(full_name)} · 物理第{event.get('page_number', '?')}页"
        if event.get('pages'):
            detail += ' · 返回' + str(len(event['pages'])) + '页'
        if 'sources' in event:
            full_name = '；'.join(s.get('document_name') or s['doc_id'] for s in event['sources'])
            detail = '；'.join(document_label(s.get('document_name') or s['doc_id']) + '：候选物理页' +
                ','.join(str(p['page_number']) for p in s['retrieved_pages']) + '，已读物理页' +
                (','.join(str(p['page_number']) for p in s['read_pages']) or '无') for s in event['sources']) or '尚无候选或已读页面'
        if event.get('termination_reason'):
            detail = event['termination_reason']
        if event.get('reason'):
            detail += ' · ' + event['reason']
        if event.get('role') and 'elapsed_seconds' in event:
            detail += f' · {event["elapsed_seconds"]:.1f}秒'
        if event.get('cost_cny') is not None:
            detail += f' · {event["cost_cny"]:.6f}元'
        rows.append('<li><strong>' + html.escape(name) + '</strong><span class="fr-step-status ' +
                    html.escape(event['status'], quote=True) + '">' +
                    html.escape(statuses.get(event['status'], event['status'])) +
                    '</span><small title="' + html.escape(full_name, quote=True) + '">' + html.escape(detail) + '</small></li>')
    latest = events[-1] if events else None
    current = (labels.get(latest['event'], latest['event']) + ' · ' +
               statuses.get(latest['status'], latest['status'])) if latest else '等待执行'
    return ('<div id="agent-progress"><div class="fr-progress-head"><h3>执行步骤</h3><span>' +
            html.escape(current) + '</span></div><ol>' + ''.join(rows) + '</ol></div>')


def stream_chat(question, mode, session, execute, pages, request_lock, output=None):
    """Closing the consumer does not cancel or repeat an already started API task."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('请输入问题')
    current = copy.deepcopy(session or new_session())
    packets, events = queue.Queue(), []
    event_lock = threading.Lock()
    pending = messages_for(current) + [{'role': 'user', 'content': question},
                                       {'role': 'assistant', 'content': '正在执行…'}]

    def event(value):
        with event_lock:
            events.append(value)
            packets.put(('event', list(events)))

    def worker():
        try:
            with request_lock:
                result = execute(question, mode, recent_context(current), event)
                result.update(conversation_id=current['conversation_id'],
                              turn_index=len(current['turns'])+1, progress=list(events))
                # Preserve the terminal before rendering; a preview failure must not lose paid work.
                record_path = Path(output) / f'qa-{time.time_ns()}.json' if output else None
                if output:
                    write_json(record_path, result)
                from .demo import qa_display
                rendered = qa_display(result, pages)
                if record_path:
                    write_json(record_path, result)
            packets.put(('result', (result, rendered)))
        except Exception as exc:
            packets.put(('error', exc))

    threading.Thread(target=worker, daemon=True).start()
    yield pending, '', progress_html([{'event': 'queued', 'status': 'started'}]), None, current, True
    while True:
        kind, value = packets.get()
        if kind == 'event':
            yield pending, '', progress_html(value), None, current, True
        elif kind == 'error':
            yield pending[:-1] + [{'role': 'assistant', 'content': '执行错误：' + str(value)}], '', \
                progress_html(events + [{'event': 'terminal', 'status': 'failed', 'reason': str(value)}]), \
                None, current, False
            return
        else:
            result, rendered = value
            current['turns'].append({k: result[k] for k in
                ('question', 'retrieval_query', 'answer', 'completed', 'termination_reason')})
            yield messages_for(current), rendered, progress_html(events), result, current, False
            return
