"""One resident model and index; display the same retrieval result as the CLI."""
import base64
import json
from pathlib import Path
import tempfile
import time
import threading

from .io import provenance, read_json, write_json
from .query import load_retriever, load_search_session, process_memory, retrieve
from .demo_style import CSS, EMPTY_ANSWER, BUSY_ANSWER, document_label

LOGO_PATH = Path(__file__).with_name('assets') / 'logo.png'
WORDMARK_PATH = LOGO_PATH.with_name('wordmark.png')


def display_request(model, config, index, pages, text, top_k, downloads, log=None, retrieval=None):
    from PIL import Image
    started = time.perf_counter()
    ranked, payload, times = retrieve(model, config, index, pages, text, int(top_k),
                                     **({'retrieval': retrieval} if retrieval else {}))
    gallery, table, warnings = [], [], []
    for rank, row in enumerate(ranked, 1):
        name = row.get('document_name', row['doc_id'])
        table.append([rank, name, row['page_number'], round(row['score'], 6), row['source']])
        try:
            with Image.open(row['preview']) as image:
                # Return a bounded copy, not arbitrary filesystem URLs.
                preview = image.convert('RGB')
                preview.thumbnail((1000, 1400))
                gallery.append((preview, f"{rank}. {document_label(name)} · 物理第 {row['page_number']} 页"))
        except (OSError, ValueError) as exc:
            warnings.append(f"第 {rank} 项预览不可读：{exc}")
    target = Path(downloads) / f'results-{time.time_ns()}.json'
    target.write_text(payload + '\n', encoding='utf-8')
    # Single-user demo: retain only the current and previous downloads.
    for old in sorted(Path(downloads).glob('results-*.json'))[:-2]:
        old.unlink()
    timing = dict(times, display_ready_seconds=time.perf_counter() - started)
    if log:
        with Path(log).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'query': text, 'top_k': int(top_k), 'timing': timing,
                'page_ids': [r['page_id'] for r in ranked], 'warnings': warnings}) + '\n')
    message = (f"检索及 JSON 准备 {times['request_seconds'] * 1000:.1f} ms；"
               f"含页面读取与显示准备 {timing['display_ready_seconds'] * 1000:.1f} ms。"
               "不含 Gradio 序列化、网络传输及浏览器渲染。")
    if warnings:
        message += '\n\n' + '\n'.join(warnings)
    return gallery, table, str(target), message


def qa_display(result, pages):
    import html
    from urllib.parse import quote
    from .agent import read_evidence
    started = time.perf_counter()
    mapping = {p['page_id']:p for p in pages}
    escape = html.escape
    if not result.get('completed', True):
        label, tone = '任务未完成', 'warning'
    elif result['answer'].get('insufficient_evidence'):
        label, tone = '证据不足', 'warning'
    else:
        label, tone = '回答已生成', ''
    chunks = ['<section class="fr-answer"><header class="fr-answer-head"><h3>本轮回答与引用</h3>'
              f'<span class="fr-answer-status {tone}">{label}</span></header><div>']
    for claim in result['answer']['claims']:
        links = []
        for pid in claim['evidence_page_ids']:
            page = mapping[pid]
            name = escape(page.get('document_name', page['doc_id']))
            label = escape(document_label(page.get('document_name') or page['doc_id']))
            links.append(f'<a class="fr-citation" href="#qa-page-{quote(pid,safe="")}" title="{name}">{label} · 第 {page["page_number"]} 页</a>')
        chunks.append('<article class="fr-claim"><p>'+escape(claim['text'])+
                      '</p><div class="fr-citations">'+''.join(links)+'</div></article>')
    chunks.append('</div>')
    if result['answer']['unresolved']:
        chunks.append('<aside class="fr-unresolved"><h4>尚未解决</h4>')
        for text in result['answer']['unresolved']:
            chunks.append('<p>'+escape(text)+'</p>')
        chunks.append('</aside>')
    if not result.get('completed', True):
        chunks.append('<p class="fr-note">终止原因：'+escape(result.get('termination_reason', 'unknown'))+'</p>')
    if result['evidence']:
        chunks.append('<div class="fr-evidence-list"><h3>本轮读取的原页</h3>')
    for evidence in result['evidence']:
        pid = evidence['page_id']
        full_name = evidence.get('document_name') or mapping[pid].get('document_name') or pid
        name = escape(full_name)
        caption = f'{name} · 物理第 {evidence["page_number"]} 页'
        label = f'{escape(document_label(full_name))} · 物理第 {evidence["page_number"]} 页'
        # Native details are revealed by fragment navigation to the figure inside.
        chunks.append(f'<details class="fr-evidence"><summary title="{name}">{label}</summary><figure id="qa-page-{escape(pid)}"><figcaption>{caption}</figcaption>')
        try:
            image,_ = read_evidence(mapping[pid],1310720)
            chunks.append(f'<img src="{escape(image["image_url"]["url"], quote=True)}" alt="{caption}">')
        except (OSError,ValueError):
            chunks.append('<p>证据预览暂不可读：'+escape(pid)+'</p>')
        chunks.append('</figure></details>')
    if result['evidence']:
        chunks.append('</div>')
    chunks.append('</section>')
    result['timing']['display_preparation_seconds'] = time.perf_counter()-started
    result['timing']['task_and_display_seconds'] = result['timing']['task_seconds'] + result['timing']['display_preparation_seconds']
    return '\n'.join(chunks)


def serve_demo(folder, teacher_config, query_config, port=7860, output=None, retrieval_config=None,
               agent_config_path=None, public_corpus=None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ValueError('演示需要安装 foliorecall[demo]') from exc
    if not 1 <= port <= 65535:
        raise ValueError('端口必须在 1–65535 之间')
    if output and Path(output).exists():
        raise ValueError('演示验证记录目录已存在，请使用新目录')
    provider, agent_config, evidence_texts = None, None, None
    if agent_config_path:
        from .generation import read_agent_config, Bailian
        from .agent import load_public_corpus
        from .search import load_index
        if not public_corpus:
            raise ValueError('启用远程问答必须提供 --public-corpus 公开资料清单')
        agent_config = read_agent_config(agent_config_path)
        _, corpus_pages = load_index(folder, teacher_config)
        evidence_texts = load_public_corpus(public_corpus, corpus_pages)
        provider = Bailian(agent_config)
    started = time.perf_counter()
    model, index, pages, retrieval = load_search_session(folder, teacher_config, query_config,
                                                        retrieval_config=retrieval_config)
    loading_seconds = time.perf_counter() - started
    log = None
    if output:
        provenance(output, {'page_config': teacher_config, 'query_config': query_config, 'index': str(folder),
                            'retrieval_config': retrieval_config, 'agent_config': agent_config,
                            'public_corpus': str(public_corpus) if public_corpus else None})
        write_json(Path(output) / 'startup.json', {'loading_seconds': loading_seconds,
            'model_loads': 1, 'device': query_config['device'], 'pages': len(pages),
            'memory': process_memory(), 'scope': 'model and index loading; excludes Gradio startup'})
        log = Path(output) / 'requests.jsonl'
    examples_path = Path(folder) / 'queries.json'
    examples = [[q['query']] for q in read_json(examples_path)] if examples_path.exists() else []
    temporary = tempfile.TemporaryDirectory(prefix='foliorecall-demo-')
    request_lock = threading.Lock()
    from html import escape
    backend = 'ONNX CPU' if query_config.get('backend') == 'onnx' else query_config['device'].upper()
    logo = base64.b64encode(WORDMARK_PATH.read_bytes()).decode('ascii')
    theme = gr.themes.Base(primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.slate,
                           font=['system-ui', 'Segoe UI', 'Microsoft YaHei', 'sans-serif'],
                           font_mono=['ui-monospace', 'Consolas', 'monospace'])
    with gr.Blocks(title='FolioRecall · 页寻', analytics_enabled=False) as app:
        gr.HTML(f'''<header class="fr-header"><div class="fr-brand">
<div><h1><img class="fr-wordmark" src="data:image/png;base64,{logo}" width="232" height="45" alt="FolioRecall"><span>页寻</span></h1><p>找到文档证据，展开有据可查的对话。</p></div></div>
<div class="fr-badges"><span class="fr-badge">{len(pages)} 页文档库</span><span class="fr-badge">{escape(backend)}</span>
<span class="fr-badge">{'视觉 + BM25' if retrieval else '视觉检索'}</span></div></header>''')
        documents = {}
        for page in pages:
            documents.setdefault(page['doc_id'], page.get('document_name') or page['doc_id'])
        library = ''.join(f'<span title="{escape(name)}">{escape(document_label(name))}</span>'
                          for name in list(documents.values())[:3])
        if len(documents) > 3:
            library += f'<span>另有 {len(documents) - 3} 份文档</span>'
        gr.HTML(f'<div class="fr-library"><strong>当前文档库 · {len(documents)} 份 / {len(pages)} 页</strong>{library}</div>')
        with gr.Tabs(selected='chat' if provider else 'search', elem_id='workspace-tabs'):
          with gr.Tab('页面检索', id='search'):
            with gr.Row(elem_classes=['fr-workspace']):
              with gr.Column(scale=1, min_width=300, elem_classes=['fr-controls']):
                gr.Markdown('## 查找原页\n输入英文问题，从文字、表格与图表中寻找相关页面。', elem_classes=['fr-section'])
                text = gr.Textbox(label='英文问题', placeholder='What would you like to find in these documents?', lines=4)
                top_k = gr.Slider(1, min(10, len(pages)), value=min(5, len(pages)), step=1, label='返回页数', info='按相关性排序')
                button = gr.Button('检索页面', variant='primary')
                if examples:
                    gr.Examples(examples=examples[:3], inputs=text, label='试试这些问题 · 点击填入，不自动提交')
                    if len(examples) > 3:
                        with gr.Accordion('更多示例', open=False):
                            gr.Examples(examples=examples[3:], inputs=text, label='示例问题')
                gr.Markdown('本地检索，不调用生成 API。相关性分数不代表答案正确概率。', elem_classes=['fr-note'])
              with gr.Column(scale=2, min_width=0, elem_classes=['fr-results']):
                gr.Markdown('## 相关页面\n检索后点击缩略图放大，查看原始版面与物理页码。', elem_classes=['fr-section'])
                gallery = gr.Gallery(label='页面预览', columns=2, object_fit='contain', height=560, elem_id='search-gallery')
                status = gr.Markdown('等待检索。', elem_id='search-status')
                with gr.Accordion('结果明细与 JSON 下载', open=False):
                    table = gr.Dataframe(headers=['排名', '文档', '物理页码', '相关性', '来源'], interactive=False,
                                         column_widths=['60px', '250px', '90px', '100px', '260px'], wrap=True, elem_id='search-table')
                    download = gr.File(label='下载结果 JSON', interactive=False)
            def submit(query, k):
                try:
                    with request_lock:
                        return display_request(model, query_config, index, pages, query, k, temporary.name, log, retrieval)
                except ValueError as exc:
                    raise gr.Error(str(exc)) from exc
            button.click(submit, [text, top_k], [gallery, table, download, status], api_name='search', concurrency_limit=1, concurrency_id='query')
            text.submit(submit, [text, top_k], [gallery, table, download, status], api_name=False, concurrency_limit=1, concurrency_id='query')
          if provider:
            from .chat import new_session, stream_chat
            with gr.Tab('文档问答', id='chat'):
                with gr.Row(elem_classes=['fr-workspace']):
                  with gr.Column(scale=1, min_width=330, elem_classes=['fr-controls']):
                    gr.Markdown('## 与文档对话\n提出问题，或接着上一轮继续追问。', elem_classes=['fr-section'])
                    mode = gr.Radio([('RAG · 直接回答','rag'), ('Agent · 工具补查','agent')], value='rag', label='回答方式')
                    session = gr.State(None)
                    question = gr.Textbox(label='英文或中文问题', placeholder='输入问题，或追问证据、条件与时间范围…', lines=3)
                    with gr.Row():
                        answer_button = gr.Button('提交问答', variant='primary', scale=2)
                        reset_button = gr.Button('新建会话', scale=1)
                    gr.Markdown('问答会调用百炼，并发送问题与相关公开页面。最近 3 轮成功回答用于理解追问。', elem_classes=['fr-note'])
                    history = gr.Chatbot(label='当前会话', height=260, allow_tags=False, elem_id='conversation',
                                         buttons=['copy', 'copy_all'])
                    gr.Markdown('刷新或新建会话清空上下文，不清空费用。引用仅来自本轮读取的页面。', elem_classes=['fr-note'])
                  with gr.Column(scale=2, min_width=0, elem_classes=['fr-results']):
                    answer_html = gr.HTML(value=EMPTY_ANSWER, elem_id='answer-panel')
                    example_file = Path(folder) / 'qa-examples.json'
                    if example_file.exists():
                        starters, followups, fresh = [], [], []
                        for item in read_json(example_file):
                            (followups if item.get('followup') else fresh if item.get('new_conversation') else starters).append(item)
                        if starters:
                            gr.Examples([[item['question'], item['mode']] for item in starters[:3]],
                                        inputs=[question, mode], label='从这里开始 · 点击填入，不自动提交')
                        with gr.Accordion('追问与更多示例', open=False):
                            if starters[3:]:
                                gr.Examples([[item['question'], item['mode']] for item in starters[3:]],
                                            inputs=[question, mode], label='更多首问')
                            if followups:
                                gr.Examples([[item['question'], item['mode']] for item in followups],
                                            inputs=[question, mode], label='先完成相关首问，再使用追问')
                            if fresh:
                                gr.Examples([[item['question'], item['mode']] for item in fresh],
                                            inputs=[question, mode], label='先点击新建会话，再尝试这些问题')
                    else:
                        gr.Markdown('从当前文档中的一个主题开始，例如询问某项结论及其证据页，再追问适用条件或时间范围。', elem_classes=['fr-note'])
                    progress = gr.HTML(elem_id='progress-panel')
                    with gr.Accordion('完整 JSON、工具记录与费用', open=False):
                        record_json = gr.JSON(label='本轮完整记录', elem_id='qa-json')
                def submit_qa(query, selected_mode):
                    from .agent import ask
                    try:
                        with request_lock:
                            result = ask(query,selected_mode,model,query_config,index,pages,agent_config,provider,retrieval,evidence_texts)
                            rendered = qa_display(result,pages)
                            if output:
                                write_json(Path(output)/f'qa-{time.time_ns()}.json',result)
                        return rendered,result
                    except ValueError as exc:
                        raise gr.Error(str(exc)) from exc
                legacy_button = gr.Button(visible=False)
                legacy_button.click(submit_qa,[question,mode],[answer_html,record_json],api_name='ask',concurrency_limit=1,concurrency_id='query')
                def chat(query, selected_mode, state):
                    from .agent import ask
                    def execute(q, selected, context, event):
                        return ask(q, selected, model, query_config, index, pages, agent_config, provider,
                                   retrieval, evidence_texts, conversation=context, on_event=event)
                    try:
                        for transcript, rendered, steps, record, updated, busy in stream_chat(
                                query, selected_mode, state, execute, pages, request_lock, output):
                            yield transcript, rendered or (BUSY_ANSWER if busy else EMPTY_ANSWER), steps, record, updated, \
                                gr.Button(interactive=not busy), gr.Button(interactive=not busy)
                    except ValueError as exc:
                        raise gr.Error(str(exc)) from exc
                targets = [history, answer_html, progress, record_json, session, answer_button, reset_button]
                answer_button.click(chat,[question,mode,session],targets,api_name='chat',concurrency_limit=1,concurrency_id='query')
                def reset_session():
                    return [], EMPTY_ANSWER, '', None, new_session(), ''
                reset_targets = [history, answer_html, progress, record_json, session, question]
                reset_button.click(reset_session, None, reset_targets, api_name=False,
                                   concurrency_limit=1, concurrency_id='query')
                history.clear(reset_session, None, reset_targets, api_name=False,
                              concurrency_limit=1, concurrency_id='query')
        with gr.Accordion('模型与运行信息', open=False, elem_id='runtime-info'):
            gr.Markdown(f"查询模型：`{query_config['model_id']}` · `{query_config['device']}` / `{query_config['dtype']}`\n\n页面库：`{Path(folder).resolve().name}` · {len(pages)} 页")
            if retrieval:
                gr.Markdown('视觉与 BM25 各取 Top-50，以 RRF k=60 融合。两路名次与原分数见 JSON。')
            if query_config.get('backend') == 'onnx':
                gr.Markdown(f"查询后端：ONNX Runtime CPU · `{query_config['onnx_file']}`；池化、投影与归一化使用 FP32。")
            if provider:
                gr.Markdown(f'问答模型：北京百炼 `{agent_config["model"]}`。每轮最多读取 6 页公开资料；新建会话不清空费用账本。')
    try:
        app.launch(server_name='127.0.0.1', server_port=port, share=False, inbrowser=False,
                   theme=theme, css=CSS, favicon_path=str(LOGO_PATH))
    finally:
        temporary.cleanup()
