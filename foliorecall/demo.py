"""One resident model and index; display the same retrieval result as the CLI."""
import json
from pathlib import Path
import tempfile
import time
import threading

from .io import provenance, read_json, write_json
from .query import load_retriever, load_search_session, process_memory, retrieve


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
                gallery.append((preview, f"{rank}. 第 {row['page_number']} 页 · {row['score']:.4f} · {name}"))
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
    chunks = ['<h3>回答与引用</h3>']
    for claim in result['answer']['claims']:
        links = []
        for pid in claim['evidence_page_ids']:
            page = mapping[pid]
            links.append(f'<a href="#qa-page-{quote(pid,safe="")}">{escape(page.get("document_name",page["doc_id"]))} · {page["page_number"]}</a>')
        chunks.append('<p>'+escape(claim['text'])+' '+'; '.join(links)+'</p>')
    for text in result['answer']['unresolved']:
        chunks.append('<p>'+escape(text)+'</p>')
    for evidence in result['evidence']:
        pid = evidence['page_id']
        try:
            image,_ = read_evidence(mapping[pid],1310720)
            chunks.append(f'<figure id="qa-page-{escape(pid)}"><figcaption>{escape(evidence.get("document_name") or pid)} · 物理第 {evidence["page_number"]} 页</figcaption><img style="max-width:100%" src="{image["image_url"]["url"]}" alt="Evidence page"></figure>')
        except (OSError,ValueError):
            chunks.append('<p>证据预览暂不可读：'+escape(pid)+'</p>')
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
    with gr.Blocks(title='FolioRecall · 页寻', analytics_enabled=False) as app:
        gr.Markdown('# FolioRecall · 页寻\n英文文档页面检索。分数表示相关性，不表示答案正确概率。')
        if retrieval:
            gr.Markdown('混合检索：视觉与 BM25 各取 Top-50，以 RRF k=60 融合；展示融合分数，不是余弦相似度。两路名次与原分数见 JSON。')
        gr.Markdown(f"当前模型：`{query_config['model_id']}` · `{query_config['device']}` / `{query_config['dtype']}`\n\n页面库：`{Path(folder).resolve().name}` · {len(pages)} 页")
        with gr.Tab('页面检索'):
            text = gr.Textbox(label='英文问题', placeholder='Enter a question about these documents')
            top_k = gr.Slider(1, min(10, len(pages)), value=min(5, len(pages)), step=1, label='Top-k')
            button = gr.Button('检索', variant='primary')
            if examples:
                gr.Examples(examples=examples, inputs=text)
            status = gr.Markdown()
            table = gr.Dataframe(headers=['排名', '文档', '物理页码', '相关性', '来源'], interactive=False,
                                 column_widths=['60px', '450px', '100px', '110px', '360px'], wrap=True)
            gallery = gr.Gallery(label='相关页面', columns=2, object_fit='contain', height=650)
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
            with gr.Tab('文档问答'):
                gr.Markdown(f'远程服务：北京百炼 `{agent_config["model"]}`。提交问题后，会发送问题、检索摘要，以及最多 6 页公开资料的图像和正文；页面检索页仍为本地处理。')
                question = gr.Textbox(label='英文或中文问题')
                mode = gr.Radio(['rag','agent'],value='rag',label='固定 RAG / 有限工具 Agent')
                history = gr.Chatbot(label='当前会话', height=360, allow_tags=False, elem_id='conversation')
                session = gr.State(None)
                gr.Markdown('模型使用最近3轮成功回答理解追问；引用须来自当前轮读取的页面。刷新或新建会话清空上下文，不清空费用。')
                example_file = Path(folder) / 'qa-examples.json'
                if example_file.exists():
                    gr.Examples([[item['question'], item['mode']] for item in read_json(example_file)], inputs=[question, mode])
                answer_button = gr.Button('向百炼提交问答')
                reset_button = gr.Button('新建会话')
                progress = gr.HTML(elem_id='progress-panel')
                answer_html = gr.HTML(elem_id='answer-panel')
                with gr.Accordion('完整JSON、工具记录与费用', open=False):
                    record_json = gr.JSON(label='工具记录、终止原因、用量与估算费用', elem_id='qa-json')
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
                            yield transcript, rendered, steps, record, updated, \
                                gr.Button(interactive=not busy), gr.Button(interactive=not busy)
                    except ValueError as exc:
                        raise gr.Error(str(exc)) from exc
                targets = [history, answer_html, progress, record_json, session, answer_button, reset_button]
                answer_button.click(chat,[question,mode,session],targets,api_name='chat',concurrency_limit=1,concurrency_id='query')
                reset_button.click(lambda: ([], '', '', None, new_session()), None,
                                   [history, answer_html, progress, record_json, session], api_name=False,
                                   concurrency_limit=1, concurrency_id='query')
    try:
        app.launch(server_name='127.0.0.1', server_port=port, share=False, inbrowser=False)
    finally:
        temporary.cleanup()
