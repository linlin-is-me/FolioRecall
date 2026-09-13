"""One resident model and index; display the same retrieval result as the CLI."""
import json
from pathlib import Path
import tempfile
import time

from .io import provenance, read_json, write_json
from .query import load_retriever, process_memory, retrieve


def display_request(model, config, index, pages, text, top_k, downloads, log=None):
    from PIL import Image
    started = time.perf_counter()
    ranked, payload, times = retrieve(model, config, index, pages, text, int(top_k))
    gallery, table, warnings = [], [], []
    for rank, row in enumerate(ranked, 1):
        name = row.get('document_name', row['doc_id'])
        table.append([rank, name, row['page_number'], row['score'], row['source']])
        try:
            with Image.open(row['preview']) as image:
                # Return a bounded copy, not arbitrary filesystem URLs.
                preview = image.convert('RGB')
                preview.thumbnail((1000, 1400))
                gallery.append((preview, f"{rank}. {name} · page {row['page_number']} · {row['score']:.4f}"))
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


def serve_demo(folder, teacher_config, query_config, port=7860, output=None):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ValueError('演示需要安装 foliorecall[demo]') from exc
    if not 1 <= port <= 65535:
        raise ValueError('端口必须在 1–65535 之间')
    if output and Path(output).exists():
        raise ValueError('演示验证记录目录已存在，请使用新目录')
    started = time.perf_counter()
    model, index, pages = load_retriever(folder, teacher_config, query_config)
    loading_seconds = time.perf_counter() - started
    log = None
    if output:
        provenance(output, {'page_config': teacher_config, 'query_config': query_config, 'index': str(folder)})
        write_json(Path(output) / 'startup.json', {'loading_seconds': loading_seconds,
            'model_loads': 1, 'device': query_config['device'], 'pages': len(pages),
            'memory': process_memory(), 'scope': 'model and index loading; excludes Gradio startup'})
        log = Path(output) / 'requests.jsonl'
    examples_path = Path(folder) / 'queries.json'
    examples = [[q['query']] for q in read_json(examples_path)] if examples_path.exists() else []
    temporary = tempfile.TemporaryDirectory(prefix='foliorecall-demo-')
    with gr.Blocks(title='FolioRecall · 页寻', analytics_enabled=False) as app:
        gr.Markdown('# FolioRecall · 页寻\n英文文档页面检索。分数表示相关性，不表示答案正确概率。')
        gr.Markdown(f"当前模型：`{query_config['model_id']}` · `{query_config['device']}` / `{query_config['dtype']}`\n\n页面库：`{Path(folder).name}` · {len(pages)} 页")
        text = gr.Textbox(label='英文问题', placeholder='Enter a question about these documents')
        top_k = gr.Slider(1, min(10, len(pages)), value=min(5, len(pages)), step=1, label='Top-k')
        button = gr.Button('检索', variant='primary')
        if examples:
            gr.Examples(examples=examples, inputs=text)
        status = gr.Markdown()
        table = gr.Dataframe(headers=['排名', '文档', '物理页码', '相关性', '来源'], interactive=False)
        gallery = gr.Gallery(label='相关页面', columns=2, object_fit='contain', height=650)
        download = gr.File(label='下载结果 JSON', interactive=False)
        def submit(query, k):
            try:
                return display_request(model, query_config, index, pages, query, k, temporary.name, log)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
        button.click(submit, [text, top_k], [gallery, table, download, status], api_name='search', concurrency_limit=1, concurrency_id='query')
        text.submit(submit, [text, top_k], [gallery, table, download, status], api_name=False, concurrency_limit=1, concurrency_id='query')
    try:
        app.launch(server_name='127.0.0.1', server_port=port, share=False, inbrowser=False)
    finally:
        temporary.cleanup()
