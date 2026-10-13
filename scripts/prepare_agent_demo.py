"""Add text-layer retrieval and QA configuration to the retained 42-page demo."""
import argparse
from pathlib import Path
import shutil
import zipfile

from foliorecall.documents import extract_page_texts
from foliorecall.io import read_json, read_rows, write_json
from foliorecall.retrieval import build_text_index, page_identity


def prepare(bundle, sources, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    source_records = read_json(sources)
    rows = read_rows(bundle / 'pages.jsonl')
    if len(rows) != 42:
        raise ValueError('需要已有42页示例')
    by_name = {r['doc_name']: r for r in source_records}
    local_rows = [dict(r, source=by_name[r['document_name']]['local_path']) for r in rows]
    if any(not Path(r['source']).is_file() for r in local_rows):
        raise ValueError('原PDF缓存不可用')
    if output.exists():
        raise ValueError('输出目录已存在，请使用新目录')
    shutil.copytree(bundle, output)
    # Extraction metadata stays outside the redistributable directory.
    temporary = output.parent / (output.name + '-text-preparation')
    temporary.mkdir()
    texts = extract_page_texts(local_rows, temporary)
    shutil.copyfile(texts, output / 'texts.jsonl')
    build_text_index(output / 'pages.jsonl', output / 'texts.jsonl', output / 'text-index')
    text_record = read_json(output / 'text-index/index.json')
    text_record['source'] = {'path': '../texts.jsonl'}
    write_json(output / 'text-index/index.json', text_record)
    text_result = read_json(output / 'text-index/result.json')
    text_result['index'] = 'text-index'
    write_json(output / 'text-index/result.json', text_result)
    # Build provenance can contain local paths and is not an example asset.
    for name in ('run.json', 'config.json', 'working-tree.patch', 'untracked'):
        path = output / 'text-index' / name
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    write_json(output / 'public-corpus.json', {
        'visibility': 'public', 'pages': page_identity(rows), 'texts': 'texts.jsonl',
        'source_urls': [r['url'] for r in source_records],
        'license': 'See SOURCES.md; retained cleared 42-page subset with attribution'})
    write_json(output / 'hybrid.json', {'mode': 'hybrid', 'text_index': 'text-index',
                                       'rrf_k': 60, 'candidate_k': 50})
    agent = read_json(Path(__file__).resolve().parents[1] / 'configs/agent-bailian-qwen37.json')
    agent.update(phase_budget_cny=20, ledger='../outputs/api-costs.sqlite3')
    write_json(output / 'configs/agent-demo.json', agent)
    write_json(output / 'qa-examples.json', [
        {'mode': 'rag', 'question': 'Which figure compares projections of the EU working-age population under baseline and no-migration scenarios?'},
        {'mode': 'rag', 'question': '这张图中，无移民情景与基准情景有什么差异？', 'followup': True},
        {'mode': 'agent', 'question': 'What do the demographic and employment reports say about changes in the working-age population and employment in the EU?'},
        {'mode': 'agent', 'question': '分别列出这两份报告支持结论的页码，并说明分析时间范围。', 'followup': True},
        {'mode': 'agent', 'question': 'What is the orbital period of Saturn?', 'new_conversation': True}])
    manifest = read_json(output / 'bundle.json')
    manifest.update(version='0.1.0', status='local release candidate pending review',
                    parent_index='retained cleared 42-page visual index', qa=True)
    write_json(output / 'bundle.json', manifest)
    (output / 'QUICKSTART.md').write_text('''# FolioRecall 42页RAG与Agent示例

先在Python 3.10环境安装foliorecall[demo,agent]，再进入本目录。
首次需要获取配置固定的公开查询学生，之后可离线检索。

```bash
python -m foliorecall serve --index . --config configs/baseline.json --query-config configs/student-ml-cpu.json --retrieval-config hybrid.json --port 7860
```

问答需在本机设置DASHSCOPE_API_KEY和DASHSCOPE_BASE_URL，再运行：

```bash
python -m foliorecall serve --index . --config configs/baseline.json --query-config configs/student-ml-cpu.json --retrieval-config hybrid.json --agent-config configs/agent-demo.json --public-corpus public-corpus.json --port 7860
```

打开http://127.0.0.1:7860。问答调用远程API并产生费用；示例内部预算20元，可修改自己的配置。
模型只读取最近3轮成功回答，刷新或新建会话清空上下文，不清空费用。
素材许可见SOURCES.md，不附原PDF、权重或凭据。
''', encoding='utf-8')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob('*')):
            if file.is_file():
                archive.write(file, file.relative_to(output.parent))
    return {'pages': 42, 'output': str(output), 'text_pages': text_record['nonempty_pages']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(prepare(args.bundle, args.sources, args.output))
