"""Export only the cleared-page candidate; no page encoding and no model load."""
import argparse
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json, write_rows
from foliorecall.search import load_index


def export_bundle(index_path, source_records, output):
    import faiss
    import numpy as np
    output = Path(output)
    if output.exists() or output.with_suffix('.zip').exists():
        raise ValueError('示例输出已存在，请使用新目录')
    config = read_json(Path(index_path) / 'config.json')
    index, rows = load_index(index_path, config)
    sources = {row['doc_name']: row for row in read_json(source_records)}
    population = 'a_demographic_perspective_on_the_future_of_european-KJ0125152ENN.pdf'
    excluded, selected = [], []
    for i, row in enumerate(rows):
        if row['document_name'] == population and row['page_number'] in (1, 6, 14):
            excluded.append({'page_id': row['page_id'], 'document_name': population, 'page_number': row['page_number']})
        else:
            selected.append((i, row))
    if len(rows) != 45 or len(selected) != 42 or len(excluded) != 3:
        raise ValueError('示例源页面与已核对的 45→42 页范围不符')
    (output / 'previews').mkdir(parents=True)
    exported = []
    for n, (_, row) in enumerate(selected):
        preview = Path('previews') / f'{n:03d}.png'
        shutil.copyfile(row['preview'], output / preview)
        exported.append(dict(row, preview=preview.as_posix(), source=sources[row['document_name']]['url']))
    vectors = np.ascontiguousarray([index.reconstruct(i) for i, _ in selected], dtype=np.float32)
    candidate = faiss.IndexFlatIP(index.d)
    candidate.add(vectors)
    faiss.write_index(candidate, str(output / 'index.faiss'))
    write_rows(output / 'pages.jsonl', exported)
    write_json(output / 'config.json', config)
    (output / 'configs').mkdir()
    for name in ('baseline.json', 'student-ml-cpu.json', 'student-ml-cuda.json'):
        shutil.copyfile(Path('configs') / name, output / 'configs' / name)
    queries = read_json('examples/demo_queries.json') + [
        {'query': 'What are the main employment and social trends in the European Union?',
         'note': '使用案例，需实际检查排名与页面，不作为人工相关性标签。'},
        {'query': 'What is the orbital period of Saturn?',
         'note': '范围外查询；该演示没有经过校准的拒答阈值，仍会返回页面。'}]
    write_json(output / 'queries.json', queries)
    write_json(output / 'bundle.json', {'name': 'FolioRecall EU reports demo', 'pages': 42,
        'parent_index': str(index_path), 'excluded': excluded, 'status': 'local candidate, publication review pending',
        'vector_operation': 'subset of existing vectors in original order; no re-encoding',
        'original_45_page_metrics_do_not_apply': True})
    source_text = '\n'.join(f"- {name}: {row['url']}" for name, row in sources.items())
    (output / 'SOURCES.md').write_text(
        '# 示例素材与来源\n\n© European Union, 2025.\n\n' + source_text +
        '\n\n人口报告主体内容采用 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。'
        '已排除物理第 1、6、14 页对应的 Adobe Stock 图片。就业报告版权页允许注明来源后再使用；'
        '非欧盟版权所有内容仍受各自权利约束。公开上传前须核对最终素材。\n\n'
        '修改：PDF 以 150 DPI 转 PNG；筛选上述 42 页并复用对应教师向量。保留原始物理页码。'
        '本包不包含原始 PDF，也不包含学生或教师权重。项目代码 Apache-2.0 不覆盖这些文档素材。\n', encoding='utf-8')
    (output / 'QUICKSTART.md').write_text(
        '# 本地候选示例\n\n在已安装 foliorecall[demo] 的环境中，从本目录运行：\n\n'
        '```bash\n# CPU 快速体验（仅需固定 ML 学生包）\n'
        'CUDA_VISIBLE_DEVICES= foliorecall serve --index . --config configs/baseline.json --query-config configs/student-ml-cpu.json\n'
        '# 已选定的 GPU 默认部署；本轮尚未进行应用 GPU 验证\n'
        'foliorecall serve --index . --config configs/baseline.json --query-config configs/student-ml-cuda.json\n```\n', encoding='utf-8')
    # Provenance is separate from the redistributable bundle; it may contain local paths.
    provenance(output.parent / (output.name + '-export-run'), {'index': str(index_path), 'excluded': excluded})
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output.parent))
    return {'bundle': str(output), 'pages': len(exported), 'zip_bytes': output.with_suffix('.zip').stat().st_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='indexes/demo-original')
    parser.add_argument('--sources', default='data/pdfs/sources.json')
    parser.add_argument('--output', default='outputs/stage4/demo-bundle')
    args = parser.parse_args()
    print(export_bundle(args.index, args.sources, args.output))
