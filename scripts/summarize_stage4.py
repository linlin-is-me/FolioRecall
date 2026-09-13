"""Summarize completed CPU evidence without filling pending GPU result cells."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, read_rows, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='outputs/stage4/summary')
parser.add_argument('--matrix', help='Aggregate remaining formal GPU/CPU jobs instead of the historical CPU round')
args = parser.parse_args()
if args.matrix:
    from foliorecall.stage4_results import summarize_matrix
    result = summarize_matrix(args.matrix, args.output)
    print(f"Completed {len(result['cells'])}/40; pending {len(result['pending'])}")
    raise SystemExit(0)
root, output = Path('outputs/stage4'), Path(args.output)
if output.exists():
    raise SystemExit('汇总目录已存在，请使用新目录')
config = read_json('configs/stage4-evaluation.json')
provenance(output, {'source': str(root), 'evaluation': config, 'operation': 'aggregate existing CPU evidence; no model execution'})
bm25, students = {}, {}
for task in config['tasks']:
    name = task['name']
    result = read_json(root / 'bm25' / name / 'result.json')
    bm25[name] = {key: result[key] for key in ('queries', 'candidates', 'metrics', 'request_seconds',
        'build_seconds', 'index_bytes', 'text_coverage', 'warm_memory', 'final_memory')}
for name in ('en', 'ml', 'distilled94'):
    result = read_json(root / 'hr-students' / name / 'result.json')
    students[name] = {key: result[key] for key in ('queries', 'candidates', 'metrics', 'request_seconds',
        'encode_seconds', 'query_seconds', 'model_loading_seconds', 'weight_file_bytes', 'index_bytes', 'warm_memory', 'final_memory')}
macro = {key: sum(r['metrics'][key] for r in bm25.values()) / len(bm25) for key in ('nDCG@10', 'Recall@5', 'Recall@10')}
reference = read_json(root / 'hr-students/ml/result.json')
initial = {row['query_id']: row for row in reference['results']}
changes = {}
for name in ('en', 'distilled94'):
    compared = read_json(root / 'hr-students' / name / 'result.json')
    deltas = [{'query_id': row['query_id'], 'query': row['query'],
        'delta_nDCG@10': row['nDCG@10'] - initial[row['query_id']]['nDCG@10'],
        'reference_ranking': initial[row['query_id']]['ranking'], 'candidate_ranking': row['ranking']}
        for row in compared['results']]
    changes[name] = {'better': sum(r['delta_nDCG@10'] > 0 for r in deltas),
        'worse': sum(r['delta_nDCG@10'] < 0 for r in deltas),
        'equal': sum(r['delta_nDCG@10'] == 0 for r in deltas),
        'largest_five': sorted(deltas, key=lambda r: (-abs(r['delta_nDCG@10']), r['query_id']))[:5]}
    write_json(output / f'hr-{name}-vs-ml.json', deltas)
from collections import Counter
qrels = read_json(root / 'data/hr/qrels.json')
texts = {r['page_id']: r['text'] for r in read_rows(root / 'data/hr/texts.jsonl')}
reviews = []
for row in changes['distilled94']['largest_five']:
    item = {key: row[key] for key in ('query_id', 'query', 'delta_nDCG@10')}
    for side in ('reference', 'candidate'):
        top = row[side + '_ranking'][0]
        item[side + '_top1'] = dict(top, official_grade=qrels[row['query_id']].get(top['page_id'], 0),
                                  upstream_markdown=texts[top['page_id']])
    reviews.append(item)
top1 = {}
for name in students:
    ranked = read_json(root / 'hr-students' / name / 'result.json')['results']
    top1[name] = Counter(r['ranking'][0]['page_id'] for r in ranked).most_common(5)
write_json(output / 'hr-error-evidence.json', {'largest_changes': reviews, 'top1_frequency': top1,
    'scope': 'official grades and source page text; no relabeling or claim of a unique causal mechanism'})
data = {t['name']: read_json(root / 'data' / t['name'] / 'source.json') for t in config['tasks']}
demo = read_json(root / 'demo-api-v2/result.json')
browser = read_json(root / 'browser-v2/interaction-checks.json')
processes = [dict(read_json(p), record=str(p)) for p in sorted((root / 'runs').glob('*/process.json'))]
summary = {'stage4_status': 'CPU round complete; GPU validation and publication pending',
    'default': config['default_deployment'], 'bm25': bm25, 'bm25_task_macro': macro,
    'hr_cpu_students': students, 'hr_changes_vs_public_ml': changes,
    'data_preparation': data, 'demo_cpu': demo, 'browser_validation': browser,
    'cpu_task_process_seconds': sum(p['seconds'] for p in processes), 'cpu_tasks': processes,
    'process_time_scope': 'sum of recorded child wall times, including demo service idle time; excludes download and development; not GPU compute or a complete CPU utilization measure',
    'pending': ['Original and LoRA750 page indexes for the five-domain comparison (reuse existing original HR)',
        'five GPU candidates and remaining four-domain CPU student evaluations',
        'installed custom-PDF GPU path and necessary GPU resume check', 'final materials review and publication authorization']}
write_json(output / 'result.json', summary)
lines = ['# 第四阶段首轮结果', '', '本轮全部为 CPU。默认部署仍为公开 ML GPU BF16；GPU应用与正式评测待执行。', '',
         '## 五领域 BM25：上游页面 OCR 文本', '',
         '| 任务 | nDCG@10 | Recall@5 | Recall@10 | 请求P50/P95 ms | 文本覆盖率 |', '|---|---:|---:|---:|---:|---:|']
for name, r in bm25.items():
    m, t = r['metrics'], r['request_seconds']
    lines.append(f"| {name} | {m['nDCG@10']:.6f} | {m['Recall@5']:.6f} | {m['Recall@10']:.6f} | {t['p50']*1000:.2f}/{t['p95']*1000:.2f} | {r['text_coverage']:.1%} |")
lines += [f"| 任务宏平均 | {macro['nDCG@10']:.6f} | {macro['Recall@5']:.6f} | {macro['Recall@10']:.6f} | 不跨任务合并延迟 | — |", '',
    '## HR完整318查询：CPU FP32学生', '',
    '| 学生 | nDCG@10 | Recall@5 | Recall@10 | 请求P50/P95 ms | 峰值RSS GB |', '|---|---:|---:|---:|---:|---:|']
for name, r in students.items():
    m, t = r['metrics'], r['request_seconds']
    lines.append(f"| {name} | {m['nDCG@10']:.6f} | {m['Recall@5']:.6f} | {m['Recall@10']:.6f} | {t['p50']*1000:.2f}/{t['p95']*1000:.2f} | {r['final_memory']['peak_rss_bytes']/1e9:.3f} |")
lines += ['', '计时为独立进程、batch1、预热5次、固定查询顺序3轮；JSON准备包含在请求内，模型加载和指标计算另计。BM25上游OCR成本未知。', '',
          '各模型配置、逐查询排名及逐请求耗时见相邻原运行目录。不同任务不混作同一质量对照，HR学生结果不代表五领域学生成绩。', '',
          '五领域共12969页、1489条英文查询。没有与本项目3000条训练查询的规范化文本交集；HR的20条查询已有流程用途，上游训练及文档级重叠仍未完全核实。']
lines += ['', '## BM25建库与数据准备', '',
    '| 领域 | 文本建库 s | 文本索引 MB | 预热/峰值RSS MB | 新下载/新提取图像 GB | 下载/准备 s |',
    '|---|---:|---:|---:|---:|---:|']
for name, r in bm25.items():
    d = data[name]
    lines.append(f"| {name} | {r['build_seconds']:.3f} | {r['index_bytes']/1e6:.2f} | {r['warm_memory']['rss_bytes']/1e6:.1f}/{r['final_memory']['peak_rss_bytes']/1e6:.1f} | {d['new_source_file_bytes']/1e9:.3f}/{d['extracted_image_bytes']/1e9:.3f} | {d['download_seconds']:.1f}/{d['preparation_seconds']:.1f} |")
lines += ['', '单位为十进制MB/GB。HR复用1110张历史页面。下载与提取分开记录，准备计时不包含下载；文本建库从读取页面markdown到索引文件写完，不包括上游OCR。', '',
    '## 学生查询分项与模型加载', '',
    '| 学生 | 编码P50/P95 ms | 编码加搜索P50/P95 ms | 加载 s | 预热RSS MB | 权重/索引 MB |',
    '|---|---:|---:|---:|---:|---:|']
for name, r in students.items():
    e, q = r['encode_seconds'], r['query_seconds']
    lines.append(f"| {name} | {e['p50']*1000:.2f}/{e['p95']*1000:.2f} | {q['p50']*1000:.2f}/{q['p95']*1000:.2f} | {r['model_loading_seconds']:.2f} | {r['warm_memory']['rss_bytes']/1e6:.1f} | {r['weight_file_bytes']/1e6:.2f}/{r['index_bytes']/1e6:.2f} |")
lines += ['', '相同CPU FP32条件，三款学生成本接近。EN在HR略高于ML；第94步明显下降。保留默认ML，等待五领域GPU及其余CPU结果。', '',
    '## 42页真实演示', '',
    f"常驻模型加载 {demo['startup']['loading_seconds']:.2f} s，只加载一次。5次预热后，3个查询各重复3轮；客户端响应P50/P95为 {demo['client_response_seconds']['p50']*1000:.1f}/{demo['client_response_seconds']['p95']*1000:.1f} ms。",
    '', '| 服务端边界 | P50/P95 ms |', '|---|---:|']
for key, label in [('encode_seconds', '分词及编码'), ('request_seconds', '结果JSON就绪'), ('display_ready_seconds', '含页面读取与显示准备')]:
    t = demo['server_timings'][key]
    lines.append(f"| {label} | {t['p50']*1000:.2f}/{t['p95']*1000:.2f} |")
lines += ['', '客户端统计包含Gradio队列、序列化及响应元数据传输，排除预览下载与浏览器渲染。真实Chrome功能查询从点击到响应文字及预览图加载为单次测量，见browser-v2/result.json，不当作P50/P95。', '',
    '人口预测问题命中物理第10页Figure 3，JSON与CLI一致；范围外的土星问题仍返回无关页面，演示没有无答案检测。截图和交互证据见browser-v2，全部测量未启用CUDA。']
(output / 'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(output / 'results.md')
