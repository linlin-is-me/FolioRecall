"""Summarize completed CPU evidence without filling pending GPU result cells."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='outputs/stage4/summary')
args = parser.parse_args()
root, output = Path('outputs/stage4'), Path(args.output)
if output.exists():
    raise SystemExit('汇总目录已存在，请使用新目录')
config = read_json('configs/stage4-evaluation.json')
bm25, students = {}, {}
for task in config['tasks']:
    name = task['name']
    result = read_json(root / 'bm25' / name / 'result.json')
    bm25[name] = {key: result[key] for key in ('queries', 'candidates', 'metrics', 'request_seconds',
        'build_seconds', 'index_bytes', 'text_coverage', 'final_memory')}
for name in ('en', 'ml', 'distilled94'):
    result = read_json(root / 'hr-students' / name / 'result.json')
    students[name] = {key: result[key] for key in ('queries', 'candidates', 'metrics', 'request_seconds',
        'encode_seconds', 'model_loading_seconds', 'weight_file_bytes', 'index_bytes', 'final_memory')}
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
processes = [dict(read_json(p), record=str(p)) for p in sorted((root / 'runs').glob('*/process.json'))]
summary = {'stage4_status': 'CPU round complete; GPU validation and publication pending',
    'default': config['default_deployment'], 'bm25': bm25, 'bm25_task_macro': macro,
    'hr_cpu_students': students, 'hr_changes_vs_public_ml': changes,
    'cpu_task_process_seconds': sum(p['seconds'] for p in processes), 'cpu_tasks': processes,
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
(output / 'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(output / 'results.md')
