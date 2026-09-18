"""Freeze deterministic retrospective development IDs; never invent QA gold labels."""
import argparse
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json, read_rows, write_json, write_rows, provenance
from foliorecall.benchmark_data import load_dataset_provenance


def select_tasks(data, config, output):
    output = Path(output)
    if output.exists():
        raise ValueError('任务目录已存在，请使用新目录')
    selected, sources = [], []
    for spec in config['tasks']:
        folder = Path(data) / spec['name']
        source = load_dataset_provenance(folder)
        if source['revision'] != spec['revision']:
            raise ValueError('任务 revision 不匹配')
        sources.append(source)
        queries = sorted(read_rows(folder/'queries.jsonl'), key=lambda q: str(q['query_id']))
        qrels = read_json(folder/'qrels.json')
        pages = read_rows(folder/'pages.jsonl')
        if len(pages) != spec['pages'] or len(queries) != spec['english_queries']:
            raise ValueError('任务页面或查询数量不匹配')
        used = set()
        predicates = {
            'numeric': lambda q: bool(re.search(r'\b(how many|how much|percentage|percent|rate|number|difference)\b', q['query'], re.I)),
            'chart': lambda q: 'Chart' in q.get('content_type', []),
            'multi_page': lambda q: len([v for v in qrels[str(q['query_id'])].values() if v > 0]) > 1,
            'single_page': lambda q: len([v for v in qrels[str(q['query_id'])].values() if v > 0]) == 1,
        }
        for category, predicate in predicates.items():
            candidates = [q for q in queries if str(q['query_id']) not in used and predicate(q)]
            if len(candidates) < 2:
                raise ValueError(f'{spec["name"]}/{category} 缺少任务，不能静默替换分类')
            for query in candidates[:2]:
                qid = str(query['query_id'])
                used.add(qid)
                selected.append({'task_id': spec['name']+':'+qid, 'task': spec['name'], 'query_id': qid,
                    'category': category, 'question_en': query['query'], 'question_zh': None,
                    'reference_answer': query.get('answer', query.get('raw_answers')),
                    'evidence_qrels': qrels[qid], 'review': {'official_answer': 'unverified',
                    'assistant': 'pending', 'human': 'not_checked'},
                    'scope': 'retrospective development; metadata category, pending page verification'})
    provenance(output, {'selection': 'string query_id ascending, two per numeric/chart/multi_page/single_page in this order; without replacement',
                        'tasks': config['tasks']})
    write_rows(output/'tasks.jsonl', selected)
    write_json(output/'sources.json', sources)
    write_json(output/'status.json', {'logical_tasks': len(selected), 'selection_frozen': True,
        'qa_reference_frozen': False, 'blocking': ['page/answer verification', 'paired Chinese questions'],
        'evaluation_allowed': False})
    return len(selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='configs/stage4-evaluation.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(select_tasks(args.data, read_json(args.config), args.output))
