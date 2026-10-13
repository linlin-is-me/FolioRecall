"""Export blinded A6 material and summarize actual human ratings; no API calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from foliorecall.io import read_json, read_rows
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_evaluation import RATINGS, summarize_attempts


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('材料目录已存在，请保留原材料并使用新目录')
    mapping = read_json(source / 'a6-private-mapping.json')
    if mapping['seed'] != 42 or not mapping['fixed_before_answers']:
        raise ValueError('人工抽样不是既有冻结安排')
    selection = mapping['selection']
    expected = {(f'jrc-{n:02}', 'en') for n in (1, 11, 21, 29, 37)} | {
        (f'jrc-{n:02}', 'zh') for n in (4, 16, 25, 33, 38)}
    if len(selection) != 10 or {(r['task_id'], r['language']) for r in selection} != expected:
        raise ValueError('必须保持既有五英五中十题')
    if sorted(r['order'] for r in selection) != list(range(1, 11)):
        raise ValueError('冻结题序不完整')
    attempts = [read_json(p) for p in (source / 'runs/attempts').glob('*/attempt-*.json')]
    bykey = {(a['key']['task_id'], a['key']['language'], a['key']['condition']): a
             for a in attempts if a['attempt'] == 1}
    # Source corpus is fixed by the A5 suite, not inferred from model answers.
    suite = read_json(source / 'suite.json')
    corpus_path = Path(suite['domains'][0]['spec']['public_corpus'])
    corpus = read_json(corpus_path)
    pages_path = corpus_path.parent / 'pages.jsonl'
    pages = {p['page_id']: p for p in read_rows(pages_path)}
    if len(pages) != 256 or set(pages) != {p['page_id'] for p in corpus['pages']}:
        raise ValueError('A6必须使用原256页库')
    public, private = [], []
    for item in sorted(selection, key=lambda r: r['order']):
        if set(item['conditions'].values()) != {'visual-rag', 'hybrid-rag', 'hybrid-agent'}:
            raise ValueError('每题必须保持三个冻结条件')
        candidates, page_ids = [], set()
        for anonymous, condition in item['conditions'].items():
            a = bykey[(item['task_id'], item['language'], condition)]
            if a['state'] != 'finished' or digest(a['result']['answer']) != a['answer_digest']:
                raise ValueError('回答未终结或摘要错误')
            page_ids.update(a['task']['evidence_page_ids'])
            for claim in a['result']['answer']['claims']:
                page_ids.update(claim['evidence_page_ids'])
            candidates.append({'id': anonymous, 'answer_digest': a['answer_digest'],
                               'answer': a['result']['answer']})
            private.append({'id': anonymous, 'attempt': a, 'condition': condition})
        task = a['task']
        public.append({'order': item['order'], 'task_id': item['task_id'],
            'language': item['language'], 'question': task['question_' + item['language']],
            'reference_facts': task['reference_facts'], 'answers': candidates,
            'pages': [{'id': pid, 'number': pages[pid]['page_number'],
                       'document': pages[pid]['document_name'], 'source': pages[pid]['source'],
                       'image': 'pages/' + pid.replace(':', '-') + '.png'} for pid in sorted(page_ids)]})
    if len(private) != 30 or len({x['id'] for x in private}) != 30:
        raise ValueError('匿名回答数或编号错误')
    bundle_id = digest(public)
    (output / 'review/pages').mkdir(parents=True)
    for task in public:
        for page in task['pages']:
            shutil.copyfile(pages[page['id']]['preview'], output / 'review' / page['image'])
    payload = json.dumps({'bundle_id': bundle_id, 'tasks': public}, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    (output / 'review/index.html').write_text(template.replace('/*__DATA__*/', payload), encoding='utf-8')
    atomic_json(output / 'private-mapping.json', {'source': str(source), 'bundle_id': bundle_id,
                                               'selection': selection, 'answers': private})
    atomic_json(output / 'manifest.json', {'source': str(source), 'bundle_id': bundle_id,
        'answers': 30, 'logical_questions': 10, 'human': 'not_checked',
        'a5_execution_id': read_json(source / 'runs/manifest.json')['execution_id'],
        'scope': 'Human materials only; no new QA, no human results yet.'})
    return {'answers': 30, 'bundle_id': bundle_id, 'page': str(output / 'review/index.html')}


def rating(value):
    if value is None or value == 'pending':
        return None
    if type(value) not in (int, float) or value not in (0, .5, 1):
        raise ValueError('评分必须为0、0.5、1或未填写/待确认')
    return value


def convert(entry, a, reviewer_type='human'):
    if reviewer_type not in ('human', 'assistant'):
        raise ValueError('只支持实际人工或助手复核')
    if entry.get('answer_digest') != a['answer_digest']:
        raise ValueError('人工记录绑定的答案摘要不匹配')
    claims, facts = a['result']['answer']['claims'], a['task']['reference_facts']
    cr, fr = entry.get('claims', []), entry.get('facts', [])
    if len(cr) != len(claims) or len(fr) != len(facts):
        raise ValueError('声明或必需事实评分数量错误')
    fact_scores = [rating(x.get('fact')) for x in cr]
    cite_scores = [rating(x.get('citation')) for x in cr]
    coverage = [rating(x.get('score')) for x in fr] if claims else [0 for _ in facts]
    insufficiency = rating(entry.get('insufficiency'))
    if (any(x is None for x in fact_scores + cite_scores + coverage) or
            insufficiency is None or type(entry.get('language_ok')) is not bool or
            entry.get('confirmed') is not True):
        return None
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    na = {}
    if not claims:
        na.update(fact_correctness='No factual claims.', citation_support='No factual claims.')
    if not facts:
        na['answer_coverage'] = 'Frozen reference has no required facts.'
    return {'key': a['key'], 'attempt': a['attempt'], 'answer_digest': a['answer_digest'],
        'reviewer_type': reviewer_type, 'human': 'checked' if reviewer_type == 'human' else 'not_checked',
        'fact_correctness': mean(fact_scores), 'citation_support': mean(cite_scores),
        'answer_coverage': mean(coverage), 'insufficiency_handling': insufficiency,
        'language_ok': entry['language_ok'], 'language_output_is_system_fallback': False,
        'not_applicable': na, 'claim_reviews': cr, 'required_fact_reviews': fr,
        'diagnostic': entry.get('notes', '')}


def metrics(records):
    return {m: {'denominator': len(xs), 'mean': sum(xs)/len(xs) if xs else None}
            for m in RATINGS for xs in [[r[m] for r in records if r[m] is not None]]}


def summarize(root, reviews, output, reviewer_type='human'):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('报告已存在，请使用新路径')
    private, entered = read_json(root / 'private-mapping.json'), read_json(reviews)
    if entered.get('reviewer_type', 'human') != reviewer_type:
        raise ValueError('评分来源与声明不匹配；助手复核不得转为人工')
    if entered.get('bundle_id') != private['bundle_id']:
        raise ValueError('人工文件不属于本次匿名材料')
    known = {x['id']: x for x in private['answers']}
    supplied = entered.get('reviews', [])
    ids = [x['id'] for x in supplied]
    if len(set(ids)) != len(ids) or set(ids) - set(known):
        raise ValueError('重复或未知匿名编号')
    humans, pending = [], set(known)
    for entry in supplied:
        h = convert(entry, known[entry['id']]['attempt'], reviewer_type)
        if h is not None:
            humans.append(h)
            pending.remove(entry['id'])
    attempts = [x['attempt'] for x in private['answers']]
    source = Path(private['source'])
    ar = {(digest(x['key']), x['attempt']): x for p in (source / 'scoring').glob('*.json') for x in [read_json(p)]}
    assistants = [ar[(digest(a['key']), a['attempt'])] for a in attempts]
    validated = summarize_attempts(attempts, 30, humans)
    validated['semantic_quality_status'] = 'reviewed_complete' if not pending else 'reviewed_subset_only'
    summarize_attempts(attempts, 30, assistants)
    byhuman = {(digest(x['key']), x['attempt']): x for x in humans}
    differences = []
    for a in attempts:
        k = (digest(a['key']), a['attempt'])
        h, s = byhuman.get(k), ar[k]
        if h:
            differences.append({'key': a['key'], 'delta_recheck_minus_a5_assistant':
                {m: h[m]-s[m] if h[m] is not None and s[m] is not None else None for m in RATINGS},
                'recheck_notes': h['diagnostic']})
    groups = {c: metrics([h for h in humans if h['key']['condition'] == c])
              for c in ('visual-rag', 'hybrid-rag', 'hybrid-agent')}
    cells = defaultdict(dict)
    for h in humans:
        cells[(h['key']['task_id'], h['key']['language'])][h['key']['condition']] = h
    paired = {}
    for base, variant in [('visual-rag', 'hybrid-rag'), ('hybrid-rag', 'hybrid-agent')]:
        paired[variant + '-minus-' + base] = {}
        for m in RATINGS:
            xs = [cell[variant][m]-cell[base][m] for cell in cells.values()
                  if base in cell and variant in cell and cell[base][m] is not None and cell[variant][m] is not None]
            paired[variant + '-minus-' + base][m] = {'common_denominator': len(xs),
                                                    'mean_delta': sum(xs)/len(xs) if xs else None}
    report = {'status': ('complete_human_reviewed' if reviewer_type == 'human' else 'complete_assistant_rechecked') if not pending else 'partial',
        'reviewer_type': reviewer_type, 'human': 'checked' if reviewer_type == 'human' and not pending else 'not_checked',
        'bundle_id': private['bundle_id'], 'reviewed': len(humans),
        'pending_ids': sorted(pending), 'logical_questions': 10,
        'review_records': humans, 'metrics': metrics(humans), 'by_condition': groups,
        'condition_pairs': paired, 'differences_from_a5_assistant': differences,
        'a5_assistant_same_sample_metrics': metrics(assistants),
        'validated_summary': validated, 'scope': 'Ten frozen questions; not thirty independent samples. Delegated assistant recheck is not independent human validation. A5 unchanged.'}
    atomic_json(output, report)
    return {'status': report['status'], 'reviewer_type': reviewer_type, 'reviewed': len(humans), 'pending': len(pending)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', required=True)
    prep.add_argument('--output', required=True)
    summary = sub.add_parser('summarize')
    summary.add_argument('--root', required=True)
    summary.add_argument('--reviews', required=True)
    summary.add_argument('--output', required=True)
    summary.add_argument('--reviewer-type', choices=('human', 'assistant'), default='human')
    args = p.parse_args()
    result = prepare(args.source, args.output) if args.command == 'prepare' else summarize(args.root, args.reviews, args.output, args.reviewer_type)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
