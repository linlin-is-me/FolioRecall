"""Freeze A5 inputs without loading a model; bind the complete index separately."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random

from foliorecall.io import read_json, read_rows, provenance
from foliorecall.qa_storage import atomic_json, digest, output_lock
from foliorecall.qa_runtime import prepare_suite, suite_jobs
from scripts.prepare_qa_development import budget_snapshot

DEVELOPMENT = Path('outputs/extensions/A/qa-development-v1')
CORPUS = Path('outputs/extensions/A/data/jrc-corpus-v1')
TASKS = Path('outputs/extensions/A/data/jrc-tasks-v2')
INDEX = Path('outputs/extensions/A/indexes/jrc-original')
CONDITIONS = ['visual-rag', 'hybrid-rag', 'hybrid-agent']
CAP = 62.379233


def file_binding(paths):
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def check_files(bindings):
    if file_binding(bindings) != bindings:
        raise ValueError('冻结来源内容发生变化；禁止初始化模型或调用API')


def ordered_keys(suite_id):
    return [dict(suite=suite_id, condition=c, domain='jrc', task_id=f'jrc-{i:02d}', language=lang, repeat=0)
            for i in range(1, 41) for lang in ('en', 'zh') for c in CONDITIONS]


def human_selection():
    english = {'jrc-01', 'jrc-11', 'jrc-21', 'jrc-29', 'jrc-37'}
    tasks = sorted(english | {'jrc-04', 'jrc-16', 'jrc-25', 'jrc-33', 'jrc-38'})
    rng = random.Random(42)
    rng.shuffle(tasks)
    rows = []
    for number, task in enumerate(tasks, 1):
        conditions = list(CONDITIONS)
        rng.shuffle(conditions)
        rows.append({'task_id': task, 'language': 'en' if task in english else 'zh',
                     'order': number, 'conditions': {f'{number:02d}-{chr(65+i)}': c for i, c in enumerate(conditions)}})
    return {'seed': 42, 'fixed_before_answers': True, 'human': 'not_checked', 'selection': rows,
            'scope': 'Private mapping only; no answers or human review materials created.'}


def prepare(root):
    root = Path(root).resolve()
    with output_lock(root):
        if root.exists():
            raise ValueError('A5目录已存在，禁止重置冻结记录或预算')
        pages = read_rows(CORPUS/'pages.jsonl')
        tasks = read_rows(TASKS/'tasks.jsonl')
        status = read_json(TASKS/'status.json')
        from foliorecall.retrieval import page_identity
        if len(pages) != 256 or len({p['page_id'] for p in pages}) != 256 or page_identity(pages) != status['corpus_pages']:
            raise ValueError('JRC完整256页冻结身份不匹配')
        if [t['task_id'] for t in tasks] != [f'jrc-{i:02d}' for i in range(1, 41)] or status.get('split') != 'acceptance':
            raise ValueError('JRC40条验收任务或顺序不匹配')
        if Counter(t['category'] for t in tasks) != dict(single_page=10, multi_page=10, cross_document=8, numeric_chart=8, insufficient=4):
            raise ValueError('冻结题型数量变化')
        agent = read_json(DEVELOPMENT/'agent.json')
        old_budget = read_json(DEVELOPMENT/'budget-before.json')
        if agent['phase_budget_cny'] != CAP or old_budget['absolute_cap_cny'] != CAP or agent['ledger'] != old_budget['ledger']:
            raise ValueError('原预算或账本不匹配')
        agent['final_answer_evidence_check'] = False
        before = budget_snapshot(agent['ledger'])
        before.update(absolute_cap_cny=CAP, do_not_rebase_on_resume=True, minimum_request_space_cny=1.06)
        provenance(root/'preparation', {'scope': 'A5 full JRC index and one acceptance round; no A6 execution',
                    'source_commit': '8b937b4', 'fixed_budget_cap': CAP})
        atomic_json(root/'agent.json', agent)
        atomic_json(root/'budget-before.json', before)
        suite = {'suite_id': 'qa-acceptance-v1', 'conditions': CONDITIONS, 'repeats': 1,
            'inter_task_cooldown_seconds': 65, 'order': 'task, en/zh, visual-rag/hybrid-rag/hybrid-agent',
            'common': {'config': 'configs/baseline.json', 'query_config': 'configs/student-ml-cpu.json',
                       'agent_config': str(root/'agent.json'), 'probe': 'outputs/extensions/A/api-probe/20260919-run1/result.json'},
            'domains': [{'name': 'jrc', 'spec': {'index': str(INDEX.resolve()), 'tasks': str((TASKS/'tasks.jsonl').resolve()),
                        'public_corpus': str((CORPUS/'public-corpus.json').resolve()),
                        'retrieval_config': str((CORPUS/'hybrid.json').resolve())}}]}
        suite['job_order'] = ordered_keys(suite['suite_id'])
        atomic_json(root/'suite.json', suite)
        selection = human_selection()
        if set(x['task_id'] for x in selection['selection']) != set(status['human_review_selection']):
            raise ValueError('人工预选与冻结十题不符')
        atomic_json(root/'a6-private-mapping.json', selection)
        gpu_root = root/'gpu-budget'
        atomic_json(gpu_root/'budget.json', {'limit_seconds': 7200, 'window_seconds': 7200, 'reserve_seconds': 0,
            'minimum_free_mib': 6144, 'maximum_idle_utilization': 10, 'idle_samples': 3,
            'authorization': 'User approved one continuous two-hour local GPU window for JRC indexing only.'})
        paths = [CORPUS/'pages.jsonl', CORPUS/'texts.jsonl', CORPUS/'public-corpus.json', CORPUS/'hybrid.json',
                 CORPUS/'text-index/index.json', TASKS/'tasks.jsonl', TASKS/'status.json',
                 Path('configs/baseline.json'), Path('configs/student-ml-cpu.json'), DEVELOPMENT/'agent.json',
                 DEVELOPMENT/'budget-before.json', Path(suite['common']['probe']), root/'agent.json', root/'suite.json',
                 root/'a6-private-mapping.json', root/'budget-before.json',
                 Path('scripts/prepare_qa_acceptance.py'), Path('scripts/run_qa_acceptance_segment.py'),
                 Path('scripts/report_qa_acceptance.py'), Path('scripts/cache_qa_translations.py')]
        atomic_json(root/'protocol.json', {'sources': file_binding(paths), 'batches': [suite['job_order'][i:i+48] for i in range(0, 240, 48)],
            'reviewer_type': 'assistant', 'human': 'not_checked', 'frozen_task_files_unchanged': True,
            'coverage_null_tasks': ['jrc-37', 'jrc-40'], 'coverage_numeric_insufficient_tasks': ['jrc-38', 'jrc-39'],
            'primary_uses_first_attempts': True, 'no_post_hoc_selection': True,
            'smoke_queries': ['European electricity market integration', 'European energy system investment needs', 'Energy infrastructure stakeholder survey']})
        return {'prepared': str(root), 'planned': 240, 'bound_index': False}


def bind_index(root):
    root = Path(root)
    with output_lock(root):
        check_files(read_json(root/'protocol.json')['sources'])
        if (root/'index-binding.json').exists():
            raise ValueError('索引身份已冻结，不覆盖')
        units, identity = prepare_suite(root/'suite.json')
        jobs = suite_jobs(units, identity['suite'])
        if len(jobs) != 240 or any(len(u['identity']['status']['corpus_pages']) != 256 for u in units):
            raise ValueError('完整索引与240项验收不匹配')
        files = [INDEX/n for n in ('index.faiss', 'pages.jsonl', 'config.json')]
        atomic_json(root/'index-binding.json', {'identity_digest_without_cache': digest(identity),
            'files': file_binding(files), 'pages': 256, 'jobs': 240})
        return {'bound_index': True, 'jobs': len(jobs), 'identity_digest_without_cache': digest(identity)}


def validate_root(root, cached=True):
    root = Path(root)
    protocol = read_json(root/'protocol.json')
    check_files(protocol['sources'])
    binding = read_json(root/'index-binding.json')
    check_files(binding['files'])
    units, identity = prepare_suite(root/'suite.json')
    if digest(identity) != binding['identity_digest_without_cache']:
        raise ValueError('验收编码、任务、配置或执行代码身份变化')
    if cached:
        units, identity = prepare_suite(root/'suite.json', root/'translations/translations.json')
    jobs = suite_jobs(units, identity['suite'])
    if [j['key'] for j in jobs] != [k for batch in protocol['batches'] for k in batch] or len(jobs) != 240:
        raise ValueError('验收批次与完整运行键不匹配')
    if any(u['agent'].get('final_answer_evidence_check') is not False or u['student']['device'] != 'cpu' for u in units):
        raise ValueError('A5必须保持CPU基线问答配置')
    return units, identity, jobs, protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--bind-index', action='store_true')
    a = p.parse_args()
    print(bind_index(a.output) if a.bind_index else prepare(a.output))
