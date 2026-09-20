"""Offline A4 error counts, paired candidates and three-round variation. No judge API."""
import argparse
from collections import defaultdict
from pathlib import Path
from foliorecall.io import read_json, read_rows
from foliorecall.qa_evaluation import RATINGS, summarize_attempts
from foliorecall.qa_storage import atomic_json, digest

ERROR_ORDER = ('query_translation', 'not_recalled', 'fixed_selection', 'tool_decision',
               'page_reading', 'evidence_use')


def load_run(path):
    root = Path(path)
    return read_json(root/'manifest.json'), [read_json(p) for p in sorted(root.glob('attempts/*/attempt-*.json'))]


def cell(key):
    return tuple(key[k] for k in ('condition', 'domain', 'task_id', 'language', 'repeat'))


def analyze(manifest, attempts, reviews):
    summary = summarize_attempts(attempts, len(manifest['job_keys']), reviews, manifest['job_keys'])
    rows = {(digest(a['key']), a['attempt']): a for a in attempts}
    accepted = {}
    errors = {kind: {'answers': 0, 'tasks': set(), 'coverage_loss': 0.} for kind in (*ERROR_ORDER, 'unknown', 'none')}
    for r in reviews:
        if r['reviewer_type'] != 'assistant' or r.get('human') != 'not_checked':
            raise ValueError('A4 助手评审与人工状态必须明确分开')
        primary = r.get('primary_error')
        if primary not in errors or not r.get('primary_error_reason'):
            raise ValueError('每回答必须有一个明确主因及依据；无法判断用 unknown')
        secondary = r.get('secondary_factors', [])
        if len(set(secondary)) != len(secondary) or any(k not in ERROR_ORDER for k in secondary):
            raise ValueError('次要因素须去重且属于约定类别')
        key = (digest(r['key']), r['attempt'])
        accepted[key] = r
        if r['attempt'] == 1:
            item = errors[primary]
            item['answers'] += 1
            item['tasks'].add((r['key']['domain'], r['key']['task_id']))
            task = rows[key].get('task', {})
            if r['answer_coverage'] is not None:
                item['coverage_loss'] += len(task.get('reference_facts', []))*(1-r['answer_coverage'])
    errors = {k: dict(v, logical_tasks=len(v['tasks']), tasks=sorted(v['tasks'])) for k,v in errors.items()}
    all_reviewed = (summary['pending'] == 0 and len(accepted) == len(attempts)
                    and all(a['state'] == 'finished' for a in attempts))
    ordered = sorted(ERROR_ORDER, key=lambda k: (-errors[k]['answers'], -errors[k]['logical_tasks'],
                                                -errors[k]['coverage_loss'], ERROR_ORDER.index(k)))
    selection = None
    if all_reviewed and errors[ordered[0]]['answers']:
        selection = {'primary_category': ordered[0], 'status': 'specification_required_not_adopted',
                     'counts': errors[ordered[0]], 'unknown_answers': errors['unknown']['answers']}
    planned_groups = defaultdict(list)
    for key in manifest['job_keys']:
        planned_groups[cell(key)[:-1]].append(key)
    variability = []
    for group, keys in planned_groups.items():
        if len(keys) != 3:
            continue
        observations = []
        for key in sorted(keys, key=lambda k: k['repeat']):
            a = rows.get((digest(key), 1))
            r = accepted.get((digest(key), 1))
            result = a.get('result', {}) if a else {}
            observations.append({'key': key, 'state': a['state'] if a else 'pending',
                'completed': result.get('completed'), 'answer_digest': a.get('answer_digest') if a else None,
                'citation_page_ids': sorted({p for claim in result.get('answer', {}).get('claims', []) for p in claim.get('evidence_page_ids', [])}),
                'requests': result.get('model_requests'), 'active_seconds': result.get('timing', {}).get('task_seconds'),
                'ratings': {m: r[m] for m in RATINGS} if r else None})
        complete = all(o['state'] == 'finished' and o['ratings'] is not None for o in observations)
        ranges = {}
        for metric in RATINGS:
            values = [o['ratings'][metric] for o in observations] if complete else []
            ranges[metric] = [min(values), max(values)] if values and all(v is not None for v in values) else None
        variability.append({'cell': group, 'observations': observations, 'all_reviewed': complete, 'ranges': ranges})
    return {'summary': summary, 'all_attempts_reviewed': all_reviewed, 'primary_errors': errors,
            'candidate_target': selection, 'three_round_variation': variability,
            'scope': 'Assistant reviews; first attempts primary. Three observations do not establish a stable P95.'}


def paired_candidates(base, candidate, base_reviews, candidate_reviews):
    def indexed(attempts, reviews):
        lookup = {(digest(r['key']), r['attempt']): r for r in reviews}
        result = {}
        for a in attempts:
            if a['state'] != 'finished':
                continue
            key = (cell(a['key']), a['attempt'])
            if key in result:
                raise ValueError('配对运行键重复')
            result[key] = (a, lookup.get((digest(a['key']), a['attempt'])))
        return result
    b, c = indexed(base, base_reviews), indexed(candidate, candidate_reviews)
    output = []
    for key, (ca, cr) in c.items():
        ba, br = b.get(key, (None, None))
        if ba and digest(ba.get('task')) != digest(ca.get('task')):
            raise ValueError('配对任务参考发生变化')
        output.append({'cell': key[0], 'attempt': key[1], 'paired_reviewed': bool(br and cr),
            'score_delta': {m: cr[m]-br[m] if cr[m] is not None and br[m] is not None else None for m in RATINGS} if br and cr else None,
            'cost_delta_cny': ca['result'].get('estimated_cost_cny', 0)-ba['result'].get('estimated_cost_cny', 0) if ba else None,
            'active_seconds_delta': ca['result']['timing']['task_seconds']-ba['result']['timing']['task_seconds'] if ba else None,
            'request_delta': ca['result']['model_requests']-ba['result']['model_requests'] if ba else None})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', required=True)
    p.add_argument('--reviews', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline-runs')
    p.add_argument('--baseline-reviews')
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError('分析结果已存在；不得覆盖')
    manifest, attempts = load_run(a.runs)
    reviews = read_rows(a.reviews)
    report = analyze(manifest, attempts, reviews)
    if a.baseline_runs or a.baseline_reviews:
        if not a.baseline_runs or not a.baseline_reviews:
            raise ValueError('配对需要基线运行与评审')
        bm, ba = load_run(a.baseline_runs)
        br = read_rows(a.baseline_reviews)
        analyze(bm, ba, br)  # Apply the same binding checks to both conditions.
        report['paired_candidate'] = paired_candidates(ba, attempts, br, reviews)
    atomic_json(a.output, report)


if __name__ == '__main__':
    main()
