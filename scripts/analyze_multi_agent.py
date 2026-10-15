"""Read-only quality, resource and client-concurrency analysis of the fixed comparison."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_evaluation import RATINGS
from scripts.run_multi_agent_experiment import ROOT, attempts, report


def metrics(reviews):
    return {m:{'denominator':len(values),'mean':statistics.mean(values) if values else None}
            for m in RATINGS for values in [[r[m] for r in reviews if r[m] is not None]]}


def analyze(root, final=False):
    root = Path(root)
    base = report(root,final)
    rows = attempts(root)
    first = [a for a in rows if a['state']=='finished' and a['attempt']==1]
    reviews = [read_json(p) for p in sorted((root/'scoring').glob('*.json'))]
    ratings = {(digest(r['key']),r['attempt']):r for r in reviews}
    groups = defaultdict(list)
    for a in first:
        stage,arm = a['key']['suite'].removeprefix('multi-agent-v1-').rsplit('-',1)
        groups[(stage,arm,'all','all')].append(a)
        groups[(stage,arm,a['key']['language'],'all')].append(a)
        groups[(stage,arm,'all',a['task']['category'])].append(a)
        if stage != 'diagnostic':
            groups[('main',arm,'all','all')].append(a)
            groups[('main',arm,a['key']['language'],'all')].append(a)
            groups[('main',arm,'all',a['task']['category'])].append(a)
    by_group = []
    for (stage,arm,language,category),selected in sorted(groups.items()):
        rs = [ratings[(digest(a['key']),1)] for a in selected if (digest(a['key']),1) in ratings]
        calls = [c for a in selected for c in a['result']['api_calls'] if c.get('cost_status')!='not_sent']
        by_group.append({'stage':stage,'arm':arm,'language':language,'category':category,
            'first_terminal':len(selected),'structure_valid':sum(a['result']['completed'] for a in selected),
            'reviewed':len(rs),'language_assessed':sum(not r['language_output_is_system_fallback'] for r in rs),
            'language_ok':sum(r['language_ok'] and not r['language_output_is_system_fallback'] for r in rs),
            'completed_with_insufficiency':sum(a['result']['completed'] and a['result']['answer']['insufficient_evidence'] for a in selected),
            'legitimate_refusals':sum(a['result']['completed'] and not a['result']['answer']['claims']
                                     and a['result']['answer']['insufficient_evidence'] for a in selected),
            'system_or_structure_failures':sum(not a['result']['completed'] for a in selected),
            'missing_reviews':len(selected)-len(rs),
            'not_applicable':{m:sum(r[m] is None for r in rs) for m in RATINGS},
            'empty_answers':sum(not a['result']['answer']['claims'] for a in selected),
            'termination':dict(Counter(a['result']['termination_reason'] for a in selected)),
            'metrics':metrics(rs),
            'actual_requests':sum(a['result']['actual_model_requests'] for a in selected),
            'actual_searches':sum(a['result']['actual_searches'] for a in selected),
            'known_cost_cny':sum(a['result']['estimated_cost_cny'] for a in selected),
            'known_input_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) for c in calls),
            'known_output_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) for c in calls),
            'calls_missing_usage':sum(c.get('usage') is None for c in calls),
            'active_seconds_mean':statistics.mean(a['result']['timing']['task_seconds'] for a in selected),
            'active_seconds_total':sum(a['result']['timing']['task_seconds'] for a in selected),
            'loading_seconds_total':sum(a['result'].get('loading_seconds',0.) for a in selected)})
    pairs = {}
    for name,values in base['pairs'].items():
        pairs[name] = {'paired_first_attempts':sum(v['paired_reviewed'] for v in values),
            'metrics':{m:{'denominator':len(ds),'mean_delta':statistics.mean(ds) if ds else None,
                         'improved':sum(v>0 for v in ds),'degraded':sum(v<0 for v in ds),'tied':sum(v==0 for v in ds)}
                for m in RATINGS for ds in [[v['score_delta'][m] for v in values
                    if v['paired_reviewed'] and v['score_delta'][m] is not None]]}}
    source_access = []
    corrections = read_json(root/'review-diagnosis-corrections.json') if (root/'review-diagnosis-corrections.json').exists() else {}
    for a in first:
        result = a['result']
        retrieved = {p['page_id'].rsplit(':',1)[0] for t in result['trace']
                     if t.get('tool')=='search_pages' and t.get('status')=='ok' for p in t['result']}
        read = {p['page_id'].rsplit(':',1)[0] for p in result['evidence']}
        graph = root/('diagnostic' if 'diagnostic' in a['key']['suite'] else 'main')/'attempts'/digest(a['key'])/'graph'
        final_pages = read_json(graph/'final-evidence.json') if (graph/'final-evidence.json').exists() else None
        operations = [read_json(p) for p in (graph/'operations').glob('*.json')]
        gaps = [op for op in operations if op.get('phase')=='gap']
        gap_requests = [op for op in gaps if op['kind']=='request' and op.get('outcome',{}).get('record',{}).get('cost_status')!='not_sent']
        merge_events = [e for e in result.get('progress',[]) if e.get('event')=='evidence_merge' and e.get('status')=='completed']
        correction = corrections.get(a['key']['suite']+'/'+a['key']['task_id'])
        original = ratings.get((digest(a['key']),1),{}).get('diagnosis')
        source_access.append({'key':a['key'],'retrieved_doc_ids':sorted(retrieved),'read_doc_ids':sorted(read),
            'final_evidence':final_pages,'workers':len(result.get('graph',{}).get('workers',{})),
            'gap_used_raw':result.get('graph',{}).get('gap_used'),
            'gap_requested':bool(merge_events and merge_events[-1].get('gap') is not None),
            'gap_actual_requests':len(gap_requests),
            'gap_actual_searches':sum(op.get('tool')=='search_pages' for op in gaps),
            'gap_successful_read_ids':sorted({t['arguments']['page_id'] for op in gaps for t in op.get('outcome',{}).get('trace',[])
                                             if t.get('tool')=='read_page' and t.get('status')=='ok'}),
            'repeated_searches':sum(t.get('repeated',False) for t in result['trace'] if t.get('tool')=='search_pages'),
            'duplicate_read_outcomes':sum(t.get('status') in ('already_read','shared_cache') for t in result['trace'] if t.get('tool')=='read_page'),
            'review_diagnosis':correction or original,
            'original_review_diagnosis':original if correction else None})
    diagnostic = []
    for task in ('cov-01','cov-03','cov-05','cov-09'):
        selected = [a for a in first if a['key']['task_id']==task and 'diagnostic' in a['key']['suite']]
        arms = {a['key']['suite'].rsplit('-',1)[1]:a for a in selected}
        if set(arms)!= {'serial','C'}:
            continue
        serial,parallel = arms['serial']['result'],arms['C']['result']
        plans = {}
        for arm,a in arms.items():
            op = root/'diagnostic/attempts'/digest(a['key'])/'graph/operations/coordinate-plan-attempt1.json'
            plans[arm] = read_json(op).get('outcome',{}).get('message') if op.exists() else None
        diagnostic.append({'task':task,'serial_seconds':serial['timing']['task_seconds'],
            'parallel_seconds':parallel['timing']['task_seconds'],
            'parallel_minus_serial_seconds':parallel['timing']['task_seconds']-serial['timing']['task_seconds'],
            'same_plan_response':plans['serial']==plans['C'],
            'serial_requests':serial['actual_model_requests'],'parallel_requests':parallel['actual_model_requests'],
            'serial_pages':[p['page_id'] for p in serial['evidence']],
            'parallel_pages':[p['page_id'] for p in parallel['evidence']],
            'serial_answer':serial['answer'],'parallel_answer':parallel['answer']})
    segments = [read_json(p) for name in ('main','diagnostic') for p in (root/name/'segments').glob('*/result.json')]
    initial = read_json(root/'diagnostic-start-cooldown.json') if (root/'diagnostic-start-cooldown.json').exists() else {}
    browser_startups = [dict(directory=str(p.parent), **read_json(p))
                        for p in sorted((root/'browser').glob('*/startup.json'))]
    browser_turns = []
    for p in sorted((root/'browser').glob('*/qa-*.json')):
        value = read_json(p)
        browser_turns.append({'path':str(p),'conversation_id':value.get('conversation_id'),
            'turn_index':value.get('turn_index'),'question':value['question'],
            'timing':value['timing'],'answer_digest':digest(value['answer']),
            'matches_registered_answer':any(a['key']['suite']=='multi-agent-v1-confirmation-C'
                and a['task']['question']==value['question'] and a['answer_digest']==digest(value['answer'])
                for a in first)})
    return {'status':base['status'],'base_report':base,'groups':by_group,'paired_quality':pairs,
            'source_access':source_access,'concurrency':diagnostic,
            'browser_startups':browser_startups,'browser_turns':browser_turns,
            'recorded_cooldown_seconds':sum(s.get('cooldown_seconds',0.) for s in segments),
            'diagnostic_initial_cooldown_seconds':initial.get('elapsed_seconds',0.),
            'timing_scope':'Parent activity is elapsed time; per-call seconds may overlap. Loading and cooldown separate; manual waiting unknown.',
            'quality_scope':'Assistant source review; human=not_checked. Seen development tasks, 18 main turns / 15 sessions; no independent acceptance claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('分析输出已存在，另存新文件')
    value = analyze(args.root,args.final)
    atomic_json(args.output,value)
    print(value['status'],len(value['groups']),len(value['source_access']))


if __name__=='__main__':
    main()
