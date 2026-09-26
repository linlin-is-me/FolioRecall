"""Read-only aggregation across an audited source and its independent continuation."""
import argparse
import json
from pathlib import Path

from foliorecall.io import read_json, read_rows
from foliorecall.qa_storage import digest, atomic_json
from foliorecall.qa_runtime import ledger_snapshot
from foliorecall.qa_evaluation import summarize_attempts, RATINGS
from scripts.analyze_qa_development import analyze, load_run
from scripts.qa_continuation import validate_source, require, derive


def load_combined(descriptor, target):
    parent, old = validate_source(descriptor)
    child, new = load_run(target)
    # Derive from preserved identity, not a newly loaded model configuration.
    expected, _, _ = derive(descriptor, parent['identity'],
                            [{'key':k} for k in parent['job_keys']], target)
    require(child['identity'] == expected and child['identity_digest'] == digest(expected), 'Continuation identity mismatch')
    require(child['execution_id'] != parent['execution_id'], 'Continuation must have a new execution ID')
    require(child['job_keys'] == descriptor['pending_keys'], 'Continuation plan mismatch')
    permitted = {digest(k) for k in child['job_keys']}
    for a in new:
        require(digest(a['key']) in permitted, 'Continuation attempt outside allowed subset')
    rows = old+new
    require(len({(digest(a['key']),a['attempt']) for a in rows}) == len(rows), 'Duplicate source attempt')
    configs = [u['agent'] for u in parent['identity']['units']]
    ledger_rows, phase = {}, {}
    sources = []
    for manifest, attempts, path in ((parent,old,descriptor['source_directory']), (child,new,str(target))):
        ledger = ledger_snapshot(configs, manifest['execution_id'])
        phase.update(ledger['phase_total_including_other_runs'])
        for row in ledger['calls']:
            require(row['id'] not in ledger_rows, 'Ledger ID belongs to multiple executions')
            ledger_rows[row['id']] = row
        sources.extend(dict(key=a['key'], attempt=a['attempt'], execution_id=manifest['execution_id'],
                            directory=str(path)) for a in attempts)
        # Every charge must belong to a recorded attempt, including unknown crashes.
        recorded = {(digest(a['key']),a['attempt']):a for a in attempts}
        linked = set()
        for a in attempts:
            if a['state'] != 'finished':
                continue
            for call in a['result']['api_calls']:
                if call.get('cost_status') == 'not_sent':
                    require(not call.get('ledger_id') and call.get('cost_cny',0) == 0, 'Unsent call charged')
                    continue
                lid = call['ledger_id']
                require(lid in ledger_rows and lid not in linked, 'Missing/duplicate call ledger ID')
                r = ledger_rows[lid]
                ctx = json.loads(r['context'])
                require(ctx.get('execution_id') == manifest['execution_id'] and ctx.get('attempt') == a['attempt']
                        and all(ctx.get(k) == v for k,v in a['key'].items()), 'Call identity mismatch')
                known = call.get('usage') is not None
                require(r['status'] == ('estimated_from_usage' if known else 'reserved_unknown_charge'), 'Charge status mismatch')
                require(abs(r['cny']-(call['cost_cny'] if known else call['reservation_cny'])) < 1e-8, 'Charge amount mismatch')
                if known:
                    require(json.loads(r['usage']) == call['usage'], 'Usage mismatch')
                linked.add(lid)
        for r in ledger['calls']:
            c = json.loads(r['context'])
            key = {k:c[k] for k in parent['job_keys'][0]}
            a = recorded.get((digest(key),c['attempt']))
            require(a is not None, 'Unattributed charge')
            require(r['id'] in linked or a['state'] == 'running', 'Terminal charge not linked to API record')
    return parent, rows, list(ledger_rows.values()), sources, phase


def summarize(descriptor, target, reviews):
    manifest, attempts, ledger, sources, phase = load_combined(descriptor, target)
    report = analyze(manifest, attempts, reviews)
    report['summary'] = summarize_attempts(attempts,len(manifest['job_keys']),reviews,manifest['job_keys'],ledger)
    report['candidate_target'] = None
    terminals = {(digest(a['key']),a['attempt']) for a in attempts if a['state']=='finished'}
    reviewed = {(digest(r['key']),r['attempt']) for r in reviews}
    first = [a for a in attempts if a['attempt']==1]
    unknown = [a['key'] for a in attempts if a['state']=='running']
    original_unknown = [e['key'] for e in descriptor['excluded']]
    report['completion'] = dict(planned=len(manifest['job_keys']), first_started=len(first),
        first_terminal=sum(a['state']=='finished' for a in first),
        structurally_completed=sum(a['state']=='finished' and a['result']['completed'] for a in first),
        unknown_keys=unknown, not_started=len(manifest['job_keys'])-len(first),
        all_available_terminal_reviewed=terminals==reviewed,
        accepted_declared_missing_only=unknown==original_unknown,
        status='partial')
    if len(first)==len(manifest['job_keys']) and terminals==reviewed and unknown==original_unknown:
        report['completion']['status']='reviewed_with_declared_missing'
    report['ledger'] = dict(calls=ledger, phase_total_including_other_runs=phase,
        known_cny=sum(r['cny'] for r in ledger if r['status']=='estimated_from_usage'),
        unknown_reserved_cny=sum(r['cny'] for r in ledger if r['status']=='reserved_unknown_charge'))
    report['sources'] = sources
    paired = {}
    for base, variant in [('visual-rag','hybrid-rag'),('hybrid-rag','hybrid-agent')]:
        items=[p for p in report['condition_pairs'] if p['base']==base and p['variant']==variant and p['cell'][-1]==1]
        paired[base+'/'+variant] = dict(planned=len(manifest['job_keys'])//3,
            paired_reviewed=sum(p['paired_reviewed'] for p in items), metrics={})
        for metric in RATINGS:
            values=[p['score_delta'][metric] for p in items if p['paired_reviewed'] and p['score_delta'][metric] is not None]
            paired[base+'/'+variant]['metrics'][metric]=dict(n=len(values),mean_delta=sum(values)/len(values) if values else None)
    report['paired_comparisons']=paired
    report['scope']='Original fixed plan; unknown is not failure or zero score. Terminal-only ratings show denominators; paired comparisons use common observations. Assistant review, not human gold. No stable P95 or candidate selection.'
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--continuation',required=True)
    p.add_argument('--runs',required=True)
    p.add_argument('--reviews',nargs='+',required=True,help='JSONL files or directories containing individual review JSON')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    require(not Path(a.output).exists(),'Output already exists')
    reviews=[]
    for path in map(Path,a.reviews):
        reviews.extend([read_json(f) for f in sorted(path.glob('*.json'))] if path.is_dir() else read_rows(path))
    result=summarize(read_json(a.continuation),a.runs,reviews)
    atomic_json(a.output,result)
    print(result['completion'])


if __name__=='__main__':
    main()
