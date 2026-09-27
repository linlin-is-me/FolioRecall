"""Read-only first-attempt pairing, preserving recovery costs and null ratings."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from foliorecall.qa_storage import atomic_json,digest
from foliorecall.qa_evaluation import RATINGS
from foliorecall.qa_runtime import ledger_snapshot
from scripts.analyze_qa_development import load_run,analyze,paired_candidates,cell

def mean(values):return sum(values)/len(values) if values else None
def group_summary(rows, base, candidate, breview, creview):
    result={'paired_first_attempts':len(rows),'metrics':{}}
    for metric in RATINGS:
        common=[p for p in rows if p['score_delta'][metric] is not None]
        delta=[p['score_delta'][metric] for p in common]
        result['metrics'][metric]={'paired_denominator':len(common),'baseline_mean':mean([breview[tuple(p['cell'])][metric] for p in common]),
            'candidate_mean':mean([creview[tuple(p['cell'])][metric] for p in common]),
            'delta_mean':mean(delta),'improved':sum(v>0 for v in delta),'regressed':sum(v<0 for v in delta),'tied':sum(v==0 for v in delta),
            'baseline_non_null':sum(breview[tuple(p['cell'])][metric] is not None for p in rows),
            'candidate_non_null':sum(creview[tuple(p['cell'])][metric] is not None for p in rows)}
    for label,lookups,field in [('structure_valid',(base,candidate),'completed'),('language_ok',(breview,creview),'language_ok')]:
        vals=[]
        for p in rows:
            k=tuple(p['cell']);a,b=(m[k] for m in lookups)
            if label=='structure_valid':a,b=a['result'],b['result']
            vals.append((None if label=='language_ok' and a.get('language_output_is_system_fallback') else a.get(field),
                         None if label=='language_ok' and b.get('language_output_is_system_fallback') else b.get(field)))
        result[label]={'baseline':sum(a is True for a,b in vals),'candidate':sum(b is True for a,b in vals),
            'baseline_assessable':sum(a is not None for a,b in vals),'candidate_assessable':sum(b is not None for a,b in vals),
            'common_assessable':sum(a is not None and b is not None for a,b in vals),
            'improved':sum(a is False and b is True for a,b in vals),'regressed':sum(a is True and b is False for a,b in vals)}
    result['resources']={}
    for label,field in [('requests','actual_model_requests'),('known_cost_cny','estimated_cost_cny'),('active_seconds','task_seconds')]:
        totals=[]
        for mapping in (base,candidate):
            values=[]
            for p in rows:
                r=mapping[tuple(p['cell'])]['result']
                values.append(r['timing'][field] if label=='active_seconds' else r[field])
            totals.append(sum(values))
        result['resources'][label]={'baseline_total':totals[0],'candidate_total':totals[1],'delta':totals[1]-totals[0]}
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='outputs/extensions/A/qa-development-v1')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    R=Path(args.root)
    target=Path(args.output)
    if target.exists():raise ValueError('报告已存在；不得覆盖')
    read=lambda p:json.loads(p.read_text())
    bm,ba=load_run(R/'baseline');cm,ca=load_run(R/'candidate-v1')
    br=[read(p) for p in sorted((R/'scoring/baseline').glob('*.json'))]
    cr=[read(p) for p in sorted((R/'scoring/candidate-v1').glob('*.json'))]
    bs=analyze(bm,ba,br);cs=analyze(cm,ca,cr)
    assert bs['all_attempts_reviewed'] and cs['all_attempts_reviewed']
    assert len(cm['job_keys'])==240
    assert len([a for a in ca if a['attempt']==1 and a['state']=='finished'])==240
    assert sum(a['result']['translation_status']=='cached_quality_only' for a in ca if a['attempt']==1)==120
    pairs=paired_candidates(ba,ca,br,cr)
    first=[p for p in pairs if p['attempt']==1]
    assert len(first)==240 and all(p['paired_reviewed'] for p in first)
    lookup=lambda rows:{cell(a['key']):a for a in rows if a['attempt']==1}
    base,candidate,breview,creview=map(lookup,(ba,ca,br,cr))

    same_evidence = defaultdict(list)
    for pair in first:
        key = tuple(pair['cell'])
        pair['baseline_ratings'] = {m: breview[key][m] for m in RATINGS}
        pair['candidate_ratings'] = {m: creview[key][m] for m in RATINGS}
        pair['baseline_structure_valid'] = base[key]['result']['completed']
        pair['candidate_structure_valid'] = candidate[key]['result']['completed']
        pair['baseline_primary_error'] = breview[key]['primary_error']
        pair['candidate_primary_error'] = creview[key]['primary_error']
        pair['candidate_diagnosis'] = creview[key]['primary_error_reason']
        pages = [[p['page_id'] for p in mapping[key]['result']['evidence']]
                 for mapping in (base, candidate)]
        pair['same_read_page_sequence'] = pages[0] == pages[1]
        same_evidence[key[0]].append(pair['same_read_page_sequence'])
    groups=defaultdict(list)
    for p in first:
        k=p['cell']
        groups['all'].append(p)
        for label,value in [('condition',k[0]),('domain',k[1]),('language',k[3]),
                            ('condition/domain/language','/'.join((k[0],k[1],k[3])))]:
            groups[label+'/'+value].append(p)
    config=read(R/'candidate-agent.json')
    ledger=ledger_snapshot([config],cm['execution_id'])
    recorded_call_ids={c['ledger_id'] for a in ca for c in a['result']['api_calls'] if c.get('ledger_id')}
    assert recorded_call_ids==set(ledger['ledger_ids'])
    assert abs(sum(a['result']['estimated_cost_cny'] for a in ca)-ledger['known_cost_cny'])<1e-8
    assert all(a['result']['budget_state']['budget_snapshot_complete'] for a in ca)
    assert all(a['result']['translation_status'] in ('not_needed','cached_quality_only') for a in ca)

    report={'status':'complete_assistant_reviewed_not_adopted','baseline_execution_id':bm['execution_id'],
     'candidate_execution_id':cm['execution_id'],'candidate_identity_digest':cm['identity_digest'],
     'baseline_identity_digest':bm['identity_digest'],'groups':{k:group_summary(v, base, candidate, breview, creview) for k,v in groups.items()},
     'first_attempt_pairs':first,'recovery_pairs_separate':[p for p in pairs if p['attempt']==2],
     'read_evidence_comparability':{k:{'paired':len(v),'same_page_sequence':sum(v)} for k,v in same_evidence.items()},
     'baseline_summary':bs['summary'],'candidate_summary':cs['summary'],'candidate_ledger':ledger,
     'regression_cases':[p for p in first if any(v is not None and v<0 for v in p['score_delta'].values())],
     'scope':['Assistant review only; human=not_checked.','Retrospective development, not independent acceptance.',
     'One fixed candidate; no post-hoc numeric adoption threshold or further prompt search.',
     'Translation cache shares are attribution only; no duplicate ledger debit.','Default remains baseline.']}

    # Retain the original score series and show the limited calibration separately.
    audit=read(R/'paired-review-calibration-v1.json')
    for e in audit['entries']:
        mapping=breview if e['run']=='baseline' else creview
        key=cell(e['key'])
        assert digest(mapping[key])==e['original_review_digest']
        mapping[key]=e['calibrated']
    calibrated=paired_candidates(ba,ca,list(breview.values()),list(creview.values()))
    calibrated=[p for p in calibrated if p['attempt']==1]
    cgroups=defaultdict(list)
    for p in calibrated:
        k=p['cell'];cgroups['all'].append(p)
        for label,value in [('condition',k[0]),('domain',k[1]),('language',k[3])]:cgroups[label+'/'+value].append(p)
    report['calibration_sensitivity']={'artifact':str(R/'paired-review-calibration-v1.json'),'scope':audit['scope'],
        'changed_reviews':len(audit['entries']),'groups':{k:group_summary(v, base, candidate, breview, creview) for k,v in cgroups.items()}}
    from scripts.run_qa_development_segment import budget_snapshot
    budget=budget_snapshot(config['ledger'])
    cap=read(R/'budget-before.json')['absolute_cap_cny']
    assert cap==config['phase_budget_cny']==62.379233
    assert len({c['id'] for c in budget['calls']})==len(budget['calls'])
    report['shared_budget']={k:v for k,v in budget.items() if k!='calls'}
    report['shared_budget'].update(cap_cny=cap,remaining_cny=cap-budget['occupied_cny'],
        known_cny=sum(c['cny'] for c in budget['calls'] if c['status']=='estimated_from_usage'),
        unknown_reserved_cny=sum(c['cny'] for c in budget['calls'] if c['status']=='reserved_unknown_charge'))
    report['baseline_ledger']=ledger_snapshot([config],bm['execution_id'])
    report['provenance']={'candidate_runtime_commit':'fc0867e268c0f280a4dd543d98b2b0712fd70062',
        'baseline_runtime_commit':'4c920db','specification':str(R/'candidate-spec-v1/specification.json'),
        'candidate_config':str(R/'candidate-agent.json'),'candidate_suite':str(R/'candidate-suite.json'),
        'frozen_translation_cache':str(R/'translations/translations.json'),'cached_chinese_first':120,
        'new_translation_calls':0,'review_digests_original':{'baseline':digest(br),'candidate':digest(cr)},
        'segment_records':str(R/'segments'),'domain_audits':str(R/'candidate-domain-reviews')}
    report['scope'] += ['Single unblinded assistant review; local calibration does not remove general rater drift.',
        'Frozen translation errors, source contradictions and OCR ambiguities are retained.',
        'Different execution dates and provider generation variability limit causal attribution to this prompt change.',
        'No stable P95 is inferred from the three-repeat measurements; failed first attempts remain in denominators.',
        'Structural fallback text is excluded from assessable model-language counts.',
        'Known cost is usage-based estimation, not a settled provider invoice; unknown reservations remain charged against the cap.']
    report['scope'].append('The optional text changes only final-answer requests. Differences in earlier Agent tool paths and request counts are observational variation, not evidence that this text changed the tool policy.')
    assert not target.exists()

    atomic_json(target,report)
    print(json.dumps(report['groups']['all'],ensure_ascii=False))



if __name__ == "__main__":
    main()
