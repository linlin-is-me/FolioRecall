"""Fixed query-depth experiment; delegates training and retrieval to existing entries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from foliorecall.io import read_json, read_rows, write_json, provenance

ROOT = Path('outputs/extensions/D/multidepth-v1')
ARMS = ('public6', 'pruned4', 'ordinary6', 'independent4', 'joint6', 'joint4')
HISTORY = Path('outputs/stage3/distill-3000')


def prepare(root):
    from foliorecall.query import load_query_encoder
    from foliorecall.multidepth import export_exit
    if (root/'preparation.json').exists():
        raise ValueError('准备结果已存在')
    original = read_json('configs/student-ml-cuda.json')
    historical = read_json(HISTORY/'settings.json')
    if original != historical['query_config'] or read_rows(HISTORY/'training-queries.jsonl') != read_rows('outputs/stage3/targets-3000/queries.jsonl'):
        raise ValueError('历史普通臂训练配置或查询变化')
    source = Path('outputs/extensions/D/onnx-int8-v1/models/torch').resolve()
    base = dict(original, device='cpu', dtype='float32', model_path=str(source))
    provenance(root/'preparation', {'baseline': historical, 'public_model':str(source)})
    write_json(root/'configs/public6.json', base)
    model = load_query_encoder(base)
    _, pruned = export_exit(model, 4, root/'models/pruned4', base)
    write_json(root/'configs/pruned4.json',pruned)
    write_json(root/'configs/train-independent4.json',dict(pruned,device='cuda',dtype='bfloat16'))
    write_json(root/'configs/train-joint.json',original)
    del model
    cfg = dict(base,model_path=str((HISTORY/'checkpoint-282').resolve()))
    model = load_query_encoder(cfg)
    _, ordinary = export_exit(model, 6, root/'models/ordinary6',base)
    write_json(root/'configs/ordinary6.json',ordinary)
    result={'arms':list(ARMS),'training_runs':['independent4','joint'],
            'historical_baseline':str(HISTORY),'historical_reuse':'same original configuration, 3000 query rows/order and unchanged training computation',
            'targets':'outputs/stage3/targets-3000','epochs':3,'steps':282,'seed':42,
            'objective':'cached original-teacher cosine; joint exits equally weighted',
            'new_api_calls':0,'scope':'seen development, not independent acceptance'}
    write_json(root/'preparation.json',result)
    return result


def train(root):
    started=time.monotonic()
    execution={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'window_seconds':14400,'processes':[]}
    write_json(root/'execution.json',execution)
    for arm,depths in [('independent4',[4]),('joint',[4,6])]:
        output=root/'training'/arm
        if (output/'result.json').exists() and read_json(output/'result.json')['complete']:
            continue
        remaining=14400-(time.monotonic()-started)-120
        if remaining < 120: raise RuntimeError('GPU窗口剩余不足，保留已有结果')
        command=[sys.executable,'-u','-m','foliorecall','distill','--query-config',str(root/f'configs/train-{arm}.json'),
            '--targets','outputs/stage3/targets-3000','--output',str(output),'--skip-evaluation',
            '--max-seconds',str(min(remaining,1800)),'--depths',*map(str,depths)]
        checkpoints=sorted(output.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints: command+=['--resume',str(checkpoints[-1])]
        root.mkdir(parents=True,exist_ok=True)
        before=time.monotonic()
        with (root/f'train-{arm}.log').open('a') as log:
            child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining+90)
        execution['processes'].append({'arm':arm,'command':command,'seconds':time.monotonic()-before,'returncode':child.returncode})
        execution['elapsed_seconds']=time.monotonic()-started
        write_json(root/'execution.json',execution)
        if child.returncode: raise RuntimeError(f'{arm}训练失败，见保留日志')
    for arm,path in [('independent4',root/'training/independent4/final'),
                     ('joint4',root/'training/joint/exports/depth-4'),('joint6',root/'training/joint/exports/depth-6')]:
        cfg=read_json(path/'query-config.json')
        write_json(root/f'configs/{arm}.json',dict(cfg,model_path=str(path.resolve()),device='cpu',dtype='float32'))
    return execution


def worker(root,arm,output,domain=None):
    from scripts.run_query_backends import environment
    environment()
    from foliorecall.query import load_query_encoder, validate_query_config, benchmark
    from foliorecall.search import load_index
    from foliorecall.evaluation import validate_candidate_corpus
    cfg=read_json(root/f'configs/{arm}.json');teacher=read_json('configs/baseline.json')
    validate_query_config(cfg,teacher)
    data=Path('data/vdr-stage2/dev') if domain is None else Path('outputs/stage4/data')/domain
    index_path=Path('indexes/vdr-dev-original') if domain is None else (
        Path('indexes/hr-original') if domain=='hr' else Path('outputs/stage4/indexes')/domain/'original')
    provenance(output,{'query':cfg,'index':str(index_path),'data':str(data)})
    index,pages=load_index(index_path,teacher)
    validate_candidate_corpus(pages,read_rows(data/'pages.jsonl'))
    started=time.perf_counter();model=load_query_encoder(cfg);loading=time.perf_counter()-started
    result=benchmark(model,cfg,index,pages,read_rows(data/'queries.jsonl'),read_json(data/'qrels.json'),warmups=5,repeats=3 if domain else 1,search_threads=1)
    result.update(arm=arm,domain=domain or 'dev',model_loading_seconds=loading)
    write_json(output/'result.json',result)
    return {k:result[k] for k in ['arm','domain','metrics','request_seconds','warm_memory']}


def run(root,arms,domains=False):
    if domains:
        tasks=[(arm,domain,root/'domains'/domain/arm) for domain in
            ['hr','computer_science','finance_en','industrial','pharmaceuticals'] for arm in arms]
    else:
        tasks=[(arm,None,root/f'dev/round-{n+1}'/arm) for n in range(3)
               for arm in arms[(2*n)%len(arms):]+arms[:(2*n)%len(arms)]]
    for arm,domain,output in tasks:
        if (output/'result.json').exists():continue
        output.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u','-m','scripts.run_multidepth','worker','--root',str(root),
             '--arm',arm,'--output',str(output)]+(['--domain',domain] if domain else [])
        before=time.monotonic()
        with (output/'stdout.log').open('w') as log:
            p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        write_json(output/'process.json',{'command':cmd,'seconds':time.monotonic()-before,'returncode':p.returncode})
        if p.returncode:raise RuntimeError(f'评测失败：{output}')
        print(f'finished {domain or "dev"} {arm}',flush=True)
    return {'complete':len(tasks)}


def report(root):
    import numpy as np
    from scripts.run_llave import paired
    arms={};first={}
    for arm in ARMS:
        results=[read_json(p) for p in sorted((root/'dev').glob(f'round-*/{arm}/result.json'))]
        if len(results)!=3:raise ValueError(f'{arm}缺少三轮开发测量')
        a=results[0];first[arm]=a
        requests=[r for result in results for r in result['requests']]
        arms[arm]={'metrics':a['metrics'],'requests':len(requests),
            **{k:{'p50_ms':float(np.percentile([r[k] for r in requests],50))*1000,
                  'p95_ms':float(np.percentile([r[k] for r in requests],95))*1000}
               for k in ['encode_seconds','request_seconds']},
            'round_p50_ms':[r['request_seconds']['p50']*1000 for r in results],
            'loading_seconds':[r['model_loading_seconds'] for r in results],
            'warm_rss_bytes':[r['warm_memory']['rss_bytes'] for r in results],
            'end_rss_bytes':[r['final_memory']['rss_bytes'] for r in results],
            'peak_rss_bytes':[r['final_memory']['peak_rss_bytes'] for r in results],
            'parameter_count':a['parameter_count'],'model_package_bytes':a['model_package_bytes'],
            'repeat_ranking_changes':sum(sum([p['page_id'] for p in x['ranking']]!=[p['page_id'] for p in y['ranking']]
                for x,y in zip(a['results'],b['results'])) for b in results[1:])}
    pairs={};cases={}
    comparisons=[('public6',arm) for arm in ARMS[1:]]+[
        ('pruned4','independent4'),('independent4','joint4'),('ordinary6','joint6')]
    for baseline,candidate in comparisons:
        name=f'{candidate}-vs-{baseline}'
        pairs[name],rows=paired(first[baseline],first[candidate])
        write_json(root/f'paired/{name}.json',rows)
        cases[name]={'improvements':sorted([r for r in rows if r['delta']['nDCG@10']>1e-12],key=lambda r:-r['delta']['nDCG@10'])[:5],
                     'regressions':sorted([r for r in rows if r['delta']['nDCG@10']< -1e-12],key=lambda r:r['delta']['nDCG@10'])[:5]}
    result={'status':'measured_pending_case_review','arms':arms,'paired':pairs,
            'scope':'single seed, seen development, 200 distinct queries; 600 timing repetitions per arm',
            'new_api_calls':0,'product_default_changed':False}
    write_json(root/'comparison.json',result);write_json(root/'cases-to-review.json',cases)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','dev','domains','worker','report'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arms',nargs='+',default=list(ARMS))
    p.add_argument('--arm');p.add_argument('--output',type=Path);p.add_argument('--domain')
    a=p.parse_args()
    if a.command=='prepare':
        from scripts.run_query_backends import environment
        environment()
        result=prepare(a.root)
    elif a.command=='train':result=train(a.root)
    elif a.command=='worker':result=worker(a.root,a.arm,a.output,a.domain)
    elif a.command=='report':result=report(a.root)
    else:result=run(a.root,a.arms,a.command=='domains')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
