"""Fixed teacher-to-student transfer experiment using existing training entries."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from foliorecall.io import read_json, read_rows, write_json, provenance
from foliorecall.search import encoding_identity

ROOT = Path('outputs/extensions/D/teacher-transfer-v1')
HOLDOUT = Path('outputs/extensions/B/heldout-seeds-v1')
ARMS = ('P', 'O', 'L')
DOMAINS = ('biomedical', 'economics', 'esg_human')
METRICS = ('nDCG@10', 'Recall@5', 'Recall@10')


def prepare(root, repo=Path('.')):
    root, repo = root.resolve(), repo.resolve()
    if (root/'plan.json').exists():
        return read_json(root/'plan.json')
    if root.exists() and any(root.iterdir()):
        raise ValueError('准备目录非空；使用新目录或现有完整计划')
    history = repo/'outputs/stage3/targets-3000'
    target = read_json(history/'manifest.json')
    queries = read_rows(history/'queries.jsonl')
    if target['count'] != 3000 or len(queries) != 3000 or len({q['query_id'] for q in queries}) != 3000:
        raise ValueError('需要原始3000条唯一训练查询')
    heldout = repo/HOLDOUT
    training = read_json(heldout/'training/ordinary-42/result.json')
    if not training['complete'] or training['completed_steps'] != 64:
        raise ValueError('普通42教师尚未完成64步')
    teacher_l = read_json(heldout/'training/ordinary-42/checkpoint-64/encoding.json')
    teacher_o = target['encoding_config']
    if not teacher_l.get('adapter') or encoding_identity(teacher_l) == encoding_identity(teacher_o):
        raise ValueError('候选必须是独立的普通42适配器')
    student = read_json(repo/'configs/student-ml-cuda.json')
    if student['target_encoding_identity'] != target['encoding_identity']:
        raise ValueError('原学生与缓存教师身份不一致')
    student.pop('model_path', None)
    student.pop('backend', None)
    original_index = repo/'indexes/vdr-dev-original'
    candidate_index = root/'indexes/dev-L'
    datasets = {'dev': {'data':str(repo/'data/vdr-stage2/dev'),
        'indexes':{'original':str(original_index), 'lora':str(candidate_index)}, 'queries':200, 'pages':1000}}
    for domain in DOMAINS:
        datasets[domain] = {'data':str(heldout/'data'/domain),
            'indexes':{name:str(heldout/'indexes'/model/domain) for name,model in
                       [('original','original'),('lora','ordinary-42')]},
            'queries':{'biomedical':160,'economics':58,'esg_human':52}[domain],
            'pages':{'biomedical':1016,'economics':452,'esg_human':1538}[domain]}
    for arm,teacher in [('O',teacher_o),('L',teacher_l)]:
        cfg = dict(deepcopy(student), target_encoding_identity=encoding_identity(teacher))
        write_json(root/f'configs/train-{arm}.json',cfg)
        write_json(root/f'configs/teacher-{arm}.json',teacher)
    public = dict(student, device='cpu', dtype='float32',
                  model_path=str(repo/'outputs/extensions/D/onnx-int8-v1/models/torch'))
    write_json(root/'configs/P.json', public)
    write_json(root/'teacher-L.json', {'status':'selected', 'selection_scope':'this experiment only',
        'encoding_config':teacher_l, 'index_encoding_identity':encoding_identity(teacher_l)})
    plan = {'arms':list(ARMS), 'teacher_seed':42, 'teacher_step':64, 'student_seed':42,
        'training_queries':3000, 'epochs':3, 'steps':282, 'datasets':datasets,
        'original_targets':str(history), 'lora_targets':str(root/'targets-L'),
        'training_data':str(repo/'data/vdr-stage2'), 'teacher_L':str(root/'teacher-L.json'),
        'initialization':{'model_id':student['model_id'],'revision':student['revision']},
        'scope':'single student seed; seen development and retrospective three-domain validation',
        'new_api_calls':0, 'product_changed':False}
    write_json(root/'plan.json',plan)
    provenance(root/'preparation',plan)
    return plan


def index_command(config, pages, output):
    command = [sys.executable,'-u','-m','foliorecall','index','--config',str(config),
               '--pages',str(pages),'--output',str(output),'--chunk-size','64','--max-seconds','3600']
    if Path(str(output)+'.work','manifest.json').is_file():
        command.append('--resume')
    return command


def child(root, name, command):
    folder = root/'processes'/f'{name}-{time.time_ns()}'
    folder.mkdir(parents=True)
    started = time.monotonic()
    record = {'name':name,'command':command,'started_at':time.time()}
    write_json(folder/'process.json',record)
    with (folder/'stdout.log').open('w') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, seconds=time.monotonic()-started,finished_at=time.time())
    write_json(folder/'process.json',record)
    print(json.dumps({'process':name,'returncode':result.returncode,'seconds':record['seconds'],
                      'log':str(folder/'stdout.log')},ensure_ascii=False),flush=True)
    if result.returncode:
        raise RuntimeError(f'{name}失败；保留日志 {folder}')


def run_initial(root, cpu_python):
    plan = read_json(root/'plan.json')
    # Target caching validates and reuses each completed 64-query chunk.
    if not (root/'targets-L/status.json').exists() or not read_json(root/'targets-L/status.json')['complete']:
        child(root,'targets-L',[sys.executable,'-u','-m','foliorecall','cache-teacher',
            '--teacher',plan['teacher_L'],'--data',plan['training_data'],'--output',plan['lora_targets'],
            '--count','3000','--max-seconds','3600'])
    if not read_json(root/'targets-L/status.json')['complete']:
        raise RuntimeError('教师目标已分块保存，继续同一命令完成剩余查询')
    if read_rows(Path(plan['original_targets'])/'queries.jsonl') != read_rows(root/'targets-L/queries.jsonl'):
        raise ValueError('两臂教师目标的查询文本或顺序不同')
    dev = plan['datasets']['dev']
    output = Path(dev['indexes']['lora'])
    if not (output/'index.faiss').exists():
        child(root,'index-dev-L',index_command(root/'configs/teacher-L.json',Path(dev['data'])/'pages.jsonl',output))
    if not (output/'index.faiss').exists():
        raise RuntimeError('开发索引已分块保存，继续同一命令完成剩余页面')
    for arm in ('O','L'):
        output = root/'training'/arm
        if not (output/'result.json').exists() or not read_json(output/'result.json')['complete']:
            command = [sys.executable,'-u','-m','foliorecall','distill','--query-config',str(root/f'configs/train-{arm}.json'),
                '--targets',plan['original_targets'] if arm=='O' else plan['lora_targets'],
                '--output',str(output),'--index',dev['indexes']['original' if arm=='O' else 'lora'],
                '--data',dev['data'],'--micro-batch','32','--skip-evaluation','--max-seconds','3600']
            checkpoints = [p for p in output.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()]
            if checkpoints:
                command += ['--resume',str(max(checkpoints,key=lambda p:int(p.name.split('-')[-1])))]
            child(root,f'train-{arm}',command)
        training = read_json(output/'result.json')
        if not training['complete'] or training['completed_steps'] != 282:
            raise RuntimeError(f'{arm}未完成282步；保留断点，继续本臂')
        cfg = read_json(output/'final/query-config.json')
        write_json(root/f'configs/{arm}.json',dict(cfg,model_path=str((output/'final').resolve()),device='cpu',dtype='float32'))
    for arm in ARMS:
        if any(not (root/f'evaluations/{arm}/{domain}/result.json').exists() for domain in plan['datasets']):
            child(root,f'evaluate-{arm}',[str(cpu_python),'-u','-m','scripts.run_teacher_transfer','worker',
                '--root',str(root),'--arm',arm])
    return report(root)


def worker(root, arm):
    from scripts.run_query_backends import environment
    environment()
    from foliorecall.query import load_query_encoder, validate_query_config, benchmark
    from foliorecall.search import load_index
    from foliorecall.evaluation import validate_candidate_corpus
    plan = read_json(root/'plan.json')
    cfg = read_json(root/f'configs/{arm}.json')
    teacher = read_json(root/f'configs/teacher-{"L" if arm=="L" else "O"}.json')
    validate_query_config(cfg,teacher)
    started = time.monotonic()
    model = load_query_encoder(cfg)
    loading = time.monotonic()-started
    for domain,spec in plan['datasets'].items():
        output = root/f'evaluations/{arm}/{domain}'
        if (output/'result.json').exists():
            continue
        data = Path(spec['data'])
        index,pages = load_index(spec['indexes']['lora' if arm=='L' else 'original'],teacher)
        validate_candidate_corpus(pages,read_rows(data/'pages.jsonl'))
        queries = read_rows(data/'queries.jsonl')
        if len(pages)!=spec['pages'] or len(queries)!=spec['queries']:
            raise ValueError('完整语料或冻结查询数量不符')
        provenance(output,{'arm':arm,'query':cfg,'teacher':teacher,'dataset':spec,
            'loading_scope':'one model shared across up to four datasets in this process'})
        result = benchmark(model,cfg,index,pages,queries,read_json(data/'qrels.json'),warmups=5,repeats=1,search_threads=1)
        result.update(arm=arm,domain=domain,model_loading_seconds=loading,
            loading_scope='shared across this worker; do not sum per-dataset loading',
            scope=plan['scope'])
        write_json(output/'result.json',result)
        print(f'{arm} {domain}: {result["metrics"]}',flush=True)
    return {'arm':arm,'complete':True}


def aggregate(results, domains):
    rows = [dict(r,query_id=f'{d}/{r["query_id"]}') for d in domains for r in results[d]['results']]
    return {'domains':{d:results[d]['metrics'] for d in domains},
        'domain_equal':{k:statistics.mean(results[d]['metrics'][k] for d in domains) for k in METRICS},
        'query_weighted':{k:statistics.mean(r[k] for r in rows) for k in METRICS},'results':rows}


def numerical_gate(values):
    score = values['L']['domain_equal']['nDCG@10']
    better_domains = [d for d in DOMAINS if all(values['L']['domains'][d]['nDCG@10'] >
                       values[a]['domains'][d]['nDCG@10']+1e-12 for a in ('P','O'))]
    return {'macro_beats_both':all(score>values[a]['domain_equal']['nDCG@10']+1e-12 for a in ('P','O')),
            'better_domains':better_domains,
            'numerically_eligible':all(score>values[a]['domain_equal']['nDCG@10']+1e-12 for a in ('P','O')) and len(better_domains)>=2}


def report(root):
    from scripts.run_teacher_holdout import paired
    plan = read_json(root/'plan.json')
    results = {a:{d:read_json(root/f'evaluations/{a}/{d}/result.json') for d in plan['datasets']} for a in ARMS}
    for a in ARMS:
        for d,spec in plan['datasets'].items():
            expected = {r['query_id']:r['query'] for r in read_rows(Path(spec['data'])/'queries.jsonl')}
            actual = results[a][d]
            if (actual['candidates']!=spec['pages'] or len(actual['results'])!=len(expected)
                    or {r['query_id']:r['query'] for r in actual['results']}!=expected):
                raise ValueError('评测完整性或查询对应关系错误')
    values = {a:aggregate(results[a],DOMAINS) for a in ARMS}
    comparisons,cases = {},{}
    for baseline,candidate in [('O','L'),('P','L'),('P','O')]:
        name = f'{candidate}-vs-{baseline}'
        summary,rows = paired(values[baseline],values[candidate])
        summary['domains']={d:{k:results[candidate][d]['metrics'][k]-results[baseline][d]['metrics'][k]
                              for k in METRICS} for d in DOMAINS}
        comparisons[name]=summary
        write_json(root/f'paired/{name}.json',rows)
        if candidate=='L':
            _,all_rows = paired(aggregate(results[baseline],plan['datasets']),aggregate(results[candidate],plan['datasets']))
            cases[name]={'improvements':sorted([r for r in all_rows if r['delta']['nDCG@10']>1e-12],key=lambda r:-r['delta']['nDCG@10'])[:5],
                         'regressions':sorted([r for r in all_rows if r['delta']['nDCG@10']< -1e-12],key=lambda r:r['delta']['nDCG@10'])[:5]}
    gate = numerical_gate(values)
    review_path = root/'case-review.json'
    review = read_json(review_path) if review_path.exists() else {}
    gate['case_review_complete'] = review.get('complete') is True
    gate['continue_to_domains'] = gate['numerically_eligible'] and gate['case_review_complete'] and review.get('ranking_improvements_confirmed') is True
    result={'status':'measured' if not review else 'reviewed', 'arms':{a:{
        'dev':results[a]['dev']['metrics'],**{k:v for k,v in values[a].items() if k!='results'}} for a in ARMS},
        'comparisons':comparisons,'gate':gate,'scope':plan['scope'],
        'new_api_calls':0,'product_changed':False,'candidate_rebuilt':False}
    write_json(root/'comparison.json',result)
    write_json(root/'cases-to-review.json',cases)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker','report'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--cpu-python',default='/mnt/d/foliorecall_cache/envs/stage4_cpu/bin/python')
    args=parser.parse_args()
    root=args.root.resolve()
    if args.command=='prepare': result=prepare(root)
    elif args.command=='run': result=run_initial(root,args.cpu_python)
    elif args.command=='worker': result=worker(root,args.arm)
    else: result=report(root)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
