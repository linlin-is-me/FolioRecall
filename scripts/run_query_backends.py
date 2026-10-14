"""Export complete ST query packages and compare CPU backends using frozen indexes."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from foliorecall.io import read_json, read_rows, write_json, provenance

ROOT = Path('outputs/extensions/D/onnx-int8-v1')
ARMS = ('torch', 'onnx-fp32', 'onnx-int8')


def environment():
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                      HF_HOME='/mnt/d/foliorecall_cache/stage4_cpu/huggingface')
    import torch
    import faiss
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    faiss.omp_set_num_threads(1)


def prepare(root):
    import numpy as np
    import torch
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from foliorecall.query import load_query_encoder, encode_query_texts
    root.mkdir(parents=True, exist_ok=True)
    if (root/'preparation.json').exists():
        raise ValueError('准备结果已存在，不重复导出')
    base = read_json('configs/student-ml-cpu.json')
    provenance(root/'preparation', base)
    started = time.perf_counter()
    model = load_query_encoder(base)
    texts = ['revenue', 'Which figure compares population projections?',
             'How do demographic change and employment evolve in the European Union?', 'a '*600]
    expected = encode_query_texts(model, texts, base)
    # ST save retains tokenizer, mean pooling, both Dense modules and Normalize.
    source = root/'models/torch'
    model.save_pretrained(str(source), create_model_card=False)
    write_json(source/'query-config.json', base)
    torch_config = dict(base, model_path=str(source.resolve()))
    write_json(root/'configs/torch.json', torch_config)
    logical_parameters = sum(p.numel() for p in model.parameters())
    fp = root/'models/onnx-fp32'
    def copy_outer(folder):
        folder.mkdir(parents=True,exist_ok=True)
        for p in source.iterdir():
            if p.name == 'model.safetensors': continue
            if p.is_dir(): shutil.copytree(p,folder/p.name,dirs_exist_ok=True)
            else: shutil.copy2(p,folder/p.name)
        (folder/'onnx').mkdir(exist_ok=True)
    copy_outer(fp)
    class Backbone(torch.nn.Module):
        def __init__(self,backbone): super().__init__();self.backbone=backbone
        def forward(self,input_ids,attention_mask):
            return self.backbone(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
    dummy=model.tokenize(texts[:2])
    torch.onnx.export(Backbone(model[0].model).eval(),(dummy['input_ids'],dummy['attention_mask']),
        str(fp/'onnx/model.onnx'),input_names=['input_ids','attention_mask'],output_names=['last_hidden_state'],
        dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},
                      'last_hidden_state':{0:'batch',1:'sequence'}},opset_version=17,dynamo=False)
    del model
    # Quantize that same graph, without another conversion or quality-driven tuning.
    quant = root/'models/onnx-int8'
    copy_outer(quant)
    # Exact dynamic avx512_vnni preset: U8 activations, symmetric per-channel S8
    # weights, no reduced range. Source: Optimum AutoQuantizationConfig.avx512_vnni.
    quantize_dynamic(str(fp/'onnx/model.onnx'),str(quant/'onnx/model_int8.onnx'),
        weight_type=QuantType.QInt8,per_channel=True,reduce_range=False,
        extra_options={'WeightSymmetric':True,'ActivationSymmetric':False})
    checks = {}
    for arm, folder in [('onnx-fp32', fp), ('onnx-int8', quant)]:
        graphs = list((folder/'onnx').glob('*.onnx'))
        if len(graphs) != 1: raise ValueError(f'导出图不唯一: {graphs}')
        cfg = dict(base, backend='onnx', model_path=str(folder.resolve()),
                   onnx_file=str(graphs[0].relative_to(folder)))
        write_json(folder/'query-config.json', dict(cfg, model_path='.'))
        write_json(root/f'configs/{arm}.json', cfg)
        reloaded = load_query_encoder(cfg)
        actual = encode_query_texts(reloaded, texts, cfg)
        batched = encode_query_texts(reloaded, texts, dict(cfg,batch_size=4))
        checks[arm] = {'max_abs_diff': float(np.max(np.abs(actual-expected))),
            'cosines': np.sum(actual*expected, axis=1).tolist(), 'shape': list(actual.shape),
            'norms': np.linalg.norm(actual,axis=1).tolist(),
            'mixed_batch_max_abs_diff':float(np.max(np.abs(batched-actual)))}
        if arm == 'onnx-fp32' and not np.allclose(actual, expected, atol=1e-5, rtol=1e-4):
            raise ValueError('FP32完整表示差异异常')
        del reloaded
    import onnx
    graph = onnx.load(str(next((quant/'onnx').glob('*.onnx'))))
    result = {'logical_parameters': logical_parameters, 'checks': checks,
        'seconds': time.perf_counter()-started, 'quantization': 'avx512_vnni; dynamic; backbone only',
        'quantized_operator_counts': dict(Counter(n.op_type for n in graph.graph.node)),
        'versions': {n: importlib.metadata.version(n) for n in
                     ['torch','sentence-transformers','transformers','onnx','onnxruntime']},
        'export_method':'native torch.onnx + ORT; Optimum requires transformers<4.58, ST6 requires >=5',
        'quantization_settings':{'per_channel':True,'weight_type':'QInt8','reduce_range':False,
                                 'WeightSymmetric':True,'ActivationSymmetric':False}}
    write_json(root/'preparation.json', result)
    print(json.dumps(result, indent=2))


def worker(root, arm, output, domain=None):
    from foliorecall.query import load_query_encoder, validate_query_config, benchmark
    from foliorecall.search import load_index
    from foliorecall.evaluation import validate_candidate_corpus
    cfg = read_json(root/f'configs/{arm}.json')
    teacher = read_json('configs/baseline.json')
    validate_query_config(cfg, teacher)
    folder = Path('data/vdr-stage2/dev') if domain is None else Path('outputs/stage4/data')/domain
    index_folder = Path('indexes/vdr-dev-original') if domain is None else (
        Path('indexes/hr-original') if domain == 'hr' else Path('outputs/stage4/indexes')/domain/'original')
    if (output/'result.json').exists(): raise ValueError('结果已存在')
    provenance(output, {'query':cfg, 'teacher':teacher,'data':str(folder),'index':str(index_folder)})
    index,pages = load_index(index_folder,teacher)
    validate_candidate_corpus(pages,read_rows(folder/'pages.jsonl'))
    started = time.perf_counter()
    model = load_query_encoder(cfg)
    loading = time.perf_counter()-started
    result = benchmark(model,cfg,index,pages,read_rows(folder/'queries.jsonl'),read_json(folder/'qrels.json'),
                       warmups=5,repeats=3 if domain else 1)
    result.update(model_loading_seconds=loading,domain=domain or 'dev',arm=arm)
    write_json(output/'result.json',result)
    print(json.dumps({k:result[k] for k in ('arm','domain','metrics','request_seconds','warm_memory')},ensure_ascii=False),flush=True)


def run(root, domains=False, arms=ARMS):
    if domains:
        tasks = [(arm,domain,root/'domains'/domain/arm) for domain in
            ['hr','computer_science','finance_en','industrial','pharmaceuticals'] for arm in arms]
    else:
        order = [ARMS, ARMS[1:]+ARMS[:1], ARMS[2:]+ARMS[:2]]
        tasks = [(arm,None,root/'dev'/f'round-{n+1}'/arm) for n,seq in enumerate(order) for arm in seq]
    for arm,domain,output in tasks:
        if (output/'result.json').exists(): continue
        output.mkdir(parents=True,exist_ok=True)
        cmd = [sys.executable,'-u','-m','scripts.run_query_backends','worker','--root',str(root),
               '--arm',arm,'--output',str(output)] + (['--domain',domain] if domain else [])
        with (output/'stdout.log').open('w') as log:
            started = time.perf_counter()
            child = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        write_json(output/'process.json',{'seconds':time.perf_counter()-started,'returncode':child.returncode,'command':cmd})
        if child.returncode: raise RuntimeError(f'运行失败，见 {output}/stdout.log')
        print(f'finished {domain or "dev"} {arm} {output}',flush=True)


def report(root):
    import numpy as np
    groups = {}
    for scope,folders in [('dev',root.glob('dev/round-*/*/result.json')),
                          ('domains',root.glob('domains/*/*/result.json'))]:
        rows = [read_json(p) for p in sorted(folders)]
        groups[scope] = {}
        for name in sorted({(r['domain'],r['arm']) for r in rows}):
            selected = [r for r in rows if (r['domain'],r['arm'])==name]
            times = [q for r in selected for q in r['requests']]
            groups[scope]['/'.join(name)] = {'metrics':selected[0]['metrics'],
                'request_ms':{f'p{p}':float(np.percentile([q['request_seconds']*1000 for q in times],p)) for p in (50,95)},
                'rounds_ms':[r['request_seconds']['p50']*1000 for r in selected],
                'warm_rss_mib':[r['warm_memory']['rss_bytes']/2**20 for r in selected],
                'peak_rss_mib':[r['final_memory']['peak_rss_bytes']/2**20 for r in selected],
                'loading_seconds':[r['model_loading_seconds'] for r in selected],
                'package_bytes':selected[0]['model_package_bytes'],'requests':len(times)}
    baseline = read_json(root/'dev/round-1/torch/result.json')
    pairs = {}
    for arm in ARMS[1:]:
        candidate = read_json(root/f'dev/round-1/{arm}/result.json')
        pairs[arm] = []
        for a,b in zip(baseline['results'],candidate['results']):
            assert a['query_id']==b['query_id']
            pairs[arm].append({'query_id':a['query_id'],'query':a['query'],
                'delta':{m:b[m]-a[m] for m in ['nDCG@10','Recall@5','Recall@10']},
                'baseline_pages':[r['page_id'] for r in a['ranking']],
                'candidate_pages':[r['page_id'] for r in b['ranking']]})
    write_json(root/'comparison.json',{'groups':groups,'pairs':pairs,'scope':'previously seen development and retrospective domains; no generation calls'})
    print(json.dumps(groups,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','dev','domains','worker','report'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--domain');p.add_argument('--output',type=Path)
    a=p.parse_args();environment()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='worker':worker(a.root,a.arm,a.output,a.domain)
    elif a.command=='report':report(a.root)
    else:run(a.root,a.command=='domains',a.arms)


if __name__=='__main__':main()
