"""CPU-only validation of newly copied local model packages; no teacher loading."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import argparse
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json
from foliorecall.query import encode_query_texts, load_query_encoder

parser = argparse.ArgumentParser()
parser.add_argument('--models', default='outputs/stage4/model-candidates-rc2-v2')
parser.add_argument('--output', default='outputs/stage4/model-export-cpu-validation-v2')
args = parser.parse_args()
root = Path(args.models).resolve()
output = Path(args.output)
if output.exists():
    raise SystemExit('验证目录已存在，请保留既有结果')
torch.set_num_threads(4)
original = read_json('outputs/stage3/candidate-configs/distilled-best-cpu.json')
copied = read_json(root / 'distilled94/query-cpu.json')
copied['model_path'] = str(root / 'distilled94')
provenance(output, {'original': original, 'copied': copied, 'device': 'cpu'})
queries = [r['query'] for r in read_json('outputs/stage4/demo-bundle/queries.json')]
vectors = []
for config in (original, copied):
    model = load_query_encoder(config)
    vectors.append(encode_query_texts(model, queries, config))
    del model
delta = float(np.max(np.abs(vectors[0]-vectors[1])))
if delta != 0:
    raise ValueError(f'导出学生与原检查点CPU向量不一致: {delta}')
from safetensors import safe_open
original_adapter = Path('outputs/stage2/lora/checkpoint-750/adapter_model.safetensors')
copied_adapter = root / 'lora750/adapter_model.safetensors'
with safe_open(original_adapter, framework='pt', device='cpu') as a, safe_open(copied_adapter, framework='pt', device='cpu') as b:
    names = list(a.keys())
    assert names == list(b.keys())
    assert all(torch.equal(a.get_tensor(k), b.get_tensor(k)) for k in names)
write_json(output / 'result.json', {'device': 'cpu', 'cuda_visible_devices': '',
    'queries': queries, 'student_vector_max_abs_difference': delta, 'adapter_tensors_equal': len(names),
    'scope': 'exported student CPU loading matches original; adapter CPU tensors match; teacher/PEFT GPU integration still pending'})
print(output / 'result.json')
