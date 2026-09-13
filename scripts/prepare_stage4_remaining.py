"""Write reviewable fixed jobs/config copies; never execute model commands."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('准备目录已存在，请指定新目录')
    config = read_json('configs/stage4-evaluation.json')
    provenance(output, config)
    configs = {'original': 'configs/baseline.json', 'lora750': 'outputs/stage2/lora/checkpoint-750/encoding.json',
        'en-gpu': 'configs/student-en-cuda.json', 'ml-gpu': 'configs/student-ml-cuda.json',
        'distilled94-gpu': 'outputs/stage3/distill-3000/checkpoint-94/query-config.json',
        'en-cpu': 'configs/student-en-cpu.json', 'ml-cpu': 'configs/student-ml-cpu.json',
        'distilled94-cpu': 'outputs/stage3/candidate-configs/distilled-best-cpu.json'}
    for name, source in configs.items():
        write_json(output / 'configs' / f'{name}.json', read_json(source))
    copies = {k: str((output / 'configs' / f'{k}.json').resolve()) for k in configs}
    gpu_python = '/root/.venvs/foliorecall_env/bin/python'
    cpu_python = '/mnt/d/foliorecall_cache/envs/stage4_cpu/bin/python'
    tasks = {t['name']: t for t in config['tasks']}
    jobs, matrix = [], []
    for name in ['hr', 'computer_science', 'pharmaceuticals', 'finance_en', 'industrial']:
        task = tasks[name]
        data = str(Path('outputs/stage4/data', name).resolve())
        indexes = {teacher: str(Path('outputs/stage4/indexes', name, teacher).resolve()) for teacher in ['original', 'lora750']}
        if name == 'hr':
            indexes['original'] = str(Path('indexes/hr-original').resolve())
        for teacher in ['original', 'lora750']:
            if name == 'hr' and teacher == 'original':
                continue
            jobs.append({'name': f'index-{name}-{teacher}', 'task': name, 'kind': 'index', 'device': 'gpu',
                'pages': task['pages'], 'output': indexes[teacher], 'command': [gpu_python, '-u', '-m', 'foliorecall', 'index',
                    '--config', copies[teacher], '--pages', str(Path(data) / 'pages.jsonl'), '--output', indexes[teacher], '--chunk-size', '64'],
                'resume': 'append --resume; use a new GPU runner name'})
        for device, candidates in [('gpu', ['original', 'lora750', 'en', 'ml', 'distilled94']), ('cpu', ['en', 'ml', 'distilled94'])]:
            for candidate in candidates:
                teacher = 'lora750' if candidate == 'lora750' else 'original'
                result = Path('outputs/stage4', device, name, candidate)
                reuse = device == 'cpu' and name == 'hr'
                if reuse:
                    result = Path('outputs/stage4/hr-students', candidate)
                item = {'task': name, 'candidate': candidate, 'device': device, 'teacher': teacher,
                    'index': indexes[teacher], 'result': str((result / 'result.json').resolve()), 'reuse': reuse}
                matrix.append(item)
                if reuse:
                    continue
                command = [gpu_python if device == 'gpu' else cpu_python, '-u', '-m', 'foliorecall', 'evaluate',
                    '--config', copies[teacher], '--index', indexes[teacher], '--data', data, '--output', str(result.resolve()), '--benchmark']
                if candidate not in ['original', 'lora750']:
                    command += ['--query-config', copies[candidate + '-' + device]]
                jobs.append({'name': f'{device}-{name}-{candidate}', 'kind': 'evaluate', 'task': name,
                    'device': device, 'output': str(result.resolve()), 'command': command})
    result = {'gpu_status': 'waiting for user GPU window', 'gpu_budget_seconds': 36000,
        'reserve_seconds': 3600, 'next_domain_estimate_margin': 1.25, 'tasks': list(tasks.values()),
        'config_sources': configs, 'config_copies': copies, 'jobs': jobs, 'matrix': matrix,
        'scope': 'fixed candidates; no checkpoint reselection; no push/upload/release authorization'}
    write_json(output / 'jobs.json', result)
    print(f'{len(jobs)} pending jobs, {len(matrix)} result cells; no model started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/stage4/remaining-preparation')
    args = parser.parse_args()
    prepare(args.output)
