"""Copy one pinned student and the movable demo into the isolated CPU workspace."""
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json, write_json

source_home = Path('/mnt/d/foliorecall_cache/huggingface')
target_home = Path('/mnt/d/foliorecall_cache/stage4_cpu/huggingface')
workspace = Path('/mnt/d/foliorecall_cache/stage4_cpu/moved example')
config = read_json('configs/student-ml-cpu.json')
model_dir = 'models--' + config['model_id'].replace('/', '--')
relative = Path('hub') / model_dir / 'snapshots' / config['revision']
if (target_home / relative).exists():
    raise SystemExit('CPU student snapshot already exists; preserve it and use the existing validation workspace')
shutil.copytree(source_home / relative, target_home / relative)
shutil.copytree('outputs/stage4/demo-bundle', workspace)
write_json('outputs/stage4/cpu-isolation.json', {'hf_home': str(target_home), 'snapshot': str(relative),
    'workspace': str(workspace), 'model_id': config['model_id'], 'revision': config['revision'],
    'operation': 'copy only the pinned ML snapshot, resolving its cache symlinks; no teacher weights copied'})
print(workspace)
