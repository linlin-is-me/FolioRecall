"""Print availability only; never print credentials or touch CUDA."""
import json
import os
import sys
from importlib.metadata import version, PackageNotFoundError
packages = {}
for name in ('torch', 'pypdfium2', 'rank-bm25', 'gradio', 'httpx'):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': sys.executable, 'packages': packages,
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'credentials_present': {k: bool(os.environ.get(k)) for k in ('DASHSCOPE_API_KEY', 'DASHSCOPE_BASE_URL')}}, indent=2))
