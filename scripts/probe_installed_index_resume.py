"""Installed-package GPU pause probe. Only this 45-page probe uses 16-page chunks."""
import argparse
import os
from pathlib import Path
import signal

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', required=True)
args = parser.parse_args()
os.chdir(args.workspace)
import foliorecall
import foliorecall.encoding as encoding
from foliorecall.indexing import build_index
from foliorecall.io import read_json

if '/site-packages/' not in str(Path(foliorecall.__file__).resolve()):
    raise RuntimeError('Probe must use the installed wheel')
original = encoding.encode_pages

def encode_then_pause(*args, **kwargs):
    vectors = original(*args, **kwargs)
    # Delivered after real GPU encoding; builder commits the block before pausing.
    os.kill(os.getpid(), signal.SIGINT)
    return vectors

encoding.encode_pages = encode_then_pause
result = build_index(read_json('configs/baseline.json'), 'pages/pages.jsonl', 'index-resumed', chunk_size=16)
if result['complete'] or result['completed_pages'] != 16:
    raise RuntimeError(f'Unexpected pause result: {result}')
print(result, flush=True)
