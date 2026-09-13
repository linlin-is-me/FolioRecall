"""Measure the actual Gradio queue response separately from query-core timing."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from gradio_client import Client

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--server-output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise SystemExit('Use a new API measurement output')
output.mkdir(parents=True)
queries = json.loads(Path('outputs/stage4/demo-bundle/queries.json').read_text())
client = Client('http://127.0.0.1:7864', verbose=False, download_files=False)
for i in range(5):
    client.predict(queries[i % len(queries)]['query'], 5, api_name='/search')
rows = []
for repeat in range(3):
    for item in queries:
        started = time.perf_counter()
        result = client.predict(item['query'], 5, api_name='/search')
        rows.append({'repeat': repeat, 'query': item['query'], 'seconds': time.perf_counter()-started,
                     'server_status': result[3]})
    print(f'API repeat {repeat+1}/3 complete', flush=True)
# Exercise the actual download URL returned by the public application endpoint.
first = client.predict(queries[0]['query'], 5, api_name='/search')
download = first[2]
if isinstance(download, dict):
    url = download.get('url') or download['path']
else:
    url = download
if not url.startswith('http'):
    url = 'http://127.0.0.1:7864/gradio_api/file=' + url
with urllib.request.urlopen(url) as response:
    ranked = json.load(response)
assert ranked[0]['page_number'] == 10
reference = json.loads(Path('outputs/stage4/installed-cpu/cli-result.json').read_text())
assert ranked == reference
(output / 'downloaded-results.json').write_text(json.dumps(ranked, indent=2))
try:
    client.predict(' ', 5, api_name='/search')
except Exception as exc:
    error = str(exc)
    assert '查询不能为空' in error, error
else:
    raise AssertionError('Empty query unexpectedly succeeded')
startup = json.loads((Path(args.server_output) / 'startup.json').read_text())
server_rows = [json.loads(line) for line in (Path(args.server_output) / 'requests.jsonl').read_text().splitlines()]
timed_server_rows = server_rows[-10:-1]
result = {'requests': rows, 'warmups': 5, 'repeats': 3, 'queries': len(queries), 'top_k': 5,
    'client_response_seconds': {f'p{p}': float(np.percentile([r['seconds'] for r in rows], p)) for p in (50, 95)},
    'server_timings': {key: {f'p{p}': float(np.percentile([r['timing'][key] for r in timed_server_rows], p)) for p in (50, 95)}
                       for key in ('encode_seconds', 'request_seconds', 'display_ready_seconds')},
    'timing_scope': 'client predict through Gradio queue to response metadata; excludes preview downloads and browser rendering; server timings are separately recorded',
    'download_matches_cli': True, 'empty_query_error': error, 'startup': startup,
    'successful_requests_in_same_server': len(server_rows)}
(output / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'requests'}, indent=2))
client.close()
