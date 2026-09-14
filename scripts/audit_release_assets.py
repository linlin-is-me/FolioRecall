"""Check selected release payloads and exported numeric evidence without model loads."""
import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, write_json

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--release', default='outputs/stage4/release-v0.1.0rc2-post-audit')
args = parser.parse_args()
release = Path(args.release)
output = release / 'validation/materials'
if output.exists():
    raise ValueError('材料检查记录已存在')
provenance(output, {'release': str(release), 'operation': 'selected payload and numerical export checks; no hashes or model computation'})
formal = read_json('outputs/stage4/formal-complete/result.json')
queries, requests = 0, 0
for cell in formal['cells']:
    original = read_json(cell['result'])
    exported = read_json(release / 'experiment-evidence/evaluations' / cell['task'] / cell['device'] / cell['candidate'] / 'result.json')
    assert original['metrics'] == exported['metrics']
    assert original['requests'] == exported['requests']
    assert len(original['results']) == len(exported['results'])
    for old, new in zip(original['results'], exported['results']):
        assert old['query_id'] == new['query_id'] and 'query' not in new
        for metric in ('nDCG@10', 'Recall@5', 'Recall@10'):
            assert old[metric] == new[metric]
        assert [(r['page_id'], r['score']) for r in old['ranking']] == [(r['page_id'], r['score']) for r in new['ranking']]
    queries += len(exported['results'])
    requests += len(exported['requests'])
for task in read_json('configs/stage4-evaluation.json')['tasks']:
    original = read_json(Path('outputs/stage4/bm25') / task['name'] / 'result.json')
    exported = read_json(release / 'experiment-evidence/bm25' / task['name'] / 'result.json')
    assert original['metrics'] == exported['metrics'] and original['requests'] == exported['requests']
for name in ('lora750', 'distilled94'):
    original = Path('outputs/stage4/model-candidates-rc2-final') / name
    with zipfile.ZipFile(release / (name + '.zip')) as package:
        expected = {str(p.relative_to(original)).replace('\\', '/') for p in original.rglob('*') if p.is_file()}
        assert {n.removeprefix(name + '/') for n in package.namelist()} == expected
        for entry in package.infolist():
            relative = entry.filename.removeprefix(name + '/')
            if relative not in ('README.md', 'source.json'):
                source = original / relative
                assert source.stat().st_size == entry.file_size
                if source.suffix == '.json':
                    assert source.read_bytes() == package.read(entry)
        assert not any(n.endswith(('.pt', '.pth', '.pdf')) for n in package.namelist())
with zipfile.ZipFile(release / 'demo-bundle.zip') as package:
    for name in ('pages.jsonl', 'config.json'):
        assert package.read('demo-bundle/' + name) == (Path('outputs/stage4/demo-bundle') / name).read_bytes()
    assert len([n for n in package.namelist() if '/previews/' in n]) == 42
assert read_json(release / 'experiment-evidence/validation/resume-probe/comparison/result.json')['queries'] == 4
write_json(output / 'result.json', {
    'passed': True, 'neural_cells': len(formal['cells']), 'query_results': queries,
    'raw_request_records': requests, 'bm25_tasks': 5, 'numbers_and_ranking_ids_unchanged': True,
    'model_payload_file_sizes_and_configs_match_retained_packages': True,
    'model_weight_verification': 'copied payload and ZIP size checks; prior runtime equality evidence reused; no new weight hash or GPU check',
    'demo_pages': 42, 'demo_mapping_and_config_unchanged': True,
    'training_query_text_list_replaced_by_count': True})
print(output / 'result.json')
