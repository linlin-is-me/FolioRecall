"""Targeted release-content check: known exclusions, manifests and model payloads."""
import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, read_rows, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--models')
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise ValueError('保留旧核对结果，请使用新目录')
provenance(output, {'demo': 'outputs/stage4/demo-bundle.zip', 'models': args.models,
    'scope': 'targeted release manifests and known material exclusions; no hashes or model computations'})
bundle = Path('outputs/stage4/demo-bundle')
rows = read_rows(bundle / 'pages.jsonl')
info = read_json(bundle / 'bundle.json')
old = {r['page_id']: r for r in read_rows('indexes/demo-original/pages.jsonl')}
excluded = {r['page_id'] for r in info['excluded']}
assert len(rows) == 42 and len({r['page_id'] for r in rows}) == 42
assert set(old) - {r['page_id'] for r in rows} == excluded
assert {r['page_number'] for r in info['excluded']} == {1, 6, 14}
for row in rows:
    assert row['page_number'] == old[row['page_id']]['page_number']
    assert not Path(row['preview']).is_absolute()
    assert (bundle / row['preview']).is_file()
with zipfile.ZipFile('outputs/stage4/demo-bundle.zip') as archive:
    names = archive.namelist()
    assert not any(name.lower().endswith('.pdf') for name in names)
    assert sum(name.endswith('.png') for name in names) == 42
    assert any(name.endswith('/SOURCES.md') or name == 'SOURCES.md' for name in names)
import pypdfium2 as pdfium
copyright_pages = []
for source in read_json('data/pdfs/sources.json'):
    path = Path('data/pdfs') / source['doc_name']
    found = []
    with pdfium.PdfDocument(path) as pdf:
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            textpage.close()
            page.close()
            if any(term in text.casefold() for term in ('creative commons', 'reproduction is authorised', 'copyright')):
                (output / (source['doc_name'] + f'.page{index+1}.txt')).write_text(text, encoding='utf-8')
                found.append(index + 1)
    if not found:
        raise ValueError(f"未定位许可文字: {source['doc_name']}")
    copyright_pages.append({'document': source['doc_name'], 'physical_pages': found, 'source_url': source['url']})
models = []
if args.models:
    for name in ('lora750', 'distilled94'):
        root = Path(args.models) / name
        with zipfile.ZipFile(root.with_suffix('.zip')) as archive:
            names = archive.namelist()
        forbidden = ('optimizer', 'scheduler', 'rng_state', 'training_args', 'training-queries', 'UPSTREAM_GENERATED_MODEL_CARD')
        assert not any(any(key in item for key in forbidden) for item in names)
        card = (root / 'README.md').read_text()
        assert 'widget:' not in card and '/mnt/d/' not in card
        assert (root / 'LICENSE').is_file() and (root / 'UPSTREAM.md').is_file()
        models.append({'name': name, 'files': names, 'unwanted_training_payloads_absent': True})
write_json(output / 'result.json', {'demo_pages': 42, 'excluded': info['excluded'],
    'original_ids_and_physical_pages_preserved': True, 'relative_previews_readable': True,
    'full_pdfs_absent': True, 'copyright_page_evidence': copyright_pages, 'models': models,
    'publication': 'not authorized; local candidates only',
    'scope': 'known third-party-image exclusions verified; report source and material conditions retained; not a blanket grant for third-party works'})
print(output / 'result.json')
