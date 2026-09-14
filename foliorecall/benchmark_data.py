"""Pinned ViDoRe corpora; no sampling of candidates or modification of qrels."""
import io
import math
from pathlib import Path
import time

from .io import provenance, read_json, read_rows, write_json, write_rows


def validate_task(pages, queries, qrels, expected):
    ids = [p['page_id'] for p in pages]
    query_ids = [q['query_id'] for q in queries]
    if len(ids) != len(set(ids)) or len(query_ids) != len(set(query_ids)):
        raise ValueError('页面或查询 ID 重复')
    if len(pages) != expected['pages'] or len(queries) != expected['english_queries']:
        raise ValueError(f'任务数量不符: {len(pages)} pages, {len(queries)} queries')
    if set(qrels) != set(query_ids):
        raise ValueError('查询与标签 ID 不一致')
    candidates = set(ids)
    for qid, relevance in qrels.items():
        if not set(relevance).issubset(candidates):
            raise ValueError(f'qrels 引用了候选库外页面: {qid}')
        if not any(score > 0 for score in relevance.values()):
            raise ValueError(f'查询无正相关标签: {qid}')
        if any(not math.isfinite(score) or score < 0 for score in relevance.values()):
            raise ValueError('相关性等级无效')


def training_reference(training_data):
    """Validate an explicit split before any dataset download or output write."""
    if training_data is None:
        return None
    from .data_preparation import normalized_query
    path = Path(training_data).resolve() / 'split.json'
    split = read_json(path)
    if not isinstance(split, dict) or not isinstance(split.get('train'), list):
        raise ValueError('训练切分须包含 train 记录列表')
    rows = split['train']
    if any(not isinstance(row, dict) or not isinstance(row.get('query'), str)
           or not row['query'].strip() for row in rows):
        raise ValueError('训练记录须包含非空 query 文本')
    return {'path': str(path), 'count': len(rows),
            'queries': {normalized_query(row['query']) for row in rows}}


def record_overlap(output, task, queries, training, run):
    from .data_preparation import normalized_query
    report = {'status': 'not_checked', 'training_file': None,
              'training_query_count': None, 'normalized_training_query_overlap': None}
    if training is not None:
        report.update(status='checked', training_file=training['path'],
                      training_query_count=training['count'],
                      normalized_training_query_overlap=[q['query_id'] for q in queries
                          if normalized_query(q['query']) in training['queries']])
    temporary = run / 'training-overlap.json.tmp'
    write_json(temporary, report)
    temporary.replace(run / 'training-overlap.json')
    report['record'] = str((run / 'training-overlap.json').resolve())
    print(f"{task['name']}: training overlap {report['status']}; {report['record']}", flush=True)
    return report


def load_dataset_provenance(data):
    """Attach the newest completed overlap audit without changing historical sources."""
    data = Path(data)
    source = read_json(data / 'source.json')
    runs = data / 'runs'
    audits = [run / 'training-overlap.json' for run in runs.iterdir()
              if run.is_dir() and run.name.isascii() and run.name.isdecimal()
              and (run / 'training-overlap.json').is_file()] if runs.is_dir() else []
    if not audits:
        return source
    record = max(audits, key=lambda path: (int(path.parent.name), path.parent.name))
    run = read_json(record.parent / 'run.json')
    run_config = run.get('config') if isinstance(run, dict) else None
    task = source.get('task')
    if not isinstance(task, dict) or not isinstance(run_config, dict) or run_config.get('task') != task:
        raise ValueError(f'交集检查与当前任务配置不符: {record}')
    queries = read_rows(data / 'queries.jsonl')
    validate_task(read_rows(data / 'pages.jsonl'), queries, read_json(data / 'qrels.json'), task)
    report = read_json(record)
    required = ('status', 'training_file', 'training_query_count', 'normalized_training_query_overlap')
    if not isinstance(report, dict) or any(key not in report for key in required):
        raise ValueError(f'交集检查记录格式无效: {record}')
    status, training_file, count, overlap = (report[key] for key in required)
    if status == 'not_checked':
        valid = training_file is None and count is None and overlap is None
    elif status == 'checked':
        valid = (isinstance(training_file, str) and bool(training_file.strip())
                 and type(count) is int and count >= 0
                 and isinstance(overlap, list) and all(isinstance(qid, str) for qid in overlap)
                 and len(overlap) == len(set(overlap))
                 and set(overlap).issubset({q['query_id'] for q in queries}))
    else:
        valid = False
    if not valid or run_config.get('training_file') != training_file:
        raise ValueError(f'交集检查状态、训练来源或查询 ID 无效: {record}')
    return dict(source, training_overlap=dict(report, record=str(record.resolve())),
                normalized_training_query_overlap=overlap)


def prepare_task(task, output, training=None, hr_reuse='data/hr'):
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    output = Path(output)
    completed = output / 'source.json'
    if completed.exists():
        source = read_json(completed)
        if source['task'] != task:
            raise ValueError('已有数据集配置不符，请使用新目录')
        queries = read_rows(output / 'queries.jsonl')
        validate_task(read_rows(output / 'pages.jsonl'), queries,
                      read_json(output / 'qrels.json'), task)
        run = output / 'runs' / str(time.time_ns())
        provenance(run, {'task': task, 'reuse': True,
                         'training_file': training['path'] if training else None})
        overlap = record_overlap(output, task, queries, training, run)
        print(f"{task['name']}: complete, reused", flush=True)
        # The source file is immutable historical evidence; return the current audit.
        return dict(source, training_overlap=overlap,
                    normalized_training_query_overlap=overlap['normalized_training_query_overlap'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'preparation-config.json'
    if manifest.exists() and read_json(manifest) != task:
        raise ValueError('未完成目录的固定版本不符')
    write_json(manifest, task)
    run = output / 'runs' / str(time.time_ns())
    provenance(run, {'task': task, 'reuse': False,
                     'training_file': training['path'] if training else None})
    started = time.perf_counter()
    files = HfApi().list_repo_files(task['dataset'], repo_type='dataset', revision=task['revision'])
    files = sorted(f for f in files if f.endswith('.parquet') and
                   f.split('/')[0] in ('corpus', 'queries', 'qrels', 'documents_metadata'))
    groups = {key: [] for key in ('corpus', 'queries', 'qrels', 'documents_metadata')}
    sources = []
    for filename in files:
        cached = try_to_load_from_cache(task['dataset'], filename, repo_type='dataset', revision=task['revision'])
        was_cached = isinstance(cached, str) and Path(cached).is_file()
        print(f"{task['name']}: {'reuse' if was_cached else 'download'} {filename}", flush=True)
        path = Path(hf_hub_download(task['dataset'], filename, repo_type='dataset', revision=task['revision']))
        groups[filename.split('/')[0]].append(path)
        sources.append({'file': filename, 'path': str(path), 'bytes': path.stat().st_size, 'already_cached': was_cached})
    download_seconds = time.perf_counter() - started
    if any(not groups[key] for key in ('corpus', 'queries', 'qrels')):
        raise ValueError('数据源缺少 corpus / queries / qrels')
    reuse = {}
    hr_reuse = Path(hr_reuse)
    if task['name'] == 'hr' and (hr_reuse / 'source.json').exists():
        if read_json(hr_reuse / 'source.json')['revision'] == task['revision']:
            reuse = {p['page_id']: p for p in read_rows(hr_reuse / 'pages.jsonl')}
    pages, texts = [], []
    images = output / 'images'
    images.mkdir(exist_ok=True)
    for path in groups['corpus']:
        parquet = pq.ParquetFile(path)
        columns = None if not reuse else [c for c in parquet.schema_arrow.names if c != 'image']
        for batch in parquet.iter_batches(batch_size=8, columns=columns):
            for row in batch.to_pylist():
                pid, doc_id = str(row['corpus_id']), str(row['doc_id'])
                if pid in reuse:
                    preview = Path(reuse[pid]['preview'])
                    if not preview.is_file():
                        raise ValueError(f'已有 HR 预览缺失: {preview}')
                else:
                    raw = row['image']['bytes']
                    with Image.open(io.BytesIO(raw)) as image:
                        extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(image.format)
                        if not extension:
                            raise ValueError(f'未支持的原始图像格式: {image.format}')
                        image.verify()
                    # The corpus row, not an untrusted upstream ID, determines the filename.
                    preview = images / f'{len(pages):06d}{extension}'
                    if not preview.exists():
                        preview.write_bytes(raw)
                filename = doc_id if doc_id.endswith('.pdf') else doc_id + '.pdf'
                pages.append({'page_id': pid, 'row_index': len(pages), 'doc_id': doc_id,
                    'page_number': int(row['page_number_in_doc']) + 1,
                    'dataset_page_number': row['page_number_in_doc'], 'preview': str(preview.resolve()),
                    'source': f"https://huggingface.co/datasets/{task['dataset']}/resolve/{task['revision']}/pdfs/{filename}",
                    'document_name': filename})
                texts.append({'page_id': pid, 'text': row.get('markdown') or ''})
            if len(pages) % 200 == 0:
                print(f"{task['name']}: prepared {len(pages)} pages", flush=True)
    all_queries = [row for p in groups['queries'] for row in pq.read_table(p).to_pylist()]
    queries = [dict(row, query_id=str(row['query_id'])) for row in all_queries
               if str(row['language']).casefold() in ('en', 'english')]
    queries.sort(key=lambda row: row['query_id'])
    qrels = {row['query_id']: {} for row in queries}
    for p in groups['qrels']:
        for row in pq.read_table(p).to_pylist():
            qid, pid, score = str(row['query_id']), str(row['corpus_id']), float(row['score'])
            if qid in qrels:
                if pid in qrels[qid]:
                    raise ValueError(f'重复 qrel: {qid}/{pid}')
                qrels[qid][pid] = score
    validate_task(pages, queries, qrels, task)
    overlap = record_overlap(output, task, queries, training, run)
    write_rows(output / 'pages.jsonl', pages)
    write_rows(output / 'texts.jsonl', texts)
    write_rows(output / 'queries.jsonl', queries)
    write_json(output / 'qrels.json', qrels)
    source = {'task': task, 'dataset': task['dataset'], 'revision': task['revision'],
        'scope': 'fixed external English queries; complete task corpus; no external tuning',
        'queries': len(queries), 'candidates': len(pages), 'sources': sources,
        'source_bytes': sum(f['bytes'] for f in sources),
        'new_source_file_bytes': sum(f['bytes'] for f in sources if not f['already_cached']),
        'extracted_image_bytes': sum(p.stat().st_size for p in images.iterdir() if p.is_file()),
        'reused_hr_pages': len(reuse), 'download_seconds': download_seconds,
        'preparation_seconds': time.perf_counter() - started - download_seconds,
        'training_overlap': overlap,
        'normalized_training_query_overlap': overlap['normalized_training_query_overlap'],
        'limitations': ['Full task retained even if an overlap is found; upstream training overlap and document-level isolation unverified'] +
            (['20 HR queries were previously used for flow checks; remaining HR queries were not selected by score'] if task['name'] == 'hr' else [])}
    write_json(completed, source)
    print(f"{task['name']}: {len(pages)} pages, {len(queries)} queries complete", flush=True)
    return source


def prepare_benchmark(config, output, training_data=None, task_name=None):
    training = training_reference(training_data)
    selected = [t for t in config['tasks'] if task_name is None or t['name'] == task_name]
    if not selected:
        raise ValueError('未知任务名')
    return [prepare_task(task, Path(output) / task['name'], training) for task in selected]
