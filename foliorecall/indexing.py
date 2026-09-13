"""Resumable page encoding; the public index appears only after finalization."""
import contextlib
import os
from pathlib import Path
import signal
import time

import numpy as np

from .documents import validate_pages
from .io import provenance, read_json, read_rows, write_json
from .search import encoding_identity, save_index


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def index_sizes(folder):
    folder = Path(folder)
    return {'deployment_index_bytes': sum((folder / name).stat().st_size
            for name in ('index.faiss', 'pages.jsonl', 'config.json') if (folder / name).is_file())}


def input_identity(pages_path, pages, config, chunk_size):
    inputs = []
    for row in pages:
        path = Path(row['preview']).resolve()
        stat = path.stat()
        inputs.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    source = Path(pages_path).parent / 'source.json'
    source = read_json(source) if source.is_file() else {}
    return {'encoding_identity': encoding_identity(config), 'chunk_size': chunk_size,
            'runtime': {key: config.get(key) for key in ('batch_size', 'device')},
            'pages': pages, 'inputs': inputs,
            'dataset': {key: source[key] for key in ('dataset', 'revision', 'task') if key in source}}


@contextlib.contextmanager
def exclusive_lock(path):
    # Supported release environment is Linux/WSL. A process death releases flock.
    import fcntl
    with Path(path).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'另一个进程正在使用此工作目录: {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def completed_chunks(work, pages, dimension):
    chunks, position = [], 0
    for path in sorted((work / 'chunks').glob('*.json')):
        record = read_json(path)
        if record['start'] != position or record['end'] <= position or record['end'] > len(pages):
            raise ValueError('缓存块不连续或行区间无效')
        if record['page_ids'] != [r['page_id'] for r in pages[position:record['end']]]:
            raise ValueError('缓存向量与页面行序不匹配')
        array = np.load(path.with_suffix('.npy'), mmap_mode='r', allow_pickle=False)
        if array.shape != (record['end'] - position, dimension) or array.dtype != np.float32:
            raise ValueError('缓存向量维度或精度不匹配')
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-3):
            raise ValueError('缓存块向量无效；保留损坏证据并使用新目录')
        del array
        position = record['end']
        chunks.append(record)
    return chunks, position


def build_index(config, pages_path, output, resume=False, chunk_size=64, max_seconds=None):
    if chunk_size < 1 or (max_seconds is not None and max_seconds <= 0):
        raise ValueError('chunk-size 和 max-seconds 必须为正')
    if resume and not Path(str(Path(output).resolve()) + '.work/manifest.json').is_file():
        raise ValueError('没有可恢复的建库清单')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('输出目录已存在，请使用新目录；完整索引不可覆盖')
    pages = validate_pages(read_rows(pages_path))
    if not pages:
        raise ValueError('页面清单为空')
    identity = input_identity(pages_path, pages, config, chunk_size)
    work = output.with_name(output.name + '.work')
    work.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(work / '.lock'):
        manifest = work / 'manifest.json'
        if manifest.exists():
            if not resume:
                raise ValueError('已有建库缓存，请显式使用 --resume')
            if read_json(manifest) != identity:
                raise ValueError('建库缓存的教师、数据、页面顺序或输入文件已变化；请使用新目录')
        elif resume:
            raise ValueError('没有可恢复的建库清单')
        else:
            atomic_json(manifest, identity)
        chunks, position = completed_chunks(work, pages, config['dimension'])
        run = work / 'runs' / str(time.time_ns())
        provenance(run, {'encoding': config, 'pages': str(Path(pages_path).resolve()),
            'output': str(output), 'resume': resume, 'chunk_size': chunk_size, 'max_seconds': max_seconds})
        return _encode_and_finish(config, pages, output, work, run, chunks, position, chunk_size, max_seconds)


def _encode_and_finish(config, pages, output, work, run, chunks, position, chunk_size, max_seconds):
    from .query import process_memory
    started = time.monotonic()
    limit = min(float(max_seconds or float('inf')), float(os.environ.get('FOLIORECALL_TASK_SECONDS', 'inf')))
    stop = [False]
    handlers = {}
    def pause(signum, frame):
        stop[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, pause)
    segment = {'started_unix': time.time(), 'start_page': position, 'model_loading_seconds': 0.,
               'probe_seconds': 0., 'encoding_seconds': 0., 'cache_write_seconds': 0., 'complete': False}
    atomic_json(run / 'segment.json', segment)
    model = None
    try:
        if position < len(pages):
            from .encoding import load_encoder, encode_pages, processing
            from PIL import Image
            import torch
            loading = time.monotonic()
            model = load_encoder(config)
            if config['device'] == 'cuda':
                torch.cuda.synchronize()
            segment['model_loading_seconds'] = time.monotonic() - loading
            segment['memory_after_loading'] = process_memory()
            probe_started = time.monotonic()
            with Image.open(pages[position]['preview']) as image:
                features = model.preprocess([image.convert('RGB')], prompt=config['prompt'], processing_kwargs=processing(config))
            write_json(run / 'encoding-probe.json', {'first_page_id': pages[position]['page_id'],
                'max_seq_length': model.max_seq_length, 'pooling': model[1].pooling_mode,
                'input_ids_shape': list(features['input_ids'].shape), 'image_grid_thw': features['image_grid_thw'].tolist()})
            del features
            segment['probe_seconds'] = time.monotonic() - probe_started
            while position < len(pages):
                if stop[0] or time.monotonic() - started >= limit:
                    break
                end = min(position + chunk_size, len(pages))
                tick = time.monotonic()
                vectors = encode_pages(model, pages[position:end], config)
                if config['device'] == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - tick
                if vectors.shape != (end-position, config['dimension']) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
                    raise ValueError('编码块向量或单位范数无效')
                tick = time.monotonic()
                folder = work / 'chunks'
                folder.mkdir(exist_ok=True)
                path = folder / f'{position:08d}.npy'
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.save(stream, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
                os.replace(path.with_suffix('.tmp'), path)
                record = {'start': position, 'end': end, 'page_ids': [r['page_id'] for r in pages[position:end]],
                          'encoding_seconds': elapsed, 'run': str(run)}
                atomic_json(path.with_suffix('.json'), record)
                segment['encoding_seconds'] += elapsed
                segment['cache_write_seconds'] += time.monotonic() - tick
                chunks.append(record)
                position = end
                segment['end_page'] = position
                segment['seconds_so_far'] = time.monotonic() - started
                atomic_json(run / 'segment.json', segment)
                print(f'cached {position}/{len(pages)}; chunk {elapsed:.2f}s', flush=True)
            segment.update(gpu=torch.cuda.get_device_name() if config['device'] == 'cuda' else None,
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if config['device'] == 'cuda' else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if config['device'] == 'cuda' else None)
        segment['end_page'] = position
        if position != len(pages):
            return {'complete': False, 'completed_pages': position, 'pages': len(pages), 'work': str(work),
                    'reason': 'pause or time limit; resume with the same command and --resume'}
        tick = time.monotonic()
        arrays = [np.load(work / 'chunks' / f"{r['start']:08d}.npy", allow_pickle=False) for r in chunks]
        vectors = np.concatenate(arrays)
        staging = work / ('final-' + run.name)
        save_index(vectors, pages, config, staging)
        segment['index_save_seconds'] = time.monotonic() - tick
        segment['complete'] = True
        segment['seconds'] = time.monotonic() - started
        segment['final_memory'] = process_memory()
        atomic_json(run / 'segment.json', segment)
        segments = [read_json(p) for p in sorted((work / 'runs').glob('*/segment.json'))]
        result = {'complete': True, 'pages': len(pages), 'work': str(work), 'segments': segments,
            'seconds': sum(s.get('seconds', s.get('seconds_so_far', 0)) for s in segments),
            'timing_scope': 'all recorded encoding process segments including loading/probes/cache/finalization; outer GPU runner accounts for unrecorded abrupt-exit time',
            'unclosed_segments': sum('seconds' not in s for s in segments),
            'encoding_seconds': sum(s['encoding_seconds'] for s in segments),
            'model_loading_seconds': sum(s['model_loading_seconds'] for s in segments),
            'probe_seconds': sum(s['probe_seconds'] for s in segments),
            'cache_write_seconds': sum(s['cache_write_seconds'] for s in segments),
            'index_save_seconds': sum(s.get('index_save_seconds', 0) for s in segments),
            'cache_bytes': sum(p.stat().st_size for p in (work / 'chunks').iterdir() if p.is_file()),
            **index_sizes(staging)}
        write_json(staging / 'build.json', result)
        # Same filesystem, previously absent destination: no half-written public index.
        if output.exists():
            raise ValueError('最终输出目录已出现，拒绝覆盖')
        os.rename(staging, output)
        return result
    except BaseException as exc:
        segment['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        segment['seconds'] = time.monotonic() - started
        segment['final_memory'] = process_memory()
        atomic_json(run / 'segment.json', segment)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
