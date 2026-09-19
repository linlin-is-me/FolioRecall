"""Small durable QA records; no model or provider initialization here."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def restart_gap(attempt, first=None):
    """Derive an observed wall-clock interval without guessing missing elapsed time."""
    if attempt.get('attempt') != 2:
        return {'gap_seconds': 0., 'gap_status': 'not_applicable'}
    started, finished = attempt.get('started_at'), (first or {}).get('finished_at')
    if started is None or finished is None:
        return {'gap_seconds': None, 'gap_status': 'missing_timestamp'}
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (started, finished)):
        return {'gap_seconds': None, 'gap_status': 'invalid_timestamp'}
    seconds = started - finished
    if not math.isfinite(seconds):
        return {'gap_seconds': None, 'gap_status': 'nonfinite_interval'}
    if seconds < 0:
        return {'gap_seconds': None, 'gap_status': 'wall_clock_rollback'}
    return {'gap_seconds': seconds, 'gap_status': 'valid'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != 'nt':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_lock(output):
    """Sibling lock also protects first creation; validated on Ubuntu/WSL."""
    import fcntl
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_name(output.name+'.lock'), 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('QA 输出目录正在使用') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
