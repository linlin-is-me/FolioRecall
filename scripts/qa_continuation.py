"""Explicit, immutable continuation of never-started QA keys; no crash retry."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

from foliorecall.io import read_json
from foliorecall.qa_storage import digest, atomic_json, output_lock


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot(source):
    source = Path(source).resolve()
    manifest = read_json(source/'manifest.json')
    require(manifest['identity_digest'] == digest(manifest['identity']), 'Source identity corrupt')
    require(manifest['job_keys'] == manifest['identity']['suite']['job_order'], 'Source plan mismatch')
    planned = {digest(k): k for k in manifest['job_keys']}
    require(len(planned) == len(manifest['job_keys']), 'Duplicate source key')
    records, hashes, seen = [], {}, set()
    for path in sorted(source.glob('attempts/*/attempt-*.json')):
        a = read_json(path)
        key = digest(a['key'])
        require(key in planned and a['key'] == planned[key], 'Attempt outside source plan')
        require(path.parent.name == key and path.name == f'attempt-{a["attempt"]:04d}.json', 'Attempt path mismatch')
        require(a['attempt'] == 1 and key not in seen, 'Continuation source must contain first attempts only')
        require(a['state'] in ('running', 'finished'), 'Unknown source attempt state')
        if a['state'] == 'finished':
            require(a['answer_digest'] == digest(a['result']['answer']), 'Source answer digest mismatch')
        else:
            require('result' not in a, 'Unexpected partial result; audit separately')
        hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(a)
        seen.add(key)
    order = {digest(k): i for i,k in enumerate(manifest['job_keys'])}
    records.sort(key=lambda a: (order[digest(a['key'])], a['attempt']))
    return manifest, records, hashes


def freeze(source, audit, target, reason):
    source, target, audit = Path(source).resolve(), Path(target).resolve(), Path(audit).resolve()
    require(source != target and not target.is_relative_to(source) and not source.is_relative_to(target), 'Continuation must be separate from source')
    require(not target.exists() and bool(reason.strip()), 'New target and explicit disposition required')
    m, attempts, hashes = snapshot(source)
    running = [a for a in attempts if a['state'] == 'running']
    record = read_json(audit)
    require(len(running) == 1 and record['status'] == 'blocked_unknown_task_budget', 'One audited interruption required')
    require(record['interrupted_key'] == running[0]['key'] and record['started'] == len(attempts), 'Interruption audit mismatch')
    seen = {digest(a['key']) for a in attempts}
    pending = [k for k in m['job_keys'] if digest(k) not in seen]
    require(len(pending) == record['not_started'], 'Audited pending count mismatch')
    return dict(schema_version=1, source_directory=str(source), target_directory=str(target),
                source_manifest_digest=digest(m), source_execution_id=m['execution_id'],
                source_attempt_hashes=hashes, audit=str(audit), audit_digest=digest(record),
                excluded=[dict(key=a['key'], attempt=a['attempt'], reason=reason) for a in running],
                pending_keys=pending, disposition='retain_unknown_never_retry',
                completion_policy='all_available_terminal_reviewed_with_declared_missing')


def validate_source(descriptor):
    d = descriptor
    require(d['schema_version'] == 1 and d['disposition'] == 'retain_unknown_never_retry', 'Unsupported disposition')
    source, target = Path(d['source_directory']).resolve(), Path(d['target_directory']).resolve()
    require(source != target and not target.is_relative_to(source) and not source.is_relative_to(target), 'Source/target overlap')
    m, attempts, hashes = snapshot(source)
    require(digest(m) == d['source_manifest_digest'] and m['execution_id'] == d['source_execution_id'], 'Source manifest changed')
    require(hashes == d['source_attempt_hashes'], 'Source attempts changed')
    require(digest(read_json(d['audit'])) == d['audit_digest'], 'Audit changed')
    running = [dict(key=a['key'], attempt=a['attempt']) for a in attempts if a['state'] == 'running']
    require(len(running) == 1 and running == [dict(key=e['key'], attempt=e['attempt']) for e in d['excluded']], 'Excluded key mismatch')
    require(all(e.get('reason', '').strip() for e in d['excluded']), 'Missing disposition reason')
    seen = {digest(a['key']) for a in attempts}
    require(d['pending_keys'] == [k for k in m['job_keys'] if digest(k) not in seen], 'Pending keys must exactly match never-started source keys')
    return m, attempts


def control_code():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('qa_continuation.py', 'run_qa_development_segment.py')}


def derive(descriptor, identity, jobs, output):
    m, attempts = validate_source(descriptor)
    require(Path(output).resolve() == Path(descriptor['target_directory']).resolve(), 'Wrong continuation target')
    require(digest(identity) == m['identity_digest'], 'Current full execution identity differs from source')
    require([j['key'] for j in jobs] == m['job_keys'], 'Full jobs differ from source')
    selected = {digest(k) for k in descriptor['pending_keys']}
    subset = [j for j in jobs if digest(j['key']) in selected]
    derived = deepcopy(identity)
    derived['suite']['job_order'] = descriptor['pending_keys']
    derived['continuation'] = dict(descriptor=descriptor, descriptor_digest=digest(descriptor), control_code=control_code())
    return derived, subset, attempts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'audit', 'target', 'output', 'reason'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    require(not Path(a.output).exists(), 'Descriptor already exists')
    with output_lock(a.source):
        d = freeze(a.source, a.audit, a.target, a.reason)
        atomic_json(a.output, d)
    print({'pending': len(d['pending_keys']), 'excluded': len(d['excluded'])})


if __name__ == '__main__':
    main()
