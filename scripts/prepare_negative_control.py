"""Prepare two matched arms from explicit assistant review decisions, not scores."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import read_json, read_rows, write_json, write_rows, provenance


def main():
    from foliorecall.trainer import training_dataset, batch_plan, validate_training_data
    audit = Path('outputs/stage2/negative-audit')
    source = Path('data/vdr-stage2')
    output = Path('data/vdr-stage2-control')
    if output.exists():
        raise ValueError('Control data already exists; do not overwrite a frozen comparison')
    rows = read_rows(audit / 'samples.jsonl')
    mapping = {p['page_id']: p for p in read_rows(source / 'pages.jsonl')}
    dev = read_rows(source / 'dev/pages.jsonl')
    dev_ids = {p['page_id'] for p in dev}
    notes = read_json(audit / 'review-notes.json')
    decisions = {r['query_id']: r['replacement_review'] for r in notes
                 if r.get('replacement_review', {}).get('decision') == 'replace_in_control_only'}
    if not decisions or not set(decisions).issubset(r['query_id'] for r in rows):
        raise ValueError('No valid reviewed replacements')
    changed = copy.deepcopy(rows)
    for row in changed:
        if row['query_id'] in decisions:
            pid = decisions[row['query_id']]['replacement']
            if pid not in row['valid_negative_ids'] or pid in dev_ids or pid == row['positive'] or pid not in mapping:
                raise ValueError('Replacement violates upstream pool, availability or split constraints')
            row['negative'] = pid
    plans = [batch_plan(training_dataset(arm, mapping), 4, 42) for arm in (rows, changed)]
    if plans[0] != plans[1] or len(plans[0]) != 64 or any(len(b) != 4 for b in plans[0]):
        raise ValueError('Arms need an explicitly shared conflict-free batch plan before training')
    config = read_json('configs/lora-baseline.json')
    for name, arm in [('original', rows), ('filtered', changed)]:
        folder = output / name
        selected = dev_ids | {r[k] for r in arm for k in ('positive', 'negative')}
        pages = [mapping[pid] for pid in sorted(selected)]
        split = {'train': arm, 'dev': read_json(source / 'split.json')['dev']}
        validate_training_data(split, pages, dev)
        write_json(folder / 'split.json', split)
        write_rows(folder / 'pages.jsonl', pages)
        (folder / 'dev').symlink_to((source / 'dev').resolve(), target_is_directory=True)
    provenance(output, config)
    write_json(output / 'comparison.json', {'seed': 42, 'queries': 256, 'steps_per_arm': 64,
        'batches': plans[0], 'replacements': decisions, 'review': str(audit / 'review-notes.json'),
        'dev': str((source / 'dev').resolve()), 'validation': 'same64x4 batches; all256 rows once; upstream and page split checked',
        'scope': 'conservative exclusion of assistant-reviewed disputed negatives; labels unchanged'})
    print('Prepared matched64x4 arms; replacements:', len(decisions))


if __name__ == '__main__':
    main()
