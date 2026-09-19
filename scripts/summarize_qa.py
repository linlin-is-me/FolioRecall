"""Import attributed reviews without changing primary attempts."""
import argparse
from pathlib import Path
from foliorecall.io import read_json, read_rows
from foliorecall.qa_evaluation import summarize_attempts
from foliorecall.qa_storage import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True)
    p.add_argument('--reviews')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():
        raise ValueError('评分输出已存在')
    root=Path(a.runs)
    attempts=[read_json(f) for f in sorted((root/'attempts').glob('*/attempt-*.json'))]
    from foliorecall.qa_runtime import ledger_snapshot
    manifest=read_json(root/'manifest.json')
    identity=manifest['identity']
    configs=[u['agent'] for u in identity['units']] if 'units' in identity else [identity['agent']]
    ledger=ledger_snapshot(configs,manifest['execution_id'])
    keys=manifest['job_keys']
    atomic_json(a.output,summarize_attempts(attempts,len(keys),read_rows(a.reviews) if a.reviews else None,
               planned_keys=keys,ledger_records=ledger['calls'])|{'ledger':ledger})


if __name__ == '__main__':
    main()
