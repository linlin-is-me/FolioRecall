"""Reconcile a pilot's shared ledger without changing charges or old results."""
import argparse
import json
import math
from pathlib import Path
import sqlite3

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, output_lock


def _unique(records, source):
    unique = {}
    for row in records:
        key = row.get('id')
        if not isinstance(key, str) or not key:
            raise ValueError(f'{source}: missing ledger ID')
        if key in unique and unique[key] != row:
            raise ValueError(f'{source}: conflicting duplicate ledger ID {key}')
        unique[key] = row
    return unique


def _amount(row):
    value = row.get('cny')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid amount for ledger ID {row.get('id')}")
    return value


def _totals(rows):
    rows = list(rows)
    return {
        'call_count': len(rows),
        'ledger_ids': sorted(row['id'] for row in rows),
        'known_cost_cny': math.fsum(_amount(row) for row in rows
                                   if row['status'] == 'estimated_from_usage'),
        'unknown_reservation_cny': math.fsum(_amount(row) for row in rows
                                            if row['status'] == 'reserved_unknown_charge'),
        'unclassified_cny': math.fsum(_amount(row) for row in rows
                                     if row['status'] not in ('estimated_from_usage', 'reserved_unknown_charge')),
        'occupied_cny': math.fsum(_amount(row) for row in rows),
    }


def reconcile_records(before, rows, translation_id, qa_id):
    """Use IDs, not timestamps or cached cost shares, to count each charge once."""
    if not translation_id or (qa_id is not None and (not qa_id or qa_id == translation_id)):
        raise ValueError('Translation and QA execution IDs must be distinct')
    previous = _unique(before['calls'], 'budget snapshot')
    current = _unique(rows, 'ledger')
    discrepancies = []
    for key, old in previous.items():
        if key not in current:
            discrepancies.append({'kind': 'historical_call_missing', 'ledger_id': key, 'before': old})
            continue
        changes = {field: {'before': value, 'after': current[key].get(field)}
                   for field, value in old.items() if current[key].get(field) != value}
        if changes:
            discrepancies.append({'kind': 'historical_call_changed', 'ledger_id': key, 'fields': changes})
    old_total = _totals(previous.values())
    if not math.isclose(old_total['occupied_cny'], before['occupied_cny'], abs_tol=1e-12, rel_tol=0):
        discrepancies.append({'kind': 'snapshot_total_mismatch',
                              'recorded': before['occupied_cny'], 'computed': old_total['occupied_cny']})
    groups = {name: [] for name in ('translation', 'qa_first', 'qa_recovery', 'qa_unattributed', 'other_executions')}
    new_rows = []
    for row in current.values():
        _amount(row)
        if row['status'] not in ('estimated_from_usage', 'reserved_unknown_charge'):
            discrepancies.append({'kind': 'unrecognized_cost_status', 'ledger_id': row['id'], 'status': row['status']})
        if row['id'] in previous:
            continue
        new_rows.append(row)
        try:
            context = json.loads(row.get('context') or '{}')
            if not isinstance(context, dict):
                raise ValueError('context is not an object')
        except (TypeError, ValueError):
            context = {}
            discrepancies.append({'kind': 'invalid_context', 'ledger_id': row['id']})
        execution = context.get('execution_id')
        if execution == translation_id:
            group = 'translation'
        elif qa_id is not None and execution == qa_id:
            attempt = context.get('attempt')
            group = 'qa_first' if type(attempt) is int and attempt == 1 else (
                'qa_recovery' if type(attempt) is int and attempt == 2 else 'qa_unattributed')
            if group == 'qa_unattributed':
                discrepancies.append({'kind': 'invalid_qa_attempt', 'ledger_id': row['id'], 'attempt': attempt})
        else:
            group = 'other_executions'
        groups[group].append(row)
    totals = {name: _totals(records) for name, records in groups.items()}
    phase = _totals(current.values())
    pilot = _totals(row for name, records in groups.items() if name != 'other_executions' for row in records)
    snapshot_delta = phase['occupied_cny'] - before['occupied_cny']
    new_total = _totals(new_rows)
    historical_delta = snapshot_delta - new_total['occupied_cny']
    if not math.isclose(historical_delta, 0, abs_tol=1e-12, rel_tol=0):
        discrepancies.append({'kind': 'historical_balance_delta', 'cny': historical_delta})
    cap = before['absolute_cap_cny']
    if phase['occupied_cny'] > cap + 1e-12:
        discrepancies.append({'kind': 'absolute_cap_exceeded', 'cap_cny': cap, 'occupied_cny': phase['occupied_cny']})
    return {
        'status': 'reconciled' if not discrepancies else 'reconciliation_discrepancy',
        'ledger_reconciled': not discrepancies,
        'execution_ids': {'translation': translation_id, 'qa': qa_id},
        'groups': totals, 'pilot': pilot, 'phase': phase, 'new_ledger_calls': new_total,
        'snapshot_occupied_cny': before['occupied_cny'], 'snapshot_delta_cny': snapshot_delta,
        'historical_balance_delta_cny': historical_delta,
        'absolute_cap_cny': cap, 'a3_remaining_cny': cap - phase['occupied_cny'],
        'phase_remaining_cny': 150 - phase['occupied_cny'],
        'discrepancies': discrepancies,
        'scope': 'usage estimates and unknown reservations, not invoice charges; cache shares are not additional charges',
    }


def reconcile_pilot(pilot):
    root = Path(pilot)
    before = read_json(root / 'budget-before.json')
    translation_id = read_json(root / 'translations' / 'execution.json')['execution_id']
    manifest = root / 'runs' / 'manifest.json'
    qa_id = read_json(manifest)['execution_id'] if manifest.exists() else None
    ledger = Path(before['ledger']).resolve()
    with sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute('SELECT * FROM calls ORDER BY at,id')]
    return reconcile_records(before, rows, translation_id, qa_id) | {'ledger': str(ledger), 'pilot_directory': str(root.resolve())}


def write_report(pilot, output):
    output = Path(output)
    if output.exists():
        raise ValueError('费用核对输出已存在；不能覆盖历史结果')
    with output_lock(output):
        if output.exists():
            raise ValueError('费用核对输出已存在；不能覆盖历史结果')
        report = reconcile_pilot(pilot)
        atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = write_report(args.pilot, args.output)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    if not report['ledger_reconciled']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
