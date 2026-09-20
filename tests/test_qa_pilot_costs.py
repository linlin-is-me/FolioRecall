from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.reconcile_qa_pilot import main, reconcile_pilot, reconcile_records, write_report


def call(key, amount, execution=None, attempt=1, status='estimated_from_usage'):
    return dict(id=key, cny=amount, status=status, at=1, usage='{}',
                context=json.dumps({'execution_id': execution, 'attempt': attempt}))


class PilotCostsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = self.root / 'costs.db'
        self.old = call('old', .2)
        self.before = dict(calls=[{k: self.old[k] for k in ('id', 'cny', 'status')}],
                           ledger=str(self.ledger), occupied_cny=.2, absolute_cap_cny=5.2)
        (self.root / 'translations').mkdir()
        (self.root / 'runs').mkdir()
        self.save('budget-before.json', self.before)
        self.save('translations/execution.json', {'execution_id': 'translations'})
        self.save('runs/manifest.json', {'execution_id': 'qa'})

    def save(self, path, value):
        (self.root / path).write_text(json.dumps(value), encoding='utf-8')

    def database(self, rows):
        with sqlite3.connect(self.ledger) as db:
            # Deliberately omit a primary key to exercise duplicate-ID validation.
            db.execute('CREATE TABLE calls (id TEXT,cny REAL,status TEXT,at REAL,usage TEXT,context TEXT)')
            db.executemany('INSERT INTO calls VALUES (:id,:cny,:status,:at,:usage,:context)', rows)

    def test_read_only_merge_known_unknown_other_and_cache_shares(self):
        rows = [self.old, call('t', .01, 'translations'), call('q', .05, 'qa'),
                call('retry', .8, 'qa', attempt=2, status='reserved_unknown_charge'), call('other', .03, 'other')]
        self.database(rows + [rows[1]])
        self.save('translations/translations.json', {str(i): {'cost_cny': 900, 'api_calls': [{'ledger_id': 't'}]} for i in range(3)})
        report = reconcile_pilot(self.root)
        self.assertTrue(report['ledger_reconciled'])
        self.assertEqual(report['pilot']['call_count'], 3)
        self.assertAlmostEqual(report['pilot']['known_cost_cny'], .06)
        self.assertAlmostEqual(report['pilot']['unknown_reservation_cny'], .8)
        self.assertAlmostEqual(report['groups']['other_executions']['occupied_cny'], .03)
        self.assertAlmostEqual(report['snapshot_delta_cny'], .89)
        self.assertAlmostEqual(report['phase']['occupied_cny'], 1.09)
        with sqlite3.connect(self.ledger) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM calls').fetchone()[0], 6)

    def test_historical_settlement_is_discrepancy_not_pilot_charge(self):
        old = dict(self.old, cny=.1, status='reserved_unknown_charge')
        self.database([old, call('q', .05, 'qa')])
        output = self.root / 'final.json'
        report = write_report(self.root, output)
        self.assertFalse(report['ledger_reconciled'])
        self.assertEqual(report['status'], 'reconciliation_discrepancy')
        self.assertAlmostEqual(report['pilot']['occupied_cny'], .05)
        self.assertAlmostEqual(report['historical_balance_delta_cny'], -.1)
        self.assertIn('historical_call_changed', {x['kind'] for x in report['discrepancies']})
        self.assertTrue(output.exists())

    def test_duplicate_id_conflict_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            reconcile_records(self.before, [self.old, dict(self.old, cny=.3)], 'translations', 'qa')

    def test_historical_output_never_overwritten_even_before_ledger_read(self):
        output = self.root / 'translation-stage-result.json'
        output.write_text('historical', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '不能覆盖'):
            write_report(self.root, output)
        self.assertEqual(output.read_text(encoding='utf-8'), 'historical')
        self.assertFalse(self.ledger.exists())

    def test_missing_qa_manifest_and_legacy_context_are_supported(self):
        (self.root / 'runs' / 'manifest.json').unlink()
        self.database([self.old, call('t', .01, 'translations'), dict(call('legacy', .03), context=None)])
        report = reconcile_pilot(self.root)
        self.assertTrue(report['ledger_reconciled'])
        self.assertIsNone(report['execution_ids']['qa'])
        self.assertEqual(report['groups']['qa_first']['call_count'], 0)
        self.assertEqual(report['groups']['other_executions']['ledger_ids'], ['legacy'])

    def test_missing_historical_call_or_bad_attempt_cannot_pass(self):
        report = reconcile_records(self.before, [call('q', .05, 'qa', attempt=3)], 'translations', 'qa')
        self.assertFalse(report['ledger_reconciled'])
        kinds = {x['kind'] for x in report['discrepancies']}
        self.assertIn('historical_call_missing', kinds)
        self.assertIn('invalid_qa_attempt', kinds)
        self.assertAlmostEqual(report['pilot']['occupied_cny'], .05)

    def test_unknown_status_is_accounted_but_discrepancy(self):
        report = reconcile_records(self.before, [self.old, call('q', .1, 'qa', status='unexpected')], 'translations', 'qa')
        self.assertFalse(report['ledger_reconciled'])
        self.assertAlmostEqual(report['pilot']['unclassified_cny'], .1)
        self.assertAlmostEqual(report['pilot']['occupied_cny'], .1)

    def test_cli_preserves_cap_and_reports_discrepancy_with_nonzero_exit(self):
        self.database([self.old, call('q', 5.1, 'qa')])
        output = self.root / 'over-cap.json'
        arguments = ['reconcile_qa_pilot', '--pilot', str(self.root), '--output', str(output)]
        with patch('sys.argv', arguments), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            main()
        self.assertEqual(error.exception.code, 1)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['absolute_cap_cny'], 5.2)
        self.assertAlmostEqual(report['a3_remaining_cny'], -.1)
        self.assertIn('absolute_cap_exceeded', {x['kind'] for x in report['discrepancies']})


if __name__ == '__main__':
    unittest.main()
