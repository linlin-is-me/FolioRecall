from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from scripts.prepare_qa_acceptance import ordered_keys, human_selection, file_binding, validate_root, CAP
from scripts.run_qa_acceptance_segment import batch_boundary, attempt_digest
from scripts.report_qa_acceptance import grouped, reconcile
from tests.test_qa_runtime import result


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.keys = ordered_keys('qa-acceptance-v1')
        self.jobs = [{'key': k, 'unit': 0, 'question': 'What?', 'task': {'reference_facts': ['fact']}} for k in self.keys]
        self.identity = {'suite': {'job_order': self.keys}}
        self.units = [{'agent': read_json('configs/agent-bailian.json')}]

    def row(self, index, reason='answered', attempt=1):
        r = result(reason)
        r.update(actual_model_requests=1, searches=1, evidence=[], translation_status='not_needed')
        return {'key': self.keys[index], 'attempt': attempt, 'state': 'finished', 'task': self.jobs[index]['task'],
                'result': r, 'answer_digest': digest(r['answer'])}

    def save(self, count):
        atomic_json(self.root/'runs/manifest.json', {'identity_digest': digest(self.identity)})
        for i in range(count):
            a = self.row(i)
            atomic_json(self.root/'runs/attempts'/digest(a['key'])/'attempt-0001.json', a)

    def review(self, a):
        return {'key': a['key'], 'attempt': a['attempt'], 'answer_digest': a['answer_digest'],
                'reviewer_type': 'assistant', 'human': 'not_checked', 'fact_correctness': None, 'answer_coverage': 0,
                'citation_support': None, 'insufficiency_handling': 0,
                'not_applicable': {'fact_correctness': 'no claims', 'citation_support': 'no claims'},
                'language_ok': None, 'language_output_is_system_fallback': True}

    def test_complete_plan_and_anonymous_human_selection(self):
        self.assertEqual(len({digest(k) for k in self.keys}), 240)
        self.assertEqual({k['domain'] for k in self.keys}, {'jrc'})
        self.assertEqual(sum(k['language'] == 'zh' for k in self.keys), 120)
        for n in range(5):
            self.assertEqual(len({k['task_id'] for k in self.keys[n*48:(n+1)*48]}), 8)
        selection = human_selection()['selection']
        self.assertEqual(sum(r['language'] == 'en' for r in selection), 5)
        self.assertEqual(sum(r['language'] == 'zh' for r in selection), 5)
        self.assertTrue(all(len(set(r['conditions'].values())) == 3 for r in selection))
        self.assertEqual(selection, human_selection()['selection'])

    def test_first_six_then_forty_two_without_crossing_batch(self):
        b = batch_boundary(self.root, self.identity, self.jobs, self.units, False)
        self.assertEqual((b['batch'], b['stop_after']), (1, 6))
        self.save(6)
        with self.assertRaisesRegex(ValueError, '首六项'):
            batch_boundary(self.root, self.identity, self.jobs, self.units, True)
        atomic_json(self.root/'checkpoints/first-six.json', {'attempts_digest': attempt_digest([self.row(i) for i in reversed(range(6))])})
        b = batch_boundary(self.root, self.identity, self.jobs, self.units, True)
        self.assertEqual((b['batch'], b['stop_after']), (1, 42))

    def test_next_batch_requires_all_prior_reviews_and_preserves_failures(self):
        self.save(48)
        with self.assertRaisesRegex(ValueError, '助手评审'):
            batch_boundary(self.root, self.identity, self.jobs, self.units, True)
        for i in range(48):
            atomic_json(self.root/'scoring'/f'{i}.json', self.review(self.row(i)))
        b = batch_boundary(self.root, self.identity, self.jobs, self.units, True)
        self.assertEqual((b['batch'], b['stop_after']), (2, 48))
        a = self.row(48, 'api_timeout')
        atomic_json(self.root/'runs/attempts'/digest(a['key'])/'attempt-0001.json', a)
        with self.assertRaisesRegex(ValueError, 'retry-one'):
            batch_boundary(self.root, self.identity, self.jobs, self.units, True)
        b = batch_boundary(self.root, self.identity, self.jobs, self.units, True, True)
        self.assertEqual((b['batch'], b['stop_after'], b['attempt']), (2, 1, 2))
        a = self.row(48, 'invalid_answer_or_evidence')
        atomic_json(self.root/'runs/attempts'/digest(a['key'])/'attempt-0001.json', a)
        self.assertEqual(batch_boundary(self.root, self.identity, self.jobs, self.units, True)['stop_after'], 47)

    def test_unknown_and_changed_source_reject_before_provider(self):
        self.save(1)
        a = self.row(0)
        a['state'] = 'running'
        atomic_json(self.root/'runs/attempts'/digest(a['key'])/'attempt-0001.json', a)
        with self.assertRaisesRegex(ValueError, '未知在途'):
            batch_boundary(self.root, self.identity, self.jobs, self.units, True)
        source = self.root/'tasks.json'
        atomic_json(source, {'frozen': True})
        atomic_json(self.root/'protocol.json', {'sources': file_binding([source])})
        atomic_json(source, {'frozen': False})
        with patch('scripts.prepare_qa_acceptance.prepare_suite') as prepare:
            with self.assertRaisesRegex(ValueError, '来源内容'):
                validate_root(self.root)
            prepare.assert_not_called()

    def test_missing_review_null_and_failure_have_distinct_counts(self):
        a, b = self.row(0, 'invalid_answer_or_evidence'), self.row(1)
        review = self.review(a)
        out = grouped([a, b], {(digest(a['key']), 1): review})
        self.assertEqual((out['terminal'], out['structurally_valid'], out['missing_review']), (2, 1, 1))
        self.assertEqual(out['metrics']['fact_correctness']['not_applicable'], 1)
        self.assertIsNone(out['metrics']['fact_correctness']['mean'])
        self.assertEqual(out['metrics']['answer_coverage']['denominator'], 1)
        self.assertEqual(out['language']['assessable'], 0)

    def test_translation_and_qa_costs_deduplicated_with_unknown_preserved(self):
        atomic_json(self.root/'agent.json', {'ledger': 'test', 'phase_budget_cny': CAP})
        for name, eid in [('translations', 't'), ('runs', 'q')]:
            atomic_json(self.root/name/'manifest.json', {'execution_id': eid, 'identity_digest': eid})
        tc = {'ledger_id': 't1', 'usage': {'completion_tokens': 1}, 'cost_cny': .1}
        qc = {'ledger_id': 'q1', 'usage': None, 'cost_cny': 0, 'reservation_cny': .8}
        atomic_json(self.root/'translations/translations.json', {'jrc-01': {'api_calls': [tc]}})
        a = self.row(0)
        a['result']['api_calls'] = [qc]
        atomic_json(self.root/'runs/attempts/a/attempt-0001.json', a)
        calls = {
            't': [{'id': 't1', 'cny': .1, 'status': 'estimated_from_usage', 'context': json.dumps({'execution_id': 't', 'task_id': 'jrc-01', 'attempt': 1})}],
            'q': [{'id': 'q1', 'cny': .8, 'status': 'reserved_unknown_charge', 'context': json.dumps(dict(a['key'], execution_id='q', attempt=1))}]}
        with patch('scripts.report_qa_acceptance.ledger_snapshot', side_effect=lambda cfg, eid: {'calls': calls[eid]}), \
                patch('scripts.report_qa_acceptance.budget_snapshot', return_value={'occupied_cny': 1.7, 'calls': calls['t']+calls['q']}):
            ledger = reconcile(self.root)
            self.assertEqual(ledger['known_cny'], .1)
            self.assertEqual(ledger['unknown_reserved_cny'], .8)
            self.assertAlmostEqual(ledger['occupied_cny'], .9)
            self.assertEqual(len(ledger['ledger_ids']), 2)


if __name__ == '__main__':
    unittest.main()
