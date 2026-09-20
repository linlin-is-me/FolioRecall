from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_translation import translation_identity
from foliorecall.qa_evaluation import RATINGS
from scripts.cache_qa_translations import execute_cache
from scripts.run_qa_development_segment import cache_gate, segment_boundary, check_budget, main as segment_main
from scripts.analyze_qa_development import analyze, paired_candidates
from tests.test_qa_runtime import job, result


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = read_json('configs/agent-bailian.json')
        self.jobs = [dict(job(i, 'zh'), unit=0) for i in range(2)]
        self.units = [{'agent': self.config, 'jobs': self.jobs}]
        self.identity = {'suite': {'suite_id': 'mock'}, 'code': 1}
        self.calls = []

    def record(self, question):
        return {'identity': translation_identity(question, self.config), 'question': question, 'query': 'What?',
                'requests': 1, 'output_tokens': 5, 'seconds': .1, 'cost_cny': .001,
                'api_calls': [{'ledger_id': 'seed-id', 'usage': {'completion_tokens': 5}, 'cost_cny': .001}]}

    def translated(self, question, config, provider):
        self.calls.append(question)
        return self.record(question)

    def execute(self, **kwargs):
        with patch('scripts.cache_qa_translations.cache_translation', side_effect=self.translated):
            return execute_cache(self.units, self.identity, self.root/'translations',
                provider_factory=lambda c: type('Provider', (), {})(), **kwargs)

    def test_seed_and_completed_resume_no_new_requests(self):
        seed = self.root/'seed.json'
        atomic_json(seed, {'hr:0': self.record(self.jobs[0]['question'])})
        self.assertEqual(self.execute(seed_cache=seed)['completed'], 2)
        self.assertEqual(len(self.calls), 1)
        self.execute(seed_cache=seed, resume=True)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(read_json(self.root/'translations/translations.json')['hr:0'], read_json(seed)['hr:0'])

    def test_atomic_record_survives_summary_interruption(self):
        original = atomic_json
        def fail(path, value):
            if Path(path).name == 'translations.json' and value:
                raise OSError('interrupted after record')
            original(path, value)
        with patch('scripts.cache_qa_translations.atomic_json', side_effect=fail), self.assertRaises(OSError):
            self.execute()
        self.execute(resume=True)
        self.assertEqual(len(self.calls), 2)

    def test_unknown_inflight_refuses_provider_and_retranslation(self):
        with patch('scripts.cache_qa_translations.cache_translation', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                execute_cache(self.units, self.identity, self.root/'translations', provider_factory=lambda c: type('P', (), {})())
        with self.assertRaisesRegex(ValueError, '未知在途'):
            self.execute(resume=True)
        self.assertEqual(self.calls, [])

    def test_changed_budget_or_seed_rejects_before_provider(self):
        self.execute()
        self.config['phase_budget_cny'] += 1
        with self.assertRaisesRegex(ValueError, '身份'):
            self.execute(resume=True)
        self.assertEqual(len(self.calls), 2)

    def test_dry_run_and_invalid_seed_do_not_create_output(self):
        self.assertEqual(self.execute(dry_run=True)['queries'], 2)
        self.assertEqual(list(self.root.iterdir()), [])
        seed = self.root/'seed.json'
        broken = self.record(self.jobs[0]['question']); broken.pop('seconds')
        atomic_json(seed, {'hr:0': broken})
        with self.assertRaises(ValueError): self.execute(seed_cache=seed)
        self.assertFalse((self.root/'translations').exists())

    def test_cache_complete_required_and_repeat_rejects_cache(self):
        jobs = []
        for c in ('visual-rag', 'hybrid-rag', 'hybrid-agent'):
            for i in range(40):
                for lang in ('en', 'zh'):
                    j = dict(job(i, lang, c), unit=0)
                    j['translation'] = self.record(j['question']) if lang == 'zh' else None
                    jobs.append(j)
        self.assertEqual(cache_gate(self.units, jobs)['cache_hits'], 120)
        jobs[1]['translation']['seconds'] = None
        with self.assertRaises(ValueError): cache_gate(self.units, jobs)
        with self.assertRaises(ValueError): cache_gate(self.units, jobs[:180], True)
        self.assertEqual(cache_gate(self.units, [dict(j, translation=None) for j in jobs[:180]], True)['live_translations'], 90)

    def saved(self, j, attempt=1, reason='answered'):
        r = result(reason); r['model_requests'] = 1
        return {'key': j['key'], 'task': j['task'], 'state': 'finished', 'attempt': attempt,
                'result': r, 'answer_digest': digest(r['answer']), 'finished_at': 10.}

    def test_domain_boundary_retry_is_separate_identity_fail_closed(self):
        j = [dict(job(i), unit=0) for i in range(3)]
        j[2]['key']['domain'] = 'finance_en'
        output = self.root/'runs'
        self.assertEqual(segment_boundary(output, self.identity, j, self.units, False)['stop_after'], 2)
        atomic_json(output/'manifest.json', {'identity_digest': digest(self.identity)})
        a = self.saved(j[0], reason='api_http_503')
        atomic_json(output/'attempts'/digest(j[0]['key'])/'attempt-0001.json', a)
        with self.assertRaisesRegex(ValueError, 'retry-one'):
            segment_boundary(output, self.identity, j, self.units, True)
        self.assertEqual(segment_boundary(output, self.identity, j, self.units, True, True)['stop_after'], 1)
        atomic_json(output/'attempts'/digest(j[0]['key'])/'attempt-0002.json', self.saved(j[0], 2))
        self.assertEqual(segment_boundary(output, self.identity, j, self.units, True)['stop_after'], 1)
        with self.assertRaisesRegex(ValueError, '身份'):
            segment_boundary(output, {'changed': True}, j, self.units, True)

    def test_unknown_attempt_blocks_next_domain(self):
        j = dict(job(), unit=0); output = self.root/'runs'
        atomic_json(output/'manifest.json', {'identity_digest': digest(self.identity)})
        atomic_json(output/'attempts'/digest(j['key'])/'attempt-0001.json',
                    {'key': j['key'], 'attempt': 1, 'state': 'running'})
        with self.assertRaisesRegex(ValueError, '未知在途'):
            segment_boundary(output, self.identity, [j], self.units, True)

    def test_budget_is_frozen_and_reservations_remain_occupied(self):
        config = dict(self.config, ledger='/original', phase_budget_cny=62.)
        frozen = {'ledger': '/original', 'absolute_cap_cny': 62.}
        with patch('scripts.run_qa_development_segment.budget_snapshot', return_value={'occupied_cny': 60.}):
            self.assertFalse(check_budget(config, frozen, 1.)['passed'])
        with self.assertRaises(ValueError): check_budget(dict(config, phase_budget_cny=63.), frozen, 1.)

    def segment(self, record='segment', **mocks):
        config = dict(self.config, ledger='/original', phase_budget_cny=62.)
        units = [{'agent': config, 'student': {'device': 'cpu'}, 'jobs': self.jobs}]
        identity = {'suite': {'suite_id': 'mock', 'inter_task_cooldown_seconds': 65}}
        atomic_json(self.root/'budget.json', {'ledger': '/original', 'absolute_cap_cny': 62.})
        atomic_json(self.root/'forecast.json', {'remaining_cny': 10., 'evidence': 'synthetic source'})
        argv = ['segment', '--suite', 'unused', '--output', str(self.root/'run'), '--record', str(self.root/record),
                '--budget', str(self.root/'budget.json'), '--forecast', str(self.root/'forecast.json')]
        with patch('sys.argv', argv), patch('scripts.run_qa_development_segment.prepare_suite', return_value=(units, identity)), \
             patch('scripts.run_qa_development_segment.cache_gate', return_value={'planned': 240}), \
             patch('scripts.run_qa_development_segment.budget_snapshot', return_value={'occupied_cny': 2.}), \
             patch('time.sleep', side_effect=AssertionError('No startup wait')) as observe, \
             patch('scripts.run_qa_development_segment.execute_plan', return_value={'stopped': 'stop_after', 'attempts_new': 2, 'pending': 0}) as execute:
            code = segment_main()
            return code, observe.call_count, execute.call_count

    def test_no_startup_observation_and_log_reuse_refused(self):
        self.assertEqual(self.segment(), (75, 0, 1))
        with self.assertRaisesRegex(ValueError, '日志已存在'): self.segment()

    def review(self, a):
        return {'key': a['key'], 'attempt': a['attempt'], 'answer_digest': a['answer_digest'],
            'reviewer_type': 'assistant', 'human': 'not_checked', 'primary_error': 'tool_decision',
            'primary_error_reason': 'Candidate was visible but unread.', 'secondary_factors': ['evidence_use'],
            **{m: 0. for m in RATINGS}}

    def test_main_error_once_language_separate_and_repeat_missing_unknown(self):
        j = [job(i) for i in range(3)]
        for i,k in enumerate(j): k['key'].update(task_id='hr:0', repeat=i)
        rows = [self.saved(j[0]), self.saved(j[1])]
        reviews = [self.review(a) for a in rows]
        report = analyze({'job_keys': [k['key'] for k in j]}, rows, reviews)
        self.assertEqual(report['primary_errors']['tool_decision']['answers'], 2)
        self.assertEqual(report['primary_errors']['tool_decision']['logical_tasks'], 1)
        self.assertIsNone(report['candidate_target'])
        self.assertIsNone(report['three_round_variation'][0]['ranges']['answer_coverage'])
        bad = deepcopy(reviews); bad[0]['key']['language'] = 'zh'
        with self.assertRaises(ValueError): analyze({'job_keys': [k['key'] for k in j]}, rows, bad)
        with self.assertRaises(ValueError): analyze({'job_keys': [k['key'] for k in j]}, rows, reviews+[reviews[0]])

    def test_candidate_pairs_do_not_mix_tasks_or_attempts(self):
        a = self.saved(job()); b = deepcopy(a); b['key']['suite'] = 'candidate'
        pair = paired_candidates([a], [b], [self.review(a)], [self.review(b)])
        self.assertEqual(pair[0]['score_delta']['answer_coverage'], 0)
        b['task']['reference_facts'] = ['changed']
        with self.assertRaises(ValueError): paired_candidates([a], [b], [self.review(a)], [self.review(b)])

    def test_condition_pairs_and_three_round_observed_ranges(self):
        jobs = []
        for condition in ('visual-rag', 'hybrid-rag'):
            for repeat in range(3):
                j = job(0, 'en', condition); j['key']['repeat'] = repeat; jobs.append(j)
        rows = [self.saved(j) for j in jobs]
        for i,a in enumerate(rows):
            a['result']['actual_model_requests'] = 1+i%3
            a['result']['answer']['claims'] = [{'text': 'Synthetic claim', 'evidence_page_ids': ['485']}]
            a['answer_digest'] = digest(a['result']['answer'])
        reviews = [self.review(a) for a in rows]
        report = analyze({'job_keys': [j['key'] for j in jobs]}, rows, reviews)
        observed = report['three_round_variation'][0]
        self.assertEqual(observed['ranges']['actual_requests'], [1,3])
        self.assertEqual(observed['observations'][0]['citation_page_ids'], ['485'])
        pairs = [p for p in report['condition_pairs'] if p['variant'] == 'hybrid-rag']
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(p['paired_reviewed'] for p in pairs))


if __name__ == '__main__': unittest.main()
