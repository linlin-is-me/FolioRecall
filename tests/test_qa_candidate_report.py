import unittest

from foliorecall.qa_evaluation import RATINGS
from scripts.report_qa_candidate import group_summary


class CandidateReportTests(unittest.TestCase):
    def fixture(self):
        keys = [('visual-rag', 'example', str(i), 'en', 0) for i in range(2)]
        base, candidate, br, cr = {}, {}, {}, {}
        pairs = []
        for i, key in enumerate(keys):
            for runs, valid, cost in ((base, i == 0, 2), (candidate, True, 3)):
                runs[key] = {'result': {'completed': valid, 'actual_model_requests': 1,
                    'estimated_cost_cny': cost, 'timing': {'task_seconds': 5}}}
            br[key] = dict.fromkeys(RATINGS, 0.5 if i == 0 else None)
            cr[key] = dict.fromkeys(RATINGS, 1)
            br[key]['answer_coverage'] = 0.5 if i == 0 else 0
            for reviews, fallback in ((br, i == 1), (cr, False)):
                reviews[key].update(language_ok=True, language_output_is_system_fallback=fallback)
            pairs.append({'cell': key, 'score_delta': {
                m: None if br[key][m] is None else cr[key][m] - br[key][m]
                for m in RATINGS}})
        return pairs, base, candidate, br, cr

    def test_null_scores_use_common_denominator_without_dropping_failed_coverage(self):
        result = group_summary(*self.fixture())
        facts = result['metrics']['fact_correctness']
        self.assertEqual(facts['paired_denominator'], 1)
        self.assertEqual((facts['baseline_non_null'], facts['candidate_non_null']), (1, 2))
        self.assertEqual(facts['delta_mean'], 0.5)
        coverage = result['metrics']['answer_coverage']
        self.assertEqual(coverage['paired_denominator'], 2)
        self.assertEqual(coverage['delta_mean'], 0.75)
        self.assertEqual(result['resources']['known_cost_cny']['baseline_total'], 4)

    def test_system_fallback_is_not_a_successful_model_language_answer(self):
        result = group_summary(*self.fixture())
        language = result['language_ok']
        self.assertEqual(language['baseline'], 1)
        self.assertEqual(language['candidate'], 2)
        self.assertEqual(language['common_assessable'], 1)
        self.assertEqual(language['improved'], 0)
        self.assertEqual(result['structure_valid']['improved'], 1)

    def test_no_common_fact_score_stays_null(self):
        pairs, base, candidate, br, cr = self.fixture()
        for key in br:
            br[key]['fact_correctness'] = None
        for pair in pairs:
            pair['score_delta']['fact_correctness'] = None
        facts = group_summary(pairs, base, candidate, br, cr)['metrics']['fact_correctness']
        self.assertEqual(facts['paired_denominator'], 0)
        self.assertIsNone(facts['baseline_mean'])
        self.assertIsNone(facts['candidate_mean'])
        self.assertIsNone(facts['delta_mean'])


if __name__ == '__main__':
    unittest.main()
