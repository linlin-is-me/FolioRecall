from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts.a6_review import convert, rating, summarize
from foliorecall.qa_storage import digest, atomic_json
from foliorecall.qa_evaluation import summarize_attempts
from tests.test_qa_runtime import job, result


class HumanReviewTests(unittest.TestCase):
    def answer(self, facts=True, claims=True):
        j = job()
        j['task']['reference_facts'] = ['required'] if facts else []
        j['task']['answerable'] = facts
        r = result()
        if claims:
            r['answer']['claims'] = [{'text': '42', 'evidence_page_ids': ['p1']}]
        return {'key': j['key'], 'task': j['task'], 'attempt': 1, 'state': 'finished',
                'result': r, 'answer_digest': digest(r['answer'])}

    def entry(self, a):
        return {'answer_digest': a['answer_digest'],
                'claims': [{'fact': 1, 'citation': .5, 'notes': 'partial'} for _ in a['result']['answer']['claims']],
                'facts': [{'score': .5, 'notes': 'partial'} for _ in a['task']['reference_facts']],
                'insufficiency': 1, 'language_ok': True, 'confirmed': True}

    def test_partial_and_pending_are_not_imported_as_zero(self):
        a = self.answer()
        e = self.entry(a)
        e['claims'][0]['fact'] = 'pending'
        self.assertIsNone(convert(e, a))
        e['claims'][0]['fact'] = None
        self.assertIsNone(convert(e, a))

    def test_binding_and_boolean_ratings_rejected(self):
        a = self.answer()
        e = self.entry(a)
        e['answer_digest'] = 'wrong'
        with self.assertRaises(ValueError): convert(e, a)
        with self.assertRaises(ValueError): rating(True)

    def test_full_human_record_uses_existing_validation(self):
        a = self.answer()
        h = convert(self.entry(a), a)
        s = summarize_attempts([a], 1, [h])
        self.assertEqual(h['reviewer_type'], 'human')
        self.assertEqual(h['citation_support'], .5)
        self.assertEqual(next(iter(s['semantic_reviews'].values()))['reviewed'], 1)

    def test_no_claims_and_no_required_facts_are_not_full_scores(self):
        a = self.answer(facts=False, claims=False)
        h = convert(self.entry(a), a)
        self.assertIsNone(h['fact_correctness'])
        self.assertIsNone(h['citation_support'])
        self.assertIsNone(h['answer_coverage'])
        summarize_attempts([a], 1, [h])

    def test_empty_answer_with_required_explanation_has_zero_coverage(self):
        a = self.answer(claims=False)
        h = convert(self.entry(a), a)
        self.assertEqual(h['answer_coverage'], 0)
        summarize_attempts([a], 1, [h])

    def test_delegated_review_never_becomes_human(self):
        a = self.answer()
        r = convert(self.entry(a), a, 'assistant')
        self.assertEqual(r['reviewer_type'], 'assistant')
        self.assertEqual(r['human'], 'not_checked')

    def test_assistant_export_cannot_be_imported_with_human_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root/'private-mapping.json', {})
            atomic_json(root/'ratings.json', {'reviewer_type': 'assistant'})
            with self.assertRaisesRegex(ValueError, '助手复核不得转为人工'):
                summarize(root, root/'ratings.json', root/'report.json')


if __name__ == '__main__':
    unittest.main()
