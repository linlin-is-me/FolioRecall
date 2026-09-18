from pathlib import Path
import tempfile
import unittest
from foliorecall.io import write_json, write_rows
from foliorecall.qa_evaluation import validate_tasks, summarize_runs


class QAEvaluationTests(unittest.TestCase):
    def test_freeze_and_bilingual_requirements(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            task={'task_id':'t1','question_en':'What?', 'question_zh':'什么？','answerable':False,
                  'reference_facts':[],'evidence_page_ids':[],'rubric':'State evidence is insufficient',
                  'review':{'assistant':'verified','human':'not_checked'}}
            write_rows(root/'tasks.jsonl',[task])
            write_json(root/'status.json',{'qa_reference_frozen':False,'evaluation_allowed':False})
            with self.assertRaises(ValueError):
                validate_tasks(root/'tasks.jsonl',[])
            write_json(root/'status.json',{'qa_reference_frozen':True,'evaluation_allowed':True})
            self.assertEqual(len(validate_tasks(root/'tasks.jsonl',[])[0]),1)
            write_rows(root/'tasks.jsonl',[dict(task,question_zh=None)])
            with self.assertRaises(ValueError):
                validate_tasks(root/'tasks.jsonl',[])

    def test_languages_and_review_provenance_stay_separate(self):
        base={'task_id':'domain:1','repeat':0,'completed':True,'timing':{'task_seconds':2},
              'estimated_cost_cny':.02,'cost_complete':True,'model_requests':3}
        rows=[dict(base,language=lang) for lang in ('en','zh')]
        summary=summarize_runs(rows)
        self.assertEqual(summary['logical_tasks'],1)
        self.assertEqual(summary['semantic_quality_status'],'pending_review')
        self.assertEqual(summary['by_language']['en']['runs'],1)
        review={'task_id':'domain:1','repeat':0,'language':'en','reviewer_type':'human',
                'fact_correctness':1.,'answer_coverage':.5,'citation_support':1.,'insufficiency_handling':1.}
        summary=summarize_runs(rows,[review])
        self.assertEqual(summary['semantic_reviews']['human']['count'],1)
        self.assertNotIn('automatic',summary['semantic_reviews'])
        with self.assertRaises(ValueError):
            summarize_runs(rows,[review,review])
        with self.assertRaises(ValueError):
            summarize_runs(rows+[rows[0]])
