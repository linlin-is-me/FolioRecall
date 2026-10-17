from pathlib import Path
import tempfile
import unittest

from foliorecall.io import read_json, write_json, write_rows
from foliorecall.search import encoding_identity
from scripts.run_teacher_transfer import prepare, index_command, aggregate, numerical_gate, DOMAINS, METRICS


class TransferPreparationTests(unittest.TestCase):
    def fixture(self, repo):
        teacher={'model_id':'teacher','revision':'fixed','dimension':2048,'adapter':None,'device':'cuda'}
        history=repo/'outputs/stage3/targets-3000'
        write_json(history/'manifest.json',{'count':3000,'encoding_config':teacher,
            'encoding_identity':encoding_identity(teacher)})
        write_rows(history/'queries.jsonl',[{'query_id':str(i),'query':f'query {i}'} for i in range(3000)])
        base=repo/'outputs/extensions/B/heldout-seeds-v1/training/ordinary-42'
        write_json(base/'result.json',{'complete':True,'completed_steps':64})
        write_json(base/'checkpoint-64/encoding.json',dict(teacher,adapter=str(base/'checkpoint-64')))
        write_json(repo/'configs/student-ml-cuda.json',{'kind':'nanovdr','model_id':'public-student',
            'revision':'fixed','tokenizer_revision':'fixed','target_encoding_identity':encoding_identity(teacher),
            'device':'cuda','dtype':'bfloat16','model_path':'old-package'})

    def test_preparation_separates_targets_but_keeps_same_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);self.fixture(repo)
            root=repo/'experiment';plan=prepare(root,repo)
            ordinary=read_json(root/'configs/train-O.json')
            candidate=read_json(root/'configs/train-L.json')
            self.assertNotEqual(ordinary['target_encoding_identity'],candidate['target_encoding_identity'])
            a=dict(ordinary);b=dict(candidate)
            a.pop('target_encoding_identity');b.pop('target_encoding_identity')
            self.assertEqual(a,b)
            self.assertNotIn('model_path',a)
            self.assertEqual(sum(d['queries'] for d in plan['datasets'].values()),470)
            self.assertEqual(plan['steps'],282)
            self.assertEqual(prepare(root,repo),plan)
            self.assertIn('ordinary-42',plan['datasets']['biomedical']['indexes']['lora'])

    def test_incomplete_teacher_is_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);self.fixture(repo)
            write_json(repo/'outputs/extensions/B/heldout-seeds-v1/training/ordinary-42/result.json',
                       {'complete':False,'completed_steps':60})
            with self.assertRaisesRegex(ValueError,'64步'):
                prepare(repo/'experiment',repo)
            self.assertFalse((repo/'experiment/plan.json').exists())

    def test_index_resume_detects_work_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'index'
            self.assertNotIn('--resume',index_command('config','pages',output))
            write_json(Path(str(output)+'.work')/'manifest.json',{'completed':64})
            self.assertIn('--resume',index_command('config','pages',output))


class TransferReportTests(unittest.TestCase):
    def values(self, score):
        return {'domain_equal':{'nDCG@10':score},
                'domains':{d:{'nDCG@10':score} for d in DOMAINS}}

    def test_beating_only_distilled_control_does_not_allow_migration(self):
        result=numerical_gate({'P':self.values(.7),'O':self.values(.5),'L':self.values(.6)})
        self.assertFalse(result['numerically_eligible'])

    def test_one_domain_does_not_suffice_even_with_higher_macro(self):
        values={'P':self.values(.7),'O':self.values(.6),'L':self.values(.72)}
        values['L']['domains']['biomedical']['nDCG@10']=.9
        for domain in ('economics','esg_human'):
            values['L']['domains'][domain]['nDCG@10']=.63
        result=numerical_gate(values)
        self.assertTrue(result['macro_beats_both'])
        self.assertFalse(result['numerically_eligible'])

    def test_aggregation_preserves_domain_and_query_weighting(self):
        results={d:{'metrics':{k:v for k in METRICS},'results':[
            {'query_id':str(i),'query':'q',**{k:v for k in METRICS}} for i in range(n)]}
            for d,v,n in [('a',.5,1),('b',1.,3)]}
        actual=aggregate(results,['a','b'])
        self.assertAlmostEqual(actual['domain_equal']['nDCG@10'],.75)
        self.assertAlmostEqual(actual['query_weighted']['nDCG@10'],.875)
        self.assertEqual(len({r['query_id'] for r in actual['results']}),4)


if __name__=='__main__': unittest.main()
