from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_evaluation import RATINGS
from scripts.qa_continuation import freeze, derive, validate_source
from scripts.run_qa_development_segment import initial_continuation_wait, segment_boundary
from scripts.summarize_qa_continuation import summarize
from tests.test_qa_runtime import job, result


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'original';self.target=self.root/'continuation'
        self.jobs=[dict(job(i),unit=0) for i in range(180)]
        self.config=dict(read_json('configs/agent-bailian.json'),ledger=str(self.root/'absent.sqlite3'))
        self.identity={'suite':{'job_order':[j['key'] for j in self.jobs],'inter_task_cooldown_seconds':65},
                       'units':[{'agent':self.config}],'code':'fixed'}
        self.manifest={'identity':self.identity,'identity_digest':digest(self.identity),
                       'execution_id':'original-id','job_keys':[j['key'] for j in self.jobs]}
        atomic_json(self.source/'manifest.json',self.manifest)
        self.reviews=[]
        for i in range(105):
            a=self.attempt(i, i==104)
            atomic_json(self.source/'attempts'/digest(a['key'])/'attempt-0001.json',a)
            if i!=104:self.reviews.append(self.review(a))
        self.audit=self.root/'audit.json'
        atomic_json(self.audit,{'status':'blocked_unknown_task_budget','interrupted_key':self.jobs[104]['key'],
                               'started':105,'not_started':75})
        self.d=freeze(self.source,self.audit,self.target,'User explicitly retains this one unknown observation.')

    def attempt(self,i,running=False):
        a={'key':self.jobs[i]['key'],'task':self.jobs[i]['task'],'attempt':1,
           'state':'running' if running else 'finished','started_at':1.}
        if not running:
            r=result();r['api_calls']=[]
            a.update(result=r,answer_digest=digest(r['answer']),finished_at=2.)
        return a

    def review(self,a):
        return dict(key=a['key'],attempt=1,answer_digest=a['answer_digest'],reviewer_type='assistant',
                    human='not_checked',primary_error='none',primary_error_reason='Synthetic reference',
                    secondary_factors=[],**{m:0. for m in RATINGS})

    def test_exact_pending_new_identity_and_source_unchanged(self):
        before={str(p):p.read_bytes() for p in self.source.rglob('*.json')}
        identity,jobs,old=derive(self.d,self.identity,self.jobs,self.target)
        self.assertEqual([j['key'] for j in jobs],[j['key'] for j in self.jobs[105:]])
        self.assertNotEqual(digest(identity),digest(self.identity))
        self.assertEqual(identity['units'],self.identity['units'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.source.rglob('*.json')})
        with self.assertRaisesRegex(ValueError,'未知在途'):
            segment_boundary(self.source,self.identity,self.jobs,[{'agent':self.config}],True)

    def test_changed_source_or_duplicate_or_started_key_refused(self):
        for keys in (self.d['pending_keys']+[self.d['pending_keys'][0]],
                     self.d['pending_keys'][1:]+[self.jobs[104]['key']]):
            d=deepcopy(self.d);d['pending_keys']=keys
            with self.assertRaisesRegex(ValueError,'Pending keys'):validate_source(d)
        path=self.source/'attempts'/digest(self.jobs[0]['key'])/'attempt-0001.json'
        a=read_json(path);a['finished_at']=3.;atomic_json(path,a)
        with self.assertRaisesRegex(ValueError,'attempts changed'):validate_source(self.d)

    def test_changed_current_identity_and_wrong_target_refused(self):
        changed=deepcopy(self.identity);changed['code']='new'
        with self.assertRaisesRegex(ValueError,'identity differs'):derive(self.d,changed,self.jobs,self.target)
        with self.assertRaisesRegex(ValueError,'Wrong continuation'):derive(self.d,self.identity,self.jobs,self.root/'other')

    def test_cooldown_first_empty_resume_and_normal_resume(self):
        with patch('scripts.run_qa_development_segment.latest_service_wait',return_value=(90.,None)), \
             patch('scripts.run_qa_development_segment.wait_for_cooldown',return_value=(90.,None)) as wait:
            self.assertEqual(initial_continuation_wait([],[],0,1000)['seconds'],90.)
            self.assertEqual(wait.call_args.args[1],90.)
            initial_continuation_wait([],[{'saved':True}],0,1000)
            self.assertEqual(wait.call_count,1)
        with patch('scripts.run_qa_development_segment.latest_service_wait',return_value=(0.,'retry_after_unresolved')), \
             patch('scripts.run_qa_development_segment.wait_for_cooldown') as wait:
            self.assertEqual(initial_continuation_wait([],[],0,1000)['stopped'],'retry_after_unresolved')
            wait.assert_not_called()

    def test_merge_retains_unknown_and_rejects_duplicate_or_new_crash(self):
        identity,_,_=derive(self.d,self.identity,self.jobs,self.target)
        atomic_json(self.target/'manifest.json',dict(identity=identity,identity_digest=digest(identity),
                    execution_id='new-id',job_keys=self.d['pending_keys']))
        for i in range(105,180):
            a=self.attempt(i)
            atomic_json(self.target/'attempts'/digest(a['key'])/'attempt-0001.json',a)
            self.reviews.append(self.review(a))
        r=summarize(self.d,self.target,self.reviews)
        self.assertEqual(r['completion']['first_terminal'],179)
        self.assertEqual(r['completion']['status'],'reviewed_with_declared_missing')
        self.assertFalse(r['all_attempts_reviewed'])
        self.assertEqual(r['summary']['primary']['incomplete_budget'],1)
        bad=self.attempt(179,True)
        atomic_json(self.target/'attempts'/digest(bad['key'])/'attempt-0001.json',bad)
        with self.assertRaisesRegex(ValueError,'未知在途'):
            segment_boundary(self.target,identity,self.jobs[105:],[{'agent':self.config}],True)
        r=summarize(self.d,self.target,self.reviews[:-1])
        self.assertEqual(r['completion']['status'],'partial')
        duplicate=self.attempt(0)
        atomic_json(self.target/'attempts'/digest(duplicate['key'])/'attempt-0001.json',duplicate)
        with self.assertRaisesRegex(ValueError,'outside allowed'):summarize(self.d,self.target,self.reviews[:-1])

    def test_crash_costs_remain_in_combined_summary_and_other_runs_not_doubled(self):
        from foliorecall.generation import CostLedger
        identity,_,_=derive(self.d,self.identity,self.jobs,self.target)
        atomic_json(self.target/'manifest.json',dict(identity=identity,identity_digest=digest(identity),
                    execution_id='new-id',job_keys=self.d['pending_keys']))
        ledger=CostLedger(self.config['ledger'],62.379233)
        ctx=dict(self.jobs[104]['key'],execution_id='original-id',attempt=1)
        first=ledger.reserve(.8,dict(ctx,request_sequence=1))
        amount=ledger.settle(first,{'prompt_tokens':1000,'completion_tokens':10})
        ledger.reserve(.8,dict(ctx,request_sequence=2))
        ledger.reserve(.2,{'execution_id':'unrelated'})
        r=summarize(self.d,self.target,self.reviews)
        self.assertAlmostEqual(r['summary']['all_attempts_known_cost_cny'],amount)
        self.assertAlmostEqual(r['summary']['unknown_reservation_cny'],.8)
        self.assertAlmostEqual(r['ledger']['phase_total_including_other_runs'][str(Path(self.config['ledger']).resolve())],1+amount)


if __name__=='__main__':unittest.main()
