import json
import unittest
from unittest.mock import patch

from foliorecall.agent import ask, validate_answer, AnswerValidationError
from foliorecall.generation import Bailian, CostLedger, TaskStopped, http_error_diagnostic, retry_after_metadata
from foliorecall.qa_runtime import suite_jobs, wait_for_cooldown, execute_plan
from tests import test_agent as fixtures
from tests import test_qa_runtime as runtime


class CitationTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    run_ask = fixtures.AgentTests.run_ask

    def allowed(self, provider):
        return json.loads(provider.inputs[-1][0][-1]['content'].split('allowed_evidence_page_ids=')[-1])

    def test_rag_physical_number_is_not_permission_and_trace_matches(self):
        self.pages[0].update(page_id='486',page_number=62)
        p=fixtures.FakeProvider([fixtures.message([{'text':'fact','evidence_page_ids':['486']}])])
        r=self.run_ask(p,'rag')
        self.assertTrue(r['completed'])
        self.assertEqual(self.allowed(p),['486'])
        self.assertEqual(r['trace'][-1]['allowed_evidence_page_ids'],['486'])
        for ids in (['62'],[486],[],['PLACEHOLDER_READ_PAGE_ID']):
            with self.assertRaises(AnswerValidationError):
                validate_answer({'claims':[{'text':'fact','evidence_page_ids':ids}],
                                 'insufficient_evidence':False,'unresolved':[]},{'486'})
        # A genuine page ID that looks like a physical number remains valid.
        self.pages[0].update(page_id='62')
        p=fixtures.FakeProvider([fixtures.message([{'text':'fact','evidence_page_ids':['62']}])])
        self.assertTrue(self.run_ask(p,'rag')['completed'])

    def test_search_unread_and_inherited_pages_are_excluded(self):
        self.pages.append(dict(self.pages[0],page_id='p2'))
        previous={'requests_used':0,'known_output_tokens':0,'unknown_output_tokens_reserved':0,
                  'searches_used':0,'distinct_page_ids':['p2'],'active_seconds_used':0}
        p=fixtures.FakeProvider([fixtures.tool('search_pages',{'query':'energy'}),
            fixtures.tool('read_page',{'page_id':'p1'}),{'content':'done'},
            fixtures.message([{'text':'fact','evidence_page_ids':['p2']}])])
        r=ask('What?','agent',None,{'device':'cpu'},None,self.pages,self.config,p,budget_state=previous)
        self.assertEqual(self.allowed(p),['p1'])
        self.assertEqual(r['failure']['output_diagnostic']['code'],'unread_citation')
        self.assertEqual(r['budget_state']['distinct_page_ids'],['p1','p2'])
        self.assertEqual(len(p.inputs),4)

    def test_duplicate_read_and_zero_read(self):
        p=fixtures.FakeProvider([fixtures.tool('read_page',{'page_id':'p1'}),
            fixtures.tool('read_page',{'page_id':'p1'}),{'content':'done'},fixtures.message()])
        self.assertTrue(self.run_ask(p)['completed'])
        self.assertEqual(self.allowed(p),['p1'])
        p=fixtures.FakeProvider([fixtures.message()])
        self.assertTrue(self.run_ask(p,max_requests=1)['completed'])
        self.assertEqual(self.allowed(p),[])

    def test_failed_read_never_grants_permission(self):
        p=fixtures.FakeProvider([fixtures.tool('read_page',{'page_id':'unknown'}),{'content':'done'},fixtures.message()])
        self.run_ask(p)
        self.assertEqual(self.allowed(p),[])


class HttpDiagnosticsTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp

    def test_retry_after_seconds_and_server_date(self):
        for value in ('0', '120'):
            row = retry_after_metadata({'Retry-After': value})
            self.assertEqual(row['retry_after_seconds'], int(value))
        headers = {'Retry-After': 'Mon, 21 Sep 2026 10:02:00 GMT',
                   'Date': 'Mon, 21 Sep 2026 10:00:00 GMT'}
        with patch('foliorecall.generation.time.time', return_value=-1000000):
            self.assertEqual(retry_after_metadata(headers)['retry_after_seconds'], 120)
        headers.pop('Date')
        self.assertEqual(retry_after_metadata(headers)['retry_after_status'], 'unresolved')
        for value in ('-1', '1.5', 'nan', '1e3', 'broken'):
            self.assertEqual(retry_after_metadata({'Retry-After': value})['retry_after_status'], 'invalid')
        self.assertEqual(retry_after_metadata({})['retry_after_status'], 'absent')

    def test_error_whitelist_redaction_and_unknown_reservation(self):
        import httpx
        provider=object.__new__(Bailian)
        provider.key='mock-secret';provider.base='https://unused';provider.context={}
        provider.ledger=CostLedger(self.config['ledger'],150)
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):
                return httpx.Response(429,headers={'Retry-After':'120','private':'hidden-header'},json={
                    'request_id':'req','error':{'code':'Throttling.AllocationQuota','type':'rate_limit',
                    'message':'mock-secret <think>hidden</think> data:image/png;base64,YWJjZA==; '+('visible '*500),
                    'reasoning_content':'private reasoning'},'request':{'Authorization':'secret'}})
        with patch('httpx.AsyncClient',Client),self.assertRaises(TaskStopped) as caught:
            provider.complete([],None,100,1)
        d=caught.exception.record['http_diagnostic']
        self.assertEqual(str(caught.exception),'api_http_429')
        self.assertEqual(d['code'],'Throttling.AllocationQuota')
        self.assertEqual(d['request_id'],'req')
        self.assertEqual(len(d['message']),2048)
        self.assertTrue(d['message_truncated'])
        for text in ('mock-secret','hidden','YWJjZA','private reasoning','Authorization'):
            self.assertNotIn(text,json.dumps(d))
        self.assertIn('retry_not_before',caught.exception.record)
        self.assertEqual(provider.ledger.records()[0]['status'],'reserved_unknown_charge')

    def test_non_json_body_and_nonstring_fields_are_not_serialized(self):
        import httpx
        d=http_error_diagnostic(httpx.Response(503,text='private raw body'), 'key')
        self.assertEqual(set(d),{'http_status','content_type'})
        d=http_error_diagnostic(httpx.Response(429,json={'error':{'message':{'secret':'hidden'},'code':['x']}}),'key')
        self.assertEqual(set(d),{'http_status','content_type'})


class Clock:
    def __init__(self):self.mono=0.;self.wall=1000.
    def sleep(self,s):self.mono+=s;self.wall+=s


class PolicyTests(unittest.TestCase):
    setUp = runtime.RuntimeTests.setUp

    def test_permutation_requires_complete_unique_keys_and_keeps_unit(self):
        jobs=[runtime.job(str(i))|{'unit':i%2} for i in range(4)]
        units=[{'jobs':jobs[:2]},{'jobs':jobs[2:]}]
        order=[j['key'] for j in reversed(jobs)]
        out=suite_jobs(units,{'job_order':order,'inter_task_cooldown_seconds':65})
        self.assertEqual(out,list(reversed(jobs)))
        self.assertEqual(suite_jobs(units,{}),jobs)
        for bad in (order[:-1],order+[order[0]],order[:-1]+[{'other':'x'}],None):
            with self.assertRaises(ValueError):suite_jobs(units,{'job_order':bad})
        for bad in (-1,True,float('nan')):
            with self.assertRaises(ValueError):suite_jobs(units,{'inter_task_cooldown_seconds':bad})

    def test_cooldown_resume_does_not_repeat_completed_jobs(self):
        jobs=[runtime.job(str(i)) for i in range(5)]
        identity={'suite':{'job_order':[j['key'] for j in jobs],'inter_task_cooldown_seconds':65}}
        clock=Clock();calls=[]
        def execute(j,b,a):calls.append(j['key']);return runtime.result()
        with patch('foliorecall.qa_runtime.time.time',side_effect=lambda:clock.wall), \
             patch('foliorecall.qa_runtime.time.monotonic',side_effect=lambda:clock.mono), \
             patch('foliorecall.qa_runtime.time.sleep',side_effect=clock.sleep):
            r=execute_plan(self.root/'run',identity,jobs,lambda:execute,[self.config],stop_after=4)
            self.assertEqual((r['attempts_new'],len(calls),r['cooldown_seconds']),(4,4,195))
            r=execute_plan(self.root/'run',identity,jobs,lambda:execute,[self.config],resume=True)
            self.assertEqual((r['attempts_new'],len(calls),r['cooldown_seconds']),(1,5,65))
            with self.assertRaises(ValueError):
                execute_plan(self.root/'run',{'suite':dict(identity['suite'],inter_task_cooldown_seconds=0)},
                             jobs,lambda:execute,[self.config],resume=True)
        self.assertEqual(len({json.dumps(k,sort_keys=True) for k in calls}),5)

    def test_short_segment_pauses_but_wall_rollback_does_not(self):
        jobs=[runtime.job('1'),runtime.job('2')]
        identity={'suite':{'inter_task_cooldown_seconds':65}}
        clock=Clock();calls=[]
        def execute(j,b,a):calls.append(j);return runtime.result()
        with patch('foliorecall.qa_runtime.time.time',side_effect=lambda:clock.wall), \
             patch('foliorecall.qa_runtime.time.monotonic',side_effect=lambda:clock.mono), \
             patch('foliorecall.qa_runtime.time.sleep',side_effect=clock.sleep):
            r=execute_plan(self.root/'run',identity,jobs,lambda:execute,[self.config],max_seconds=60)
            self.assertEqual(r['stopped'],'max_seconds');self.assertEqual(len(calls),1)
            clock.wall-=10
            r=execute_plan(self.root/'run',identity,jobs,lambda:execute,[self.config],resume=True)
            self.assertIsNone(r['stopped']);self.assertEqual(len(calls),2)
            self.assertEqual(r['cooldown_seconds'],65)

    def test_dry_run_does_not_wait_and_order_mismatch_fails(self):
        jobs=[runtime.job('1'),runtime.job('2')]
        identity={'suite':{'job_order':[j['key'] for j in jobs],'inter_task_cooldown_seconds':65}}
        with patch('foliorecall.qa_runtime.wait_for_cooldown',side_effect=AssertionError):
            self.assertEqual(execute_plan(self.root/'none',identity,jobs,None,[],dry_run=True)['planned'],2)
        with self.assertRaises(ValueError):
            execute_plan(self.root/'none',identity,list(reversed(jobs)),None,[],dry_run=True)
        self.assertFalse((self.root/'none').exists())

    def test_recovery_cooldown_still_inherits_budget_and_retry_after(self):
        jobs=[runtime.job('1')];identity={'suite':{'inter_task_cooldown_seconds':65}}
        clock=Clock();calls=[]
        def execute(j,b,a):
            calls.append(b)
            r=runtime.result('api_http_429' if a==1 else 'answered')
            if a==1:r['failure']={'retry_after_seconds':120,'retry_after_status':'resolved'}
            return r
        with patch('foliorecall.qa_runtime.time.time',side_effect=lambda:clock.wall), \
             patch('foliorecall.qa_runtime.time.monotonic',side_effect=lambda:clock.mono), \
             patch('foliorecall.qa_runtime.time.sleep',side_effect=clock.sleep):
            execute_plan(self.root/'run',identity,jobs,lambda:execute,[self.config])
            r=execute_plan(self.root/'run',identity,jobs,lambda:execute,[self.config],resume=True)
            self.assertEqual(r['cooldown_seconds'],120)
            self.assertEqual(calls[-1],runtime.result()['budget_state'])
            self.assertEqual((r['primary']['completed'],r['recovery']['completed']),(0,1))

    def test_wall_clock_jump_during_wait_does_not_change_duration(self):
        clock=Clock()
        def sleep(s):clock.sleep(s);clock.wall-=5
        with patch('foliorecall.qa_runtime.time.time',side_effect=lambda:clock.wall), \
             patch('foliorecall.qa_runtime.time.monotonic',side_effect=lambda:clock.mono), \
             patch('foliorecall.qa_runtime.time.sleep',side_effect=sleep):
            waited,reason=wait_for_cooldown(clock.mono,65,clock.mono,None)
        self.assertIsNone(reason)
        self.assertEqual(waited,65)

    def test_exhausted_retry_cannot_bypass_service_wait_for_next_task(self):
        jobs = [runtime.job('1'), runtime.job('2')]
        identity = {'suite': {'inter_task_cooldown_seconds': 65}}
        clock = Clock(); calls = []
        def execute(j, b, a):
            calls.append(j['key']['task_id'])
            r = runtime.result('api_http_429', usage=8) if len(calls) == 1 else runtime.result()
            if len(calls) == 1:
                r['api_calls'] = [{'retry_after_seconds': 120, 'retry_after_status': 'resolved'}]
            return r
        with patch('foliorecall.qa_runtime.time.time', side_effect=lambda: clock.wall), \
             patch('foliorecall.qa_runtime.time.monotonic', side_effect=lambda: clock.mono), \
             patch('foliorecall.qa_runtime.time.sleep', side_effect=clock.sleep):
            execute_plan(self.root/'run', identity, jobs, lambda: execute, [self.config])
            clock.wall += 100000
            r = execute_plan(self.root/'run', identity, jobs, lambda: execute, [self.config], resume=True, max_seconds=100)
            self.assertEqual((r['stopped'], len(calls)), ('max_seconds', 1))
            clock.wall -= 200000
            r = execute_plan(self.root/'run', identity, jobs, lambda: execute, [self.config], resume=True)
            self.assertEqual(r['cooldown_seconds'], 120)
            self.assertEqual(calls, ['hr:1', 'hr:2'])

    def test_unresolved_service_date_blocks_following_task(self):
        jobs = [runtime.job('1'), runtime.job('2')]
        r = runtime.result('api_http_429', usage=8)
        r['failure'] = {'retry_after_status': 'unresolved', 'retry_not_before': 1}
        execute_plan(self.root/'run', {}, jobs, lambda: lambda *a: r, [self.config])
        with patch('foliorecall.qa_runtime.time.sleep', side_effect=AssertionError):
            out = execute_plan(self.root/'run', {}, jobs, lambda: (_ for _ in ()).throw(AssertionError()),
                               [self.config], resume=True)
        self.assertEqual(out['stopped'], 'retry_after_unresolved')
        self.assertEqual(out['attempts_new'], 0)


if __name__=='__main__':unittest.main()
