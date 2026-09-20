from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from foliorecall.agent import ask
from foliorecall.generation import Bailian, CostLedger, TaskStopped
from foliorecall.io import read_json
from foliorecall.qa_runtime import execute_plan, ledger_snapshot, transient, prepare_suite
from foliorecall.qa_storage import atomic_json, digest, output_lock, restart_gap
from foliorecall.qa_translation import translation_identity, validate_translation, cache_translation
from foliorecall.qa_evaluation import summarize_attempts, RATINGS
from tests.test_agent import FakeProvider, message, tool


def job(n=0, language='en', condition='visual-rag'):
    return {'key':dict(suite='mock',condition=condition,domain='hr',task_id=f'hr:{n}',language=language,repeat=0),
            'question':'What?' if language=='en' else '什么？',
            'task':{'answerability':'complete','reference_facts':['42']}}


def result(reason='answered', usage=1):
    return {'answer':{'claims':[],'insufficient_evidence':True,'unresolved':[]},'completed':reason=='answered',
            'termination_reason':reason,'timing':{'task_seconds':1},'estimated_cost_cny':.01,
            'api_calls':[], 'budget_state':{'requests_used':usage,'known_output_tokens':5,
                'unknown_output_tokens_reserved':0,'searches_used':1,'distinct_page_ids':['p1'],
                'active_seconds_used':1,'budget_snapshot_complete':True,'translation_debit':None}}


def compete(path,queue):
    try:
        with output_lock(path): queue.put('unexpected')
    except ValueError: queue.put('locked')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.config=read_json('configs/agent-bailian.json')
        self.config['ledger']=str(self.root/'cost.db')

    def execute(self,jobs,executor,**kwargs):
        return execute_plan(self.root/'run',{'version':1},jobs,lambda:executor,[self.config],**kwargs)

    def test_pause_resume_success_no_duplicates_and_completed_no_provider(self):
        jobs=[job(0),job(1)];calls=[]
        def run(j,b,a):calls.append((j['key'],a));return result()
        self.assertEqual(self.execute(jobs,run,stop_after=1)['pending'],1)
        self.assertEqual(self.execute(jobs,run,resume=True)['primary']['completed'],2)
        self.assertEqual(len(calls),2)
        def forbidden():raise AssertionError('provider initialized')
        execute_plan(self.root/'run',{'version':1},jobs,forbidden,[self.config],resume=True)
        with self.assertRaises(ValueError):
            execute_plan(self.root/'run',{'version':2},jobs,forbidden,[self.config],resume=True)

    def test_dry_run_has_no_output_lock_provider_or_ledger(self):
        with patch('foliorecall.qa_runtime.ledger_snapshot',side_effect=AssertionError):
            out=self.execute([job()],lambda *a:(_ for _ in ()).throw(AssertionError()),dry_run=True)
        self.assertEqual(out['planned'],1);self.assertEqual(list(self.root.iterdir()),[])

    def test_time_limit_does_not_start_next_item(self):
        called=[]
        out=self.execute([job()],lambda *a:called.append(1),max_seconds=.01,command_started=time.monotonic()-1)
        self.assertEqual(out['stopped'],'max_seconds');self.assertEqual(called,[])

    def test_one_explicit_recovery_failure_pauses_then_skip_no_third(self):
        calls=[];jobs=[job(0),job(1)]
        def run(j,b,a):
            calls.append((j['key']['task_id'],a,b))
            return result('api_http_503') if j['key']['task_id']=='hr:0' else result()
        self.assertEqual(self.execute(jobs,run)['stopped'],'api_http_503')
        self.assertEqual(len(calls),1)
        self.assertEqual(self.execute(jobs,run,resume=True)['recovery']['attempted'],1)
        self.assertEqual(calls[1][2],result()['budget_state'])
        out=self.execute(jobs,run,resume=True)
        self.assertEqual([(x[0],x[1]) for x in calls],[('hr:0',1),('hr:0',2),('hr:1',1)])
        self.assertEqual(out['primary']['completion_rate'],.5)

    def test_recovery_success_stays_out_of_primary(self):
        def run(j,b,a):return result('api_timeout' if a==1 else 'answered')
        self.execute([job()],run)
        out=self.execute([job()],run,resume=True,stop_after=1)
        self.assertEqual(out['primary']['completed'],0);self.assertEqual(out['recovery']['completed'],1)

    def test_retry_after_returns_without_sleep_or_calls(self):
        r=result('api_http_429');r['failure']={'retry_not_before':time.time()+300}
        self.execute([job()],lambda *a:r)
        out=self.execute([job()],lambda *a:(_ for _ in ()).throw(AssertionError()),resume=True)
        self.assertEqual(out['stopped'],'retry_after_unresolved')
        self.assertEqual(out['attempts_new'],0)

    def test_nonretryable_errors_and_exhausted_budget(self):
        for reason in ('invalid_answer_or_evidence','api_response_json_error','missing_usage','api_http_401','api_http_403','api_http_404','request_limit','phase_budget_exhausted'):
            self.assertFalse(transient(reason))
        self.execute([job()],lambda *a:result('api_timeout',usage=8))
        out=self.execute([job()],lambda *a:(_ for _ in ()).throw(AssertionError()),resume=True)
        self.assertEqual(out['recovery']['attempted'],0)

    def test_crash_unknown_budget_no_retry_and_ledger_preserved(self):
        def crash(j,b,a):
            CostLedger(self.config['ledger'],150).reserve(.5,dict(j['key'],execution_id=j['execution_id'],attempt=a,request_sequence=1))
            raise RuntimeError('synthetic interruption')
        with self.assertRaises(RuntimeError):self.execute([job()],crash)
        out=self.execute([job()],lambda *a:(_ for _ in ()).throw(AssertionError()),resume=True)
        self.assertEqual(out['stopped'],'incomplete_budget')
        self.assertEqual(out['unknown_reservation_cny'],.5)
        self.assertEqual(out['primary']['incomplete_budget'],1)

    def test_atomic_failure_keeps_previous_record_and_lock_refuses_second_process(self):
        target=self.root/'result.json';atomic_json(target,{'old':1})
        with patch('foliorecall.qa_storage.os.replace',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):atomic_json(target,{'new':2})
        self.assertEqual(read_json(target),{'old':1})
        ctx=multiprocessing.get_context('fork');queue=ctx.Queue()
        with output_lock(self.root/'run'):
            p=ctx.Process(target=compete,args=(self.root/'run',queue));p.start();p.join(5)
            self.assertFalse(p.is_alive());self.assertEqual(queue.get(timeout=1),'locked')

    def test_saved_terminal_attempt_survives_state_write_failure(self):
        real=atomic_json
        def failure(path,value):
            if Path(path).name=='state.json':raise OSError('state interrupted')
            real(path,value)
        with patch('foliorecall.qa_runtime.atomic_json',side_effect=failure):
            with self.assertRaises(OSError):self.execute([job()],lambda *a:result())
        out=self.execute([job()],lambda *a:(_ for _ in ()).throw(AssertionError()),resume=True)
        self.assertEqual(out['primary']['completed'],1)

    def test_repeat_selection_mismatch_rejected_before_loading(self):
        suite=read_json('configs/qa-repeat-suite.json')
        suite['selection']=str(self.root/'selection.json')
        atomic_json(suite['selection'],{'selection_before_system_answers':True,'tasks':[]})
        atomic_json(self.root/'suite.json',suite)
        with patch('foliorecall.qa_runtime.prepare_unit',side_effect=AssertionError),self.assertRaises(ValueError):
            prepare_suite(self.root/'suite.json')

    def test_malformed_budget_rejected_before_provider(self):
        broken=result('api_timeout');broken['budget_state']['requests_used']=-1
        self.execute([job()],lambda *a:broken)
        with self.assertRaises(ValueError):
            execute_plan(self.root/'run',{'version':1},[job()],lambda:(_ for _ in ()).throw(AssertionError()),[self.config],resume=True)

    def test_recovery_records_valid_wait_from_preserved_timestamps(self):
        with patch('foliorecall.qa_runtime.time.time',return_value=100.):
            self.execute([job()],lambda *a:result('api_timeout'))
        with patch('foliorecall.qa_runtime.time.time',return_value=125.):
            summary=self.execute([job()],lambda *a:result(),resume=True)
        folder=self.root/'run'/'attempts'/digest(job()['key'])
        first,second=(read_json(folder/f'attempt-{n:04d}.json') for n in (1,2))
        self.assertEqual(first['finished_at'],100.)
        self.assertEqual(second['started_at'],125.)
        self.assertEqual(second['gap_seconds'],25.)
        self.assertEqual(second['gap_status'],'valid')
        self.assertEqual(summary['recovery']['manual_restart_gap_seconds'],25.)
        self.assertEqual(summary['recovery']['manual_restart_gap_unknown_count'],0)

    def test_recovery_wall_clock_rollback_is_unknown_not_zero(self):
        with patch('foliorecall.qa_runtime.time.time',return_value=100.):
            self.execute([job()],lambda *a:result('api_timeout'))
        with patch('foliorecall.qa_runtime.time.time',return_value=90.):
            summary=self.execute([job()],lambda *a:result(),resume=True)
        folder=self.root/'run'/'attempts'/digest(job()['key'])
        first,second=(read_json(folder/f'attempt-{n:04d}.json') for n in (1,2))
        self.assertEqual(first['finished_at'],100.)
        self.assertEqual(second['started_at'],90.)
        self.assertEqual(second['gap_status'],'wall_clock_rollback')
        self.assertIsNone(second['gap_seconds'])
        self.assertIsNone(summary['recovery']['manual_restart_gap_seconds'])
        self.assertEqual(summary['recovery']['manual_restart_gap_unknown_count'],1)
        self.assertEqual(summary['primary']['completed'],0)
        self.assertEqual(summary['recovery']['completed'],1)
        self.assertEqual(summary['recovery']['total_actual_task_seconds'],1)


class BudgetAndProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.config=read_json('configs/agent-bailian.json');self.config['ledger']=str(self.root/'cost.db')
        self.pages=[dict(page_id='p1',doc_id='doc',page_number=1,score=.5,preview='unused')]
        self.patch=patch('foliorecall.agent.retrieve',return_value=(self.pages,'{}',{'request_seconds':.01}));self.patch.start();self.addCleanup(self.patch.stop)
        self.image=patch('foliorecall.agent.read_evidence',side_effect=lambda p,n:({'type':'text','text':'mock image'},dict(page_id=p['page_id'],seconds=.001)))
        self.image.start();self.addCleanup(self.image.stop)

    def ask(self,provider,mode='rag',question='What?',budget=None,translation=None,**options):
        return ask(question,mode,None,{},None,self.pages,dict(self.config,**options),provider,
                   budget_state=budget,translation=translation)

    def cached(self,**changes):
        return dict(identity=translation_identity('什么？',self.config),query='What?',requests=1,output_tokens=20,
                    seconds=2.,cost_cny=.003,api_calls=[dict(ledger_id='translation',usage={'prompt_tokens':10,'completion_tokens':20})],**changes)

    def test_unknown_outputs_and_all_cumulative_recovery_budgets(self):
        first=self.ask(FakeProvider([TaskStopped('api_timeout',{'ledger_id':'failed','reservation_cny':.5})]))
        b=first['budget_state'];self.assertEqual(b['requests_used'],1);self.assertEqual(b['unknown_output_tokens_reserved'],2048)
        provider=FakeProvider([message([{'text':'42','evidence_page_ids':['p1']}])])
        recovered=self.ask(provider,budget=b)
        self.assertEqual(recovered['budget_state']['requests_used'],2)
        self.assertEqual(recovered['budget_state']['searches_used'],2)
        self.assertEqual(recovered['budget_state']['distinct_page_ids'],['p1'])
        self.assertGreater(recovered['budget_state']['active_seconds_used'],b['active_seconds_used'])
        self.assertEqual(recovered['distinct_pages'],1)
        for field,value,reason in [('requests_used',8,'request_limit'),('searches_used',3,'search_limit'),
                                   ('active_seconds_used',180,'task_timeout'),('unknown_output_tokens_reserved',8000,'output_token_limit')]:
            limited=dict(b,**{field:value});p=FakeProvider([])
            self.assertEqual(self.ask(p,budget=limited)['termination_reason'],reason)
            self.assertEqual(p.inputs,[])
        # Old evidence must be read again in this attempt before it can be cited.
        p=FakeProvider([message(),message([{'text':'42','evidence_page_ids':['p1']}])])
        self.assertFalse(self.ask(p,mode='agent',budget=b)['completed'])

    def test_final_request_and_1024_tokens_reserved(self):
        p=FakeProvider([tool('search_pages',{'query':'x'}),message()])
        r=self.ask(p,mode='agent',max_requests=2,max_output_tokens=1100)
        self.assertEqual(p.inputs[0][2],76);self.assertIsNone(p.inputs[1][1])
        self.assertTrue(r['completed'])

    def test_cached_debits_once_recovery_and_live_fallback(self):
        cache=self.cached();p=FakeProvider([TaskStopped('api_timeout')])
        first=self.ask(p,question='什么？',translation=cache)
        b=first['budget_state'];self.assertEqual(b['requests_used'],2)
        self.assertEqual(b['known_output_tokens'],20);self.assertGreaterEqual(b['active_seconds_used'],2)
        second=self.ask(FakeProvider([message()]),question='什么？',translation=cache,budget=b)
        self.assertEqual(second['budget_state']['requests_used'],3)
        self.assertEqual(second['budget_state']['known_output_tokens'],25)
        self.assertLess(second['budget_state']['active_seconds_used']-b['active_seconds_used'],1)
        self.assertEqual(second['translation_cost_share_cny'],.001)
        missing=dict(cache);missing.pop('seconds')
        p=FakeProvider([{'content':'{"query":"live"}'},message()])
        r=self.ask(p,question='什么？',translation=missing)
        self.assertEqual(r['translation_status'],'live_missing_cache_budget');self.assertEqual(len(p.inputs),2)
        with self.assertRaises(ValueError):validate_translation(dict(cache,identity='wrong'),'什么？',self.config)
        exhausted=dict(cache,seconds=180)
        p=FakeProvider([]);self.assertEqual(self.ask(p,question='什么？',translation=exhausted)['termination_reason'],'task_timeout')
        self.assertEqual(p.inputs,[])
        for field,value in [('requests',8),('output_tokens',8000)]:
            over=deepcopy(cache);over[field]=value
            if field=='requests':over['api_calls']*=8;over['output_tokens']=160
            else:over['api_calls'][0]['usage']['completion_tokens']=8000
            self.assertEqual(self.ask(FakeProvider([]),question='什么？',translation=over)['termination_reason'],'translation_budget_exhausted')

    def test_legacy_ledger_migration_known_usage_survives_invalid_response(self):
        path=self.config['ledger']
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE calls (id TEXT PRIMARY KEY,cny REAL,status TEXT,usage TEXT,at REAL)')
            db.execute('INSERT INTO calls VALUES (?,?,?,?,?)',('old',.01,'estimated_from_usage','{}',1))
        ledger=CostLedger(path,150);self.assertEqual(ledger.total(),.01)
        provider=object.__new__(Bailian);provider.ledger=ledger;provider.key='mock';provider.base='https://unused';provider.context={'execution_id':'x'}
        import httpx
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):return httpx.Response(200,json={'usage':{'prompt_tokens':100,'completion_tokens':10},'choices':[]})
        with patch('httpx.AsyncClient',Client),self.assertRaises(TaskStopped) as caught:
            provider.complete([],None,100,10)
        self.assertEqual(str(caught.exception),'api_invalid_response')
        self.assertEqual(caught.exception.record['usage']['completion_tokens'],10)
        self.assertGreater(caught.exception.record['cost_cny'],0)
        self.assertEqual(len(ledger.records()),2);self.assertEqual(ledger.records()[1]['reservation_cny'],.789432)
        self.assertEqual(ledger_snapshot([self.config],'x')['unknown_reservation_cny'],0)

    def test_translation_generation_same_prompt_and_budget_metadata(self):
        class Provider(FakeProvider):
            def complete(self,*args):return super().complete(*args)|{'ledger_id':'cache1'}
        p=Provider([{'content':'{"query":"energy"}'}]);r=cache_translation('什么？',self.config,p)
        self.assertTrue(validate_translation(r,'什么？',self.config));self.assertEqual(r['requests'],1)
        for invalid in (None,'unknown',-1):
            broken=deepcopy(r);broken['api_calls'][0]['usage']['completion_tokens']=invalid
            self.assertFalse(validate_translation(broken,'什么？',self.config))
        with self.assertRaises(TaskStopped) as caught:
            cache_translation('什么？',self.config,Provider([{'content':'not JSON'}]))
        self.assertEqual(caught.exception.record['ledger_id'],'cache1')
        self.assertEqual(caught.exception.record['usage']['completion_tokens'],5)

    def test_http_failures_preserve_reservation_retry_after_and_missing_usage(self):
        import httpx
        provider=object.__new__(Bailian);provider.ledger=CostLedger(self.config['ledger'],150)
        provider.key='mock';provider.base='https://unused';provider.context={}
        for response,reason in [(httpx.Response(429,headers={'Retry-After':'120'}),'api_http_429'),
                                (httpx.Response(200,json={'choices':[]}),'missing_usage'),
                                (httpx.ConnectError('mock transport'),'api_transport_error')]:
            class Client:
                def __init__(self,**kwargs):pass
                async def __aenter__(self):return self
                async def __aexit__(self,*args):pass
                async def post(self,*args,**kwargs):
                    if isinstance(response,Exception):raise response
                    return response
            with patch('httpx.AsyncClient',Client),self.assertRaises(TaskStopped) as caught:
                provider.complete([],None,123,10)
            self.assertEqual(str(caught.exception),reason)
            self.assertEqual(caught.exception.record['max_tokens'],123)
            self.assertEqual(caught.exception.record['cost_status'],'reserved_unknown_charge')
            if reason=='api_http_429':self.assertGreater(caught.exception.record['retry_not_before'],time.time()+110)
        self.assertEqual(len(provider.ledger.records()),3)

    def test_recovery_distinct_page_limit_and_old_pages_can_be_read_again(self):
        b=result()['budget_state'];b['distinct_page_ids']=[f'p{i}' for i in range(1,7)]
        self.pages.append(dict(page_id='p7',doc_id='doc',page_number=7,preview='unused'))
        p=FakeProvider([tool('read_page',{'page_id':'p7'}),message()])
        r=self.ask(p,mode='agent',budget=b)
        self.assertEqual(len(r['budget_state']['distinct_page_ids']),6)
        self.assertEqual(r['distinct_pages'],0);self.assertIsNone(p.inputs[-1][1])
        p=FakeProvider([tool('read_page',{'page_id':'p1'}),message(),message([{'text':'42','evidence_page_ids':['p1']}])])
        r=self.ask(p,mode='agent',budget=b)
        self.assertTrue(r['completed']);self.assertEqual(r['distinct_pages'],1)
        self.assertEqual(len(r['budget_state']['distinct_page_ids']),6)

    def test_three_conditions_two_languages_mock_suite(self):
        jobs=[job(language=lang,condition=c) for c in ('visual-rag','hybrid-rag','hybrid-agent') for lang in ('en','zh')]
        def execute(j,b,a):
            responses=[]
            if j['key']['language']=='zh':responses.append({'content':'{"query":"What?"}'})
            agent=j['key']['condition']=='hybrid-agent'
            if agent:responses.extend([tool('search_pages',{'query':'What?'}),tool('read_page',{'page_id':'p1'}),message()])
            responses.append(message([{'text':'42','evidence_page_ids':['p1']}]))
            return self.ask(FakeProvider(responses),'agent' if agent else 'rag',j['question'],budget=b)
        summary=execute_plan(self.root/'suite',{'mock':True},jobs,lambda:execute,[self.config])
        self.assertEqual(summary['primary']['completed'],6)
        self.assertEqual(len(summary['primary_language_pairs']),3)
        self.assertEqual(len(summary['by_condition_domain_language']),6)
        self.assertFalse(Path(self.config['ledger']).exists())


class ScoringTests(unittest.TestCase):
    def row(self,attempt=1):
        r=result('api_timeout' if attempt==1 else 'answered')
        return dict(key=job()['key'],task=job()['task'],attempt=attempt,state='finished',result=r,answer_digest=digest(r['answer']))

    def review(self,row):
        return dict(key=row['key'],attempt=row['attempt'],answer_digest=row['answer_digest'],reviewer_type='assistant',
                    fact_correctness=None,answer_coverage=0,citation_support=None,insufficiency_handling=0,
                    not_applicable={'fact_correctness':'no claims','citation_support':'no claims'})

    def test_first_primary_recovery_separate_nulls_missing_and_empty_answers(self):
        rows=[self.row(1),self.row(2)];review=self.review(rows[1])
        out=summarize_attempts(rows,2,[review])
        self.assertEqual(out['primary']['completed'],0);self.assertEqual(out['recovery']['completed'],1)
        self.assertEqual(out['pending'],1)
        metrics=next(iter(out['semantic_reviews'].values()))['metrics']
        self.assertIsNone(metrics['fact_correctness']['mean'])
        for bad in (dict(review,answer_digest='wrong'),dict(review,attempt=3),dict(review,answer_coverage=1),
                    dict(review,key=dict(review['key'],language='zh')),dict(review,key=dict(review['key'],condition='hybrid-rag')),
                    dict(review,key=dict(review['key'],repeat=1))):
            with self.assertRaises(ValueError):summarize_attempts(rows,2,[bad])
        with self.assertRaises(ValueError):summarize_attempts(rows,2,[review,review])
        self.assertEqual(summarize_attempts(rows,2)['semantic_quality_status'],'pending_review')

    def test_unanswerable_metrics_not_automatic_perfect_and_ledger_dedup(self):
        row=self.row();row['task']={'answerability':'insufficient','reference_facts':[]}
        review=self.review(row);review.update(answer_coverage=None,insufficiency_handling=1)
        review['not_applicable']['answer_coverage']='no required facts'
        self.assertEqual(summarize_attempts([row],1,[review])['pending'],0)
        with self.assertRaises(ValueError):summarize_attempts([row],1,[dict(review,fact_correctness=1)])
        row['result']['api_calls']=[{'ledger_id':'same','cost_cny':.1}]*2
        out=summarize_attempts([row],1,ledger_records=[{'id':'same','status':'estimated_from_usage','cny':.1},
            {'id':'orphan','status':'reserved_unknown_charge','cny':.5}])
        self.assertEqual(out['all_attempts_known_cost_cny'],.1);self.assertEqual(out['unknown_reservation_cny'],.5)

    def test_tampered_answer_unknown_run_and_partial_coverage(self):
        row=self.row();row['task']['answerability']='partial'
        review=self.review(row)
        with self.assertRaises(ValueError):
            summarize_attempts([row],1,planned_keys=[job(2)['key']])
        bad=dict(review,answer_coverage=None,not_applicable=dict(review['not_applicable'],answer_coverage='partial'))
        with self.assertRaises(ValueError):summarize_attempts([row],1,[bad])
        row['result']['answer']['unresolved']=['changed after run']
        with self.assertRaises(ValueError):summarize_attempts([row],1,[review])

    def test_legacy_gaps_recomputed_without_mutating_attempts(self):
        first,second=self.row(1),self.row(2)
        first['finished_at']=100.
        second.update(started_at=125.,gap_seconds=-99.)
        before=deepcopy([first,second])
        out=summarize_attempts([first,second],1)
        self.assertEqual(out['recovery']['manual_restart_gap_seconds'],25.)
        self.assertEqual(out['recovery_intervals'][0]['gap_status'],'valid')
        self.assertEqual([first,second],before)

        second.update(started_at=90.,gap_seconds=-10.)
        out=summarize_attempts([first,second],1)
        self.assertIsNone(out['recovery']['manual_restart_gap_seconds'])
        self.assertEqual(out['recovery_intervals'][0]['gap_status'],'wall_clock_rollback')
        self.assertEqual(second['gap_seconds'],-10.)

    def test_partial_restart_total_separates_known_and_missing(self):
        rows=[]
        for n,started in enumerate((125.,None)):
            first,second=self.row(1),self.row(2)
            first['key']=second['key']=job(n)['key']
            first['finished_at']=100.
            second.update(started_at=started,gap_seconds=0.)
            rows.extend([first,second])
        out=summarize_attempts(rows,2)
        self.assertIsNone(out['recovery']['manual_restart_gap_seconds'])
        self.assertEqual(out['recovery']['manual_restart_gap_known_seconds'],25.)
        self.assertEqual(out['recovery']['manual_restart_gap_unknown_count'],1)
        self.assertEqual(out['recovery']['manual_restart_gap_status_counts'],
                         {'missing_timestamp':1,'valid':1})
        self.assertEqual(out['primary']['completed'],0)
        self.assertEqual(out['recovery']['completed'],2)

    def test_missing_or_nonfinite_restart_timestamps_are_unknown(self):
        for second,first,status in (
                ({'attempt':2},{'finished_at':100.},'missing_timestamp'),
                ({'attempt':2,'started_at':125.},{},'missing_timestamp'),
                ({'attempt':2,'started_at':float('nan')},{'finished_at':100.},'invalid_timestamp'),
                ({'attempt':2,'started_at':125.},{'finished_at':float('inf')},'invalid_timestamp'),
                ({'attempt':2,'started_at':True},{'finished_at':100.},'invalid_timestamp'),
                ({'attempt':2,'started_at':1e308},{'finished_at':-1e308},'nonfinite_interval')):
            with self.subTest(status=status,second=second,first=first):
                self.assertEqual(restart_gap(second,first),{'gap_seconds':None,'gap_status':status})
