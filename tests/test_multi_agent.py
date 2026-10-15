import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from foliorecall.generation import TaskStopped, validate_agent_config
from foliorecall.io import read_json
from foliorecall.multi_agent import ask_multi
from tests.test_agent import tool, message


class ScriptedProvider:
    def __init__(self, script=None, barrier=None):
        self.context = {}
        self.inputs, self.counts = [], {}
        self.lock = threading.Lock()
        self.active = {'now':0,'peak':0}
        self.script, self.barrier = script, barrier

    def complete(self, messages, tools, max_tokens, timeout):
        context = dict(self.context)
        key = context['node']
        with self.lock:
            self.inputs.append({'context':context,'messages':copy.deepcopy(messages),'max_tokens':max_tokens})
            self.counts[key] = self.counts.get(key,0)+1
            count = self.counts[key]
            self.active['now'] += 1
            self.active['peak'] = max(self.active['now'],self.active['peak'])
        try:
            if self.barrier and context['agent_id'].startswith('worker') and key.endswith('-1'):
                self.barrier.wait(5)
            value = self.script(context,messages,count) if self.script else None
            if value is None:
                value = self.default(context)
            if isinstance(value,BaseException):
                raise value
            return {'message':value,'usage':{'prompt_tokens':20,'completion_tokens':10},
                    'cost_cny':.001,'cost_status':'estimated_from_usage_not_invoice'}
        finally:
            with self.lock:
                self.active['now'] -= 1

    @staticmethod
    def default(c):
        node, role = c['node'], c['agent_id']
        if node.startswith('query-'):
            return {'content':'{"query":"standalone comparison"}'}
        if node == 'coordinate-plan':
            return {'content':json.dumps({'tasks':[{'question':'part1','query':'query1'},{'question':'part2','query':'query2'}]})}
        if node == 'coordinate-merge':
            return {'content':'{"gap":null}'}
        if node == 'coordinate-final':
            return message([{'text':'Supported comparison','evidence_page_ids':['p1','p2']}])
        step = int(node.split('-')[-1])
        if step == 1:
            return tool('search_pages',{'query':role})
        if step == 2:
            return tool('read_page',{'page_id':'p1' if role=='worker-1' else 'p2'})
        return {'content':'Evidence gathered; preserve period and condition differences.'}


class MultiAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        preview = self.root/'preview.png'
        Image.new('RGB',(32,32),'white').save(preview)
        self.pages = [{'page_id':f'p{i}','doc_id':f'd{i}','page_number':i,
                       'document_name':f'doc{i}.pdf','preview':str(preview),'source':'https://example.org'} for i in (1,2)]
        self.config = read_json('configs/agent-multi-demo.json')
        self.config['ledger'] = str(self.root/'test-costs.sqlite3')
        self.search = patch('foliorecall.agent.retrieve',return_value=(self.pages,'{}',{'request_seconds':.001}))
        self.search.start()
        self.addCleanup(self.search.stop)

    def run_task(self,provider=None,**kwargs):
        return ask_multi(kwargs.pop('question','Compare the two reports'),None,{},None,self.pages,
            kwargs.pop('config',self.config),provider or ScriptedProvider(),self.root/'task',
            evidence_texts={'p1':'First report 2020','p2':'Second report 2025'},cooldown_seconds=0,**kwargs)

    def test_parallel_calls_serial_search_and_current_evidence(self):
        provider = ScriptedProvider(barrier=threading.Barrier(2))
        value = self.run_task(provider)
        self.assertTrue(value['completed'],value)
        self.assertEqual(provider.active['peak'],2)
        self.assertEqual(value['searches'],2)
        self.assertEqual(value['distinct_pages'],2)
        self.assertEqual(value['model_requests'],9)
        self.assertEqual(provider.context,{})
        self.assertEqual(len({x['context']['request_sequence'] for x in provider.inputs}),len(provider.inputs))
        self.assertNotIn('base64',json.dumps(read_json(self.root/'task'/'attempt-0001.json')))
        final = next(x for x in provider.inputs if x['context']['node']=='coordinate-final')
        self.assertIn('base64',json.dumps(final['messages']))
        self.assertEqual(read_json(self.root/'task'/'final-evidence.json')['page_ids'],['p1','p2'])
        for call in provider.inputs:
            if call['context']['agent_id'].startswith('worker') or call['context']['node']=='coordinate-merge':
                self.assertNotIn('Return JSON with exactly:',call['messages'][0]['content'])

    def test_serial_option_keeps_same_allocation_and_result(self):
        provider = ScriptedProvider()
        value = self.run_task(provider,config=dict(self.config,max_parallel_workers=1))
        self.assertTrue(value['completed'])
        self.assertEqual(provider.active['peak'],1)
        self.assertEqual(value['model_requests'],9)

    def test_shared_page_delivered_to_both_workers_without_double_count(self):
        def script(c,m,n):
            if c['node'].endswith('-2') and c['agent_id'].startswith('worker'):
                return tool('read_page',{'page_id':'p1'})
            if c['node']=='coordinate-final':
                return message([{'text':'Shared evidence','evidence_page_ids':['p1']}])
        provider = ScriptedProvider(script)
        value = self.run_task(provider)
        self.assertTrue(value['completed'])
        self.assertEqual(value['distinct_pages'],1)
        for role in ('worker-1','worker-2'):
            call = next(x for x in provider.inputs if x['context']['node']==f'main-{role}-3')
            self.assertIn('base64',json.dumps(call['messages']))

    def test_gap_is_bounded_and_uses_reclaimed_budget(self):
        def script(c,m,n):
            if c['node']=='coordinate-merge':
                return {'content':json.dumps({'gap':{'worker_id':'worker-1','question':'time range','query':'period'}})}
            if c['node']=='gap-worker-1-1':
                return tool('read_page',{'page_id':'p2'})
            if c['node']=='gap-worker-1-2':
                return {'content':'Done'}
        p = ScriptedProvider(script)
        value = self.run_task(p)
        self.assertTrue(value['completed'],value)
        self.assertEqual(p.counts['coordinate-merge'],1)
        self.assertEqual(value['model_requests'],11)
        self.assertLessEqual(value['model_requests'],12)

    def test_invalid_plan_is_terminal_without_fallback_or_repair(self):
        provider = ScriptedProvider(lambda c,m,n:{'content':'{}'})
        value = self.run_task(provider)
        self.assertFalse(value['completed'])
        self.assertEqual(len(provider.inputs),1)

    def test_completed_resume_does_not_call_provider(self):
        provider = ScriptedProvider()
        original = self.run_task(provider)
        count = len(provider.inputs)
        self.assertEqual(self.run_task(provider,resume=True),original)
        self.assertEqual(len(provider.inputs),count)
        # Simulate a crash after the final graph checkpoint but before terminal JSON publication.
        (self.root/'task'/'attempt-0001.json').unlink()
        (self.root/'task'/'result.json').unlink()
        restored = self.run_task(provider,resume=True)
        self.assertTrue(restored['completed'])
        self.assertEqual(len(provider.inputs),count)
        self.assertGreaterEqual(restored['timing']['task_seconds'],original['timing']['task_seconds'])

    def test_known_network_retry_reuses_successful_calls_and_keeps_first(self):
        def script(c,m,n):
            if c['node']=='main-worker-2-2' and n==1:
                return TaskStopped('api_transport_error',{'ledger_id':'unknown','usage':None})
        provider = ScriptedProvider(script)
        first = self.run_task(provider)
        self.assertFalse(first['completed'])
        self.assertEqual(first['termination_reason'],'api_transport_error')
        second = self.run_task(provider,resume=True)
        self.assertTrue(second['completed'],second)
        self.assertEqual(provider.counts['coordinate-plan'],1)
        self.assertEqual(provider.counts['main-worker-1-2'],1)
        self.assertEqual(provider.counts['main-worker-2-2'],2)
        self.assertEqual(read_json(self.root/'task'/'attempt-0001.json'),first)
        self.assertGreater(second['budget_state']['unknown_output_tokens_reserved'],0)

    def test_crash_after_saved_response_resumes_without_reissuing_call(self):
        from foliorecall.multi_agent import Collaboration
        original = Collaboration.save_op
        crashed = []
        def save(runtime,key,value):
            original(runtime,key,value)
            if key.startswith('coordinate-plan') and value.get('outcome') and not crashed:
                crashed.append(True)
                raise KeyboardInterrupt()
        provider = ScriptedProvider()
        with patch.object(Collaboration,'save_op',save),self.assertRaises(KeyboardInterrupt):
            self.run_task(provider)
        value = self.run_task(provider,resume=True)
        self.assertTrue(value['completed'],value)
        self.assertEqual(provider.counts['coordinate-plan'],1)

    def test_unknown_inflight_is_not_replayed(self):
        provider = ScriptedProvider(lambda c,m,n:KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.run_task(provider)
        count = len(provider.inputs)
        value = self.run_task(provider,resume=True)
        self.assertEqual(value['termination_reason'],'unknown_inflight')
        self.assertFalse(value['budget_state']['budget_snapshot_complete'])
        self.assertEqual(len(provider.inputs),count)

    def test_unread_final_citation_fails_without_regeneration(self):
        def script(c,m,n):
            if c['node']=='coordinate-final':
                return message([{'text':'History','evidence_page_ids':['old-page']}])
        p = ScriptedProvider(script)
        value = self.run_task(p)
        self.assertFalse(value['completed'])
        self.assertEqual(p.counts['coordinate-final'],1)

    def test_context_rewrite_and_session_exclusion(self):
        history = [{'question':str(i),'retrieval_query':str(i),'answer':message(), 'completed':True} for i in range(5)]
        history.append(dict(history[0],question='failed',completed=False))
        p = ScriptedProvider()
        value = self.run_task(p,question='这两份报告呢？',conversation=history)
        self.assertTrue(value['completed'])
        self.assertEqual(value['context_turns'],3)
        self.assertEqual(p.counts['query-rewrite'],1)
        self.assertNotIn('query-translation',p.counts)
        first = json.loads(p.inputs[0]['messages'][1]['content'])
        self.assertEqual([x['question'] for x in first['previous_turns']],['2','3','4'])

    def test_configuration_change_cannot_resume(self):
        self.run_task()
        with self.assertRaisesRegex(ValueError,'身份'):
            self.run_task(config=dict(self.config,max_pages=7),resume=True)
        for value in (True,0,3,'2'):
            with self.assertRaises(ValueError):
                validate_agent_config(dict(self.config,max_parallel_workers=value))

    def test_single_task_without_forced_second_worker(self):
        def script(c,m,n):
            if c['node']=='coordinate-plan':
                return {'content':json.dumps({'tasks':[{'question':'one part','query':'one query'}]})}
            if c['node']=='coordinate-final':
                return message([{'text':'One report','evidence_page_ids':['p1']}])
        p = ScriptedProvider(script)
        value = self.run_task(p)
        self.assertTrue(value['completed'])
        self.assertEqual(len(value['graph']['workers']),1)
        self.assertFalse(any(x['context']['agent_id']=='worker-2' for x in p.inputs))

    def test_two_failures_keep_attempts_and_never_start_a_third(self):
        p = ScriptedProvider(lambda c,m,n:TaskStopped('api_http_503') if c['node']=='main-worker-2-2' else None)
        first = self.run_task(p)
        second = self.run_task(p,resume=True)
        count = len(p.inputs)
        self.assertFalse(first['completed']); self.assertFalse(second['completed'])
        self.assertEqual(self.run_task(p,resume=True),second)
        self.assertEqual(len(p.inputs),count)
        self.assertEqual(len(list((self.root/'task').glob('attempt-*.json'))),2)

    def test_final_can_resume_with_all_searches_already_used(self):
        def script(c,m,n):
            if c['node']=='coordinate-final' and n==1:
                return TaskStopped('api_transport_error')
        p = ScriptedProvider(script)
        config = dict(self.config,max_searches=2)
        first = self.run_task(p,config=config)
        self.assertEqual(first['searches'],2)
        second = self.run_task(p,config=config,resume=True)
        self.assertTrue(second['completed'],second)
        self.assertEqual(second['actual_searches'],0)
        self.assertEqual(second['actual_model_requests'],1)

    def test_cached_translation_debit_is_once_and_no_live_translation(self):
        from foliorecall.qa_translation import translation_identity
        question = '比较两份报告'
        cached = {'identity':translation_identity(question,self.config),'query':'Compare reports',
                  'requests':1,'output_tokens':12,'seconds':.01,'cost_cny':.001,
                  'api_calls':[{'ledger_id':'old-translation','usage':{'completion_tokens':12}}]}
        p = ScriptedProvider()
        value = self.run_task(p,question=question,translation=cached)
        self.assertTrue(value['completed'])
        self.assertNotIn('query-translation',p.counts)
        self.assertEqual(value['model_requests'],value['actual_model_requests']+1)
        self.assertEqual(value['budget_state']['translation_debit'],cached['identity'])

    def test_runtime_can_recover_multi_final_without_search_budget(self):
        from foliorecall.qa_runtime import can_recover
        value = {'mode':'multi-agent','termination_reason':'api_timeout','budget_state':{
            'budget_snapshot_complete':True,'requests_used':9,'known_output_tokens':100,
            'unknown_output_tokens_reserved':2048,'searches_used':4,'distinct_page_ids':['p1'],
            'active_seconds_used':1,'translation_debit':None}}
        self.assertTrue(can_recover(value,{'question':'English?'},self.config))
        self.assertFalse(can_recover(dict(value,mode='agent'),{'question':'English?'},self.config))

    def test_parent_requests_and_output_are_reserved_before_parallel_calls(self):
        from foliorecall.agent import AgentExecution
        from foliorecall.multi_agent import Collaboration
        class FullOutput(ScriptedProvider):
            def complete(self,messages,tools,max_tokens,timeout):
                record = super().complete(messages,tools,max_tokens,timeout)
                record['usage']['completion_tokens'] = max_tokens
                return record
        entered = threading.Barrier(3)
        release = threading.Event()
        def script(c,m,n):
            if c['agent_id'].startswith('worker'):
                entered.wait(5)
                release.wait(5)
            return {'content':'{}'}
        p = FullOutput(script)
        config = dict(self.config,max_requests=4,max_output_tokens=4096)
        e = AgentExecution('q','agent',None,{},None,self.pages,config,p)
        runtime = Collaboration(self.root/'budget',e,1)
        runtime.allocate(['worker-1','worker-2'])
        errors = []
        def run(role):
            try:
                runtime.request(role,role,[],TOOLS)
            except BaseException as exc:
                errors.append(exc)
        from foliorecall.agent import TOOLS
        threads = [threading.Thread(target=run,args=(r,)) for r in ('worker-1','worker-2')]
        for t in threads:t.start()
        entered.wait(5)
        try:
            self.assertEqual(runtime.resources()['requests_used'],2)
            self.assertEqual(runtime.resources()['unknown_output_tokens_reserved'],2048)
            with self.assertRaisesRegex(TaskStopped,'request_limit'):
                runtime.request('excess','coordinator',[])
        finally:
            release.set()
            for t in threads:t.join(5)
        self.assertEqual(errors,[])
        runtime.request('merge','coordinator',[],phase='merge',max_tokens=1024)
        runtime.request('final','coordinator',[],phase='final_answer',max_tokens=1024)
        self.assertEqual(runtime.resources()['known_output_tokens'],4096)
        self.assertEqual(runtime.resources()['requests_used'],4)

    def test_missing_optional_dependency_does_not_affect_single_agent(self):
        import sys
        from foliorecall.multi_agent import require_multiagent
        from foliorecall.agent import ask
        from tests.test_agent import FakeProvider
        with patch.dict(sys.modules,{'langgraph.graph':None}):
            with self.assertRaisesRegex(ValueError,'multiagent'):
                require_multiagent()
            value = ask('q','rag',None,{},None,self.pages,self.config,FakeProvider([message()]))
        self.assertTrue(value['completed'])

    def test_render_failure_still_preserves_terminal_record(self):
        from foliorecall.chat import stream_chat
        provider = ScriptedProvider()
        def execute(q,mode,history,event):
            return self.run_task(provider,on_event=event,conversation=history)
        with patch('foliorecall.demo.qa_display',side_effect=OSError('preview failed')):
            updates = list(stream_chat('q','multi-agent',None,execute,self.pages,threading.Lock(),self.root))
        self.assertFalse(updates[-1][-1])
        self.assertTrue(read_json(self.root/'task'/'result.json')['completed'])
        self.assertTrue(list(self.root.glob('qa-*.json')))

    def test_parallel_events_are_labeled_and_escaped(self):
        from foliorecall.chat import progress_html
        p = ScriptedProvider(barrier=threading.Barrier(2))
        events = []
        self.run_task(p,on_event=events.append)
        self.assertEqual([e['sequence'] for e in events],list(range(1,len(events)+1)))
        self.assertTrue(any(e.get('role')=='worker-1' for e in events))
        self.assertTrue(any(e.get('role')=='worker-2' for e in events))
        self.assertIn('检索Agent 1',progress_html(events))

    def test_batch_running_attempt_resumes_only_at_known_graph_boundary(self):
        from foliorecall.multi_agent import Collaboration
        from foliorecall.qa_runtime import execute_plan
        from tests.test_qa_runtime import job
        planned = dict(job(),mode='multi-agent')
        provider = ScriptedProvider()
        def execute(j,b,a):
            return ask_multi(j['question'],None,{},None,self.pages,self.config,provider,j['graph_output'],
                resume=(Path(j['graph_output'])/'manifest.json').exists(),cooldown_seconds=0)
        original = Collaboration.save_op
        crashed = []
        def save(runtime,key,value):
            original(runtime,key,value)
            if key.startswith('coordinate-plan') and value.get('outcome') and not crashed:
                crashed.append(True)
                raise KeyboardInterrupt()
        with patch.object(Collaboration,'save_op',save),self.assertRaises(KeyboardInterrupt):
            execute_plan(self.root/'batch',{'test':1},[planned],lambda:execute,[self.config])
        summary = execute_plan(self.root/'batch',{'test':1},[planned],lambda:execute,[self.config],resume=True)
        self.assertEqual(summary['primary']['completed'],1)
        self.assertEqual(summary['recovery']['attempted'],0)
        self.assertEqual(provider.counts['coordinate-plan'],1)

    def test_batch_unknown_call_keeps_old_running_guard(self):
        from foliorecall.qa_runtime import execute_plan
        from tests.test_qa_runtime import job
        planned = dict(job(),mode='multi-agent')
        provider = ScriptedProvider(lambda c,m,n:KeyboardInterrupt())
        def execute(j,b,a):
            return ask_multi(j['question'],None,{},None,self.pages,self.config,provider,j['graph_output'])
        with self.assertRaises(KeyboardInterrupt):
            execute_plan(self.root/'batch',{'test':1},[planned],lambda:execute,[self.config])
        def forbidden():
            raise AssertionError('unknown call must not initialize executor')
        result = execute_plan(self.root/'batch',{'test':1},[planned],forbidden,[self.config],resume=True)
        self.assertEqual(result['stopped'],'incomplete_budget')
        self.assertEqual(len(provider.inputs),1)

    def test_longest_parallel_service_wait_is_preserved(self):
        from foliorecall.qa_runtime import latest_service_wait
        calls = [{'retry_after_status':'resolved','retry_after_seconds':120},
                 {'retry_after_status':'absent'}]
        self.assertEqual(latest_service_wait([{'result':{'mode':'multi-agent','api_calls':calls}}]),(120,None))
        calls[0]['retry_after_status']='unresolved'
        self.assertEqual(latest_service_wait([{'result':{'mode':'multi-agent','api_calls':calls}}])[1],'retry_after_unresolved')

    def test_temporary_cost_ledger_reservation_failure_is_not_sent(self):
        from foliorecall.agent import AgentExecution, TOOLS
        from foliorecall.multi_agent import Collaboration
        from foliorecall.generation import CostLedger
        ledger = CostLedger(self.root/'isolated.sqlite3',13.4,self.config['model'])
        entered, release = threading.Event(), threading.Event()
        class Provider:
            context = {}
            def complete(self,messages,tools,max_tokens,timeout):
                key = ledger.reserve(6,self.context)
                entered.set()
                release.wait(5)
                raise TaskStopped('api_transport_error',{'ledger_id':key,'usage':None,'reservation_cny':6})
        e = AgentExecution('q','agent',None,{},None,self.pages,self.config,Provider())
        runtime = Collaboration(self.root/'cost-task',e,1)
        runtime.allocate(['worker-1','worker-2'])
        failures = []
        def run():
            try:runtime.request('call1','worker-1',[],TOOLS)
            except TaskStopped as exc:failures.append(str(exc))
        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(entered.wait(5))
        try:
            with self.assertRaisesRegex(TaskStopped,'phase_budget_exhausted'):
                runtime.request('call2','worker-2',[],TOOLS)
        finally:
            release.set(); thread.join(5)
        self.assertEqual(failures,['api_transport_error'])
        self.assertEqual(runtime.resources()['requests_used'],1)
        self.assertEqual(ledger.total(),6)
        self.assertEqual(len(ledger.records()),1)

    def test_cli_collaboration_and_completed_resume_loads_no_model(self):
        import io
        import sys
        from foliorecall.__main__ import main
        from foliorecall.io import write_json
        for name, value in [('teacher',{}),('student',{}),('agent',self.config),('corpus',{})]:
            write_json(self.root/f'{name}.json',value)
        args = ['foliorecall','ask','Compare reports','--mode','multi-agent',
                '--index',str(self.root/'index'),'--config',str(self.root/'teacher.json'),
                '--query-config',str(self.root/'student.json'),'--agent-config',str(self.root/'agent.json'),
                '--public-corpus',str(self.root/'corpus.json'),'--output',str(self.root/'cli-task')]
        p = ScriptedProvider()
        texts = {'p1':'First report','p2':'Second report'}
        with patch('foliorecall.search.load_index',return_value=(None,self.pages)), \
             patch('foliorecall.agent.load_public_corpus',return_value=texts), \
             patch('foliorecall.query.load_search_session',return_value=(None,None,self.pages,None)), \
             patch('foliorecall.generation.Bailian',return_value=p),patch.object(sys,'argv',args),patch('sys.stdout',io.StringIO()):
            self.assertEqual(main(),0)
        count = len(p.inputs)
        with patch('foliorecall.search.load_index',return_value=(None,self.pages)), \
             patch('foliorecall.agent.load_public_corpus',return_value=texts), \
             patch('foliorecall.query.load_search_session',side_effect=AssertionError('model loaded')), \
             patch('foliorecall.generation.Bailian',side_effect=AssertionError('provider initialized')), \
             patch.object(sys,'argv',args+['--resume']),patch('sys.stdout',io.StringIO()):
            self.assertEqual(main(),0)
        self.assertEqual(len(p.inputs),count)

    def test_service_wait_prevents_second_request_after_error(self):
        def script(c,m,n):
            if c['node']=='coordinate-plan' and n==1:
                return TaskStopped('api_http_429',{'retry_after_status':'resolved','retry_after_seconds':120})
        p = ScriptedProvider(script)
        first = self.run_task(p)
        self.assertEqual(first['termination_reason'],'api_http_429')
        with patch('foliorecall.multi_agent.time.sleep') as sleep:
            second = self.run_task(p,resume=True)
        sleep.assert_called_once_with(120)
        self.assertTrue(second['completed'])

    def test_bad_merge_does_not_regenerate(self):
        def bad(c,m,n):
            if c['node']=='coordinate-merge':return {'content':'{"gap":42}'}
        p = ScriptedProvider(bad)
        value = self.run_task(p)
        self.assertFalse(value['completed'])
        self.assertEqual(p.counts['coordinate-merge'],1)
        self.assertNotIn('coordinate-final',p.counts)

    def test_task_timeout_does_not_send_request(self):
        from foliorecall.multi_agent import Collaboration
        p = ScriptedProvider()
        with patch.object(Collaboration,'check_time',side_effect=TaskStopped('task_timeout')):
            value = self.run_task(p)
        self.assertFalse(value['completed'])
        self.assertEqual(value['termination_reason'],'task_timeout')
        self.assertEqual(p.inputs,[])

    def test_malformed_tool_call_becomes_terminal(self):
        p = ScriptedProvider(lambda c,m,n:{'content':None,'tool_calls':[None]}
                             if c['node']=='main-worker-1-1' else None)
        value = self.run_task(p)
        self.assertFalse(value['completed'])
        self.assertEqual(value['termination_reason'],'invalid_answer_or_evidence')

    def test_provider_timeout_retains_successful_sibling(self):
        p = ScriptedProvider(lambda c,m,n:TaskStopped('api_timeout')
                             if c['node']=='main-worker-1-2' and n==1 else None)
        first = self.run_task(p)
        self.assertEqual(first['termination_reason'],'api_timeout')
        self.assertIn('p2',first['budget_state']['distinct_page_ids'])
        second = self.run_task(p,resume=True)
        self.assertTrue(second['completed'])
        self.assertEqual(p.counts['main-worker-2-2'],1)


if __name__ == '__main__':
    unittest.main()
