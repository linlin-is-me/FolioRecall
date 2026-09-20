import json
import unittest
from unittest.mock import patch

from foliorecall.agent import answer_contract
from foliorecall.generation import TaskStopped
from scripts.prepare_qa_repair import prepare, repair_configuration
from foliorecall.io import write_json
from tests import test_agent as fixtures
FakeProvider, message, tool = fixtures.FakeProvider, fixtures.message, fixtures.tool


class FinalizationTests(unittest.TestCase):
    setUp = fixtures.AgentTests.setUp
    run_ask = fixtures.AgentTests.run_ask

    def test_both_natural_endings_use_exactly_one_final_without_draft(self):
        for draft in ('Preface then JSON', message()['content']):
            p=FakeProvider([tool('read_page',{'page_id':'p1'}), {'content':draft},
                message([{'text':'42','evidence_page_ids':['p1']}])])
            r=self.run_ask(p)
            self.assertTrue(r['completed'])
            self.assertEqual([c['phase'] for c in r['api_calls']],['tool_decision','tool_decision','final_answer'])
            self.assertEqual(len(p.inputs),3)
            self.assertIsNone(p.inputs[-1][1])
            self.assertFalse(any(m.get('content') == draft for m in p.inputs[-1][0]))
            context=json.dumps(p.inputs[-1][0])
            self.assertIn('data:image/png;base64,',context)
            self.assertIn('Energy is 42',context)
            self.assertIn('What?',context)
            self.assertEqual(r['trace'][-1]['reason'],'natural_no_tool')
            self.assertEqual(r['trace'][-1]['source_request_sequence'],2)
            self.assertAlmostEqual(r['timing']['api_seconds'],
                r['timing']['tool_decision_api_seconds']+r['timing']['final_answer_api_seconds'])

    def test_final_invalid_has_diagnostic_and_no_repair_call(self):
        p=FakeProvider([{'content':'Finished.'},{'content':'prefix {}'}])
        r=self.run_ask(p)
        self.assertEqual(r['termination_reason'],'invalid_answer_or_evidence')
        self.assertEqual(r['failure']['output_diagnostic']['request_phase'],'final_answer')
        self.assertEqual(len(p.inputs),2)
        self.assertAlmostEqual(r['estimated_cost_cny'],.002)

    def test_provider_final_payload_and_ledger_phase(self):
        import httpx
        from foliorecall.generation import Bailian,CostLedger
        provider=object.__new__(Bailian)
        provider.key='mock';provider.base='https://unused'
        provider.ledger=CostLedger(self.config['ledger'],150);provider.context={}
        payloads=[]
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):
                payloads.append(kwargs['json'])
                answer={'content':'Finished.'} if len(payloads)==1 else message()
                return httpx.Response(200,json={'usage':{'prompt_tokens':10,'completion_tokens':5},
                    'choices':[{'finish_reason':'stop','message':answer}]})
        with patch('httpx.AsyncClient',Client):r=self.run_ask(provider)
        self.assertTrue(r['completed'])
        self.assertIn('tools',payloads[0]);self.assertNotIn('response_format',payloads[0])
        self.assertNotIn('tools',payloads[1]);self.assertEqual(payloads[1]['response_format'],{'type':'json_object'})
        self.assertEqual([json.loads(c['context'])['phase'] for c in provider.ledger.records()],
                         ['tool_decision','final_answer'])

    def test_cached_request_reserves_final_and_history_only_once(self):
        from foliorecall.agent import ask
        from foliorecall.qa_translation import translation_identity
        config=dict(self.config,max_requests=3)
        cache={'identity':translation_identity('多少？',config),'query':'energy','requests':1,
               'output_tokens':5,'seconds':1.,'cost_cny':.001,
               'api_calls':[{'ledger_id':'old','usage':{'prompt_tokens':10,'completion_tokens':5}}]}
        p=FakeProvider([{'content':'Finished.'},message()])
        r=ask('多少？','agent',None,{'device':'cpu'},None,self.pages,config,p,translation=cache)
        self.assertEqual((r['model_requests'],r['actual_model_requests'],r['output_tokens']),(3,2,15))
        self.assertGreaterEqual(r['budget_state']['active_seconds_used'],1.)
        self.assertIn('in Chinese',p.inputs[-1][0][-1]['content'])
        self.assertIsNone(p.inputs[-1][1])

    def test_last_request_direct_final_and_no_tools(self):
        p=FakeProvider([message()]);r=self.run_ask(p,max_requests=1)
        self.assertTrue(r['completed'])
        self.assertIsNone(p.inputs[0][1])
        self.assertEqual(r['trace'][-1]['reason'],'resource_limit')

    def test_no_time_after_natural_stop_means_no_final_request(self):
        now=[0.]
        class Provider(FakeProvider):
            def complete(self,*args):
                response=super().complete(*args);now[0]=181.;return response
        p=Provider([{'content':'Finished.'}])
        with patch('foliorecall.agent.time.monotonic',side_effect=lambda:now[0]):
            r=self.run_ask(p)
        self.assertEqual(r['termination_reason'],'task_timeout')
        self.assertEqual(len(p.inputs),1)

    def test_unknown_final_usage_reserved_and_phase_budget_not_sent(self):
        for error,reason in [(TaskStopped('api_timeout',{'ledger_id':'x'}),'api_timeout'),
                             (TaskStopped('phase_budget_exhausted'),'phase_budget_exhausted')]:
            p=FakeProvider([{'content':'Finished.'},error]);r=self.run_ask(p)
            self.assertEqual(r['termination_reason'],reason)
            self.assertEqual(r['api_calls'][-1]['phase'],'final_answer')
            self.assertEqual(r['model_requests'],2 if reason=='api_timeout' else 1)
            self.assertEqual(r['budget_state']['unknown_output_tokens_reserved'],2048 if reason=='api_timeout' else 0)

    def test_partial_answer_contract_and_strict_empty_citations(self):
        self.assertIn('exclusively in unresolved',answer_contract('多少？')[1])
        good={'claims':[{'text':'42','evidence_page_ids':['p1']}],
              'insufficient_evidence':True,'unresolved':['Missing comparison.']}
        r=self.run_ask(FakeProvider([{'content':json.dumps(good)}]),'rag')
        self.assertTrue(r['completed'])
        good['claims'][0]['evidence_page_ids']=[]
        r=self.run_ask(FakeProvider([{'content':json.dumps(good)}]),'rag')
        self.assertEqual(r['failure']['output_diagnostic']['code'],'empty_citations')

    def test_repair_identity_and_missing_original_cache_fail_before_output(self):
        src=self.root/'source';src.mkdir()
        write_json(src/'agent.json',dict(self.config,phase_budget_cny=5.002621))
        write_json(src/'budget-before.json',{'ledger':self.config['ledger'],'absolute_cap_cny':5.002621})
        write_json(src/'suite.json',{'common':{}})
        a,s,b=repair_configuration(src,self.root/'new',[],'qa-pilot-repair-v2')
        self.assertEqual(s['suite_id'],'qa-pilot-repair-v2')
        self.assertEqual(a['phase_budget_cny'],5.002621)
        with self.assertRaises(ValueError):prepare(src,self.root/'new','qa-pilot-repair-v2',src)
        self.assertFalse((self.root/'new').exists())


if __name__=='__main__':unittest.main()
