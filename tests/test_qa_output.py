import json
from pathlib import Path
from unittest.mock import patch
import unittest

from foliorecall.agent import diagnostic_text, validate_answer, AnswerValidationError
from foliorecall.io import write_json
from foliorecall.qa_runtime import transient
from foliorecall.qa_translation import translation_identity
from scripts.prepare_qa_repair import repair_configuration
from scripts.reconcile_qa_pilot import reconcile_records
from tests import test_agent as fixtures
FakeProvider, message, tool = fixtures.FakeProvider, fixtures.message, fixtures.tool


class OutputTests(unittest.TestCase):
    setUp=fixtures.AgentTests.setUp
    run_ask=fixtures.AgentTests.run_ask

    def test_provider_metadata_whitelist_and_known_usage(self):
        import httpx
        from foliorecall.generation import Bailian,CostLedger
        provider=object.__new__(Bailian)
        provider.key='mock-secret';provider.base='https://unused'
        provider.ledger=CostLedger(self.config['ledger'],150);provider.context={}
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):
                return httpx.Response(200,json={'id':'response','model':'fixed','usage':{'prompt_tokens':10,'completion_tokens':5},
                    'private_extra':'not retained','choices':[{'finish_reason':'length','message':{
                        'role':'assistant','content':'{bad','reasoning_content':'hidden-reasoning'}}]})
        with patch('httpx.AsyncClient',Client):
            z=self.run_ask(provider,'rag')
        d=z['failure']['output_diagnostic']
        self.assertEqual((d['response_id'],d['response_model'],d['finish_reason']),('response','fixed','length'))
        self.assertTrue(z['cost_complete'])
        for forbidden in ('hidden-reasoning','private_extra','mock-secret'):
            self.assertNotIn(forbidden,json.dumps(z))

    def test_precise_validation_codes(self):
        good={'claims':[{'text':'fact','evidence_page_ids':['p1']}], 'insufficient_evidence':False,'unresolved':[]}
        cases=[([], 'answer_shape','$'),
               (dict(good,claims=None),'field_type','$/claims'),
               (dict(good,claims=[]),'empty_complete_answer','$/claims'),
               (dict(good,unresolved=[5]),'field_type','$/unresolved/0')]
        for ids,code,path in [([], 'empty_citations','$/claims/0/evidence_page_ids'),
                              ([1],'citation_type','$/claims/0/evidence_page_ids/0'),
                              (['p2'],'unread_citation','$/claims/0/evidence_page_ids/0')]:
            cases.append((dict(good,claims=[{'text':'x','evidence_page_ids':ids}]),code,path))
        for value,code,path in cases:
            with self.subTest(code=code),self.assertRaises(AnswerValidationError) as caught:
                validate_answer(value,{'p1'})
            self.assertEqual((caught.exception.code,caught.exception.path),(code,path))
        self.assertEqual(validate_answer(good,{'p1'}),good)

    def test_parse_diagnostic_keeps_usage_without_extra_call(self):
        class Provider(FakeProvider):
            def complete(self,*args):
                return super().complete(*args)|{'ledger_id':'ledger','response_id':'response',
                    'response_model':'fixed-model','finish_reason':'length'}
        p=Provider([{'content':'```json\n{bad'}])
        z=self.run_ask(p,'rag')
        d=z['failure']['output_diagnostic']
        self.assertEqual((d['stage'],d['code']),('answer_parse','json_syntax'))
        self.assertEqual((d['ledger_id'],d['request_sequence'],d['finish_reason']),('ledger',1,'length'))
        self.assertEqual(d['read_page_ids'],['p1'])
        self.assertEqual(z['estimated_cost_cny'],.001)
        self.assertTrue(z['cost_complete'])
        self.assertEqual(len(p.inputs),1)
        self.assertFalse(transient(z['termination_reason']))

    def test_agent_validation_and_nontext_content(self):
        z=self.run_ask(FakeProvider([message([{'text':'x','evidence_page_ids':['p1']}])]))
        self.assertEqual(z['failure']['output_diagnostic']['code'],'unread_citation')
        z=self.run_ask(FakeProvider([{'content':{'reasoning_content':'secret'}}]),'rag')
        d=z['failure']['output_diagnostic']
        self.assertEqual(d['code'],'content_type')
        self.assertIsNone(d['visible_output']['text'])
        self.assertNotIn('secret',json.dumps(z))

    def test_redaction_and_limits(self):
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'credential-from-env'}):
            d=diagnostic_text('credential-from-env sk-other_123 Bearer token-123\n'
                'Authorization: token=private\n<think>hidden</think> '
                '"reasoning_content":"hidden-too" data:image/png;base64,YWJjZA==\n'
                'visible '*6000)
        for secret in ('credential-from-env','sk-other_123','token-123','token=private','hidden','YWJjZA=='):
            self.assertNotIn(secret,d['text'])
        self.assertEqual(len(d['text']),32768)
        self.assertTrue(d['truncated'])

    def test_language_contract_rag_live_and_cached(self):
        p=FakeProvider([{'content':'{"query":"energy"}'},message([{'text':'数值为42','evidence_page_ids':['p1']}])])
        z=self.run_ask(p,'rag','多少？')
        self.assertEqual(z['answer_language'],'zh')
        self.assertIn('in Chinese',p.inputs[-1][0][-1]['content'])
        self.assertEqual(z['model_requests'],2)
        from foliorecall.agent import ask
        cache={'identity':translation_identity('多少？',self.config),'query':'energy','requests':1,
            'output_tokens':5,'seconds':1.,'cost_cny':.001,
            'api_calls':[{'ledger_id':'old','usage':{'prompt_tokens':10,'completion_tokens':5}}]}
        p=FakeProvider([message()])
        z=ask('多少？','rag',None,{'device':'cpu'},None,self.pages,self.config,p,translation=cache)
        self.assertIn('in Chinese',p.inputs[-1][0][-1]['content'])
        self.assertEqual((z['model_requests'],z['actual_model_requests']),(2,1))
        self.assertEqual(z['translation_status'],'cached_quality_only')

    def test_agent_language_normal_and_forced_final(self):
        for lang,q,word in [('en','What?','in English'),('zh','多少？','in Chinese')]:
            replies=([{'content':'{"query":"energy"}'}] if lang=='zh' else [])+[tool('read_page',{'page_id':'p1'}),message()]
            p=FakeProvider(replies)
            z=self.run_ask(p,question=q,max_requests=3 if lang=='zh' else 2)
            self.assertEqual(z['answer_language'],lang)
            for messages,tools,_,_ in p.inputs[(1 if lang=='zh' else 0):]:
                self.assertIn(word,messages[-1]['content'])
            self.assertIsNone(p.inputs[-1][1])
            self.assertTrue(z['completed'])

    def test_page_error_not_falsely_diagnosed_as_model_output(self):
        with patch('foliorecall.agent.read_evidence',side_effect=OSError('private/path')):
            z=self.run_ask(FakeProvider([]),'rag')
        self.assertNotIn('output_diagnostic',z['failure'])
        self.assertEqual(z['actual_model_requests'],0)
        self.assertNotIn('private/path',json.dumps(z))

    def test_inherited_cap_and_historical_translation_not_new_charge(self):
        src=self.root/'source';src.mkdir()
        write_json(src/'agent.json',dict(self.config,phase_budget_cny=5.002621))
        write_json(src/'budget-before.json',{'ledger':self.config['ledger'],'absolute_cap_cny':5.002621})
        write_json(src/'suite.json',{'suite_id':'old','common':{},'domains':[]})
        old={'id':'t','cny':.000839,'status':'estimated_from_usage','context':json.dumps({'execution_id':'translate'})}
        agent,suite,before=repair_configuration(src,self.root/'repair',[old])
        self.assertEqual(agent['phase_budget_cny'],5.002621)
        new={'id':'q','cny':.03,'status':'estimated_from_usage','context':json.dumps({'execution_id':'new','attempt':1})}
        report=reconcile_records(before,[old,new],'translate','new')
        self.assertEqual(report['groups']['translation']['call_count'],0)
        self.assertEqual(report['pilot']['known_cost_cny'],.03)


if __name__=='__main__': unittest.main()
