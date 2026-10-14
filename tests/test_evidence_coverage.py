import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.agent import ask, source_access_status, EVIDENCE_COVERAGE_POLICY
from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from foliorecall.qa_reviews import save_judgments
from scripts.run_evidence_coverage import task_plan, ordered_jobs, conversation_context
from tests.test_agent import FakeProvider, tool, message


class SnapProvider(FakeProvider):
    def complete(self, messages, tools, max_tokens, timeout):
        result = super().complete(messages, tools, max_tokens, timeout)
        self.inputs[-1] = (copy.deepcopy(messages), tools, max_tokens, timeout)
        return result


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json('configs/agent-bailian.json')
        self.pages = [{'page_id':'a:1','doc_id':'a','document_name':'same.pdf','page_number':1},
                      {'page_id':'b:23','doc_id':'b','document_name':'same.pdf','page_number':23}]

    def execute(self, responses, enabled=None, mode='agent', **config):
        values = dict(self.config, **config)
        if enabled is not None: values['evidence_coverage_search'] = enabled
        provider = SnapProvider(responses)
        events = []
        with patch('foliorecall.agent.retrieve', return_value=(self.pages,'',{'request_seconds':.001})), \
             patch('foliorecall.agent.read_evidence', return_value=({'type':'image_url'}, {'seconds':.001})):
            result = ask('Compare both reports',mode,None,{},None,self.pages,values,provider,
                         evidence_texts={'b:23':'References'},on_event=events.append)
        return result, provider, events

    def test_disabled_preserves_requests_and_rag_ignores_enabled(self):
        responses = [tool('search_pages',{'query':'q'}),tool('read_page',{'page_id':'a:1'}),
                     {'content':'done'},message([{'text':'fact','evidence_page_ids':['a:1']}])]
        _, base, _ = self.execute(responses)
        _, disabled, _ = self.execute(responses,False)
        self.assertEqual([x[:3] for x in base.inputs], [x[:3] for x in disabled.inputs])
        a, base, _ = self.execute([message()],False,'rag')
        b, enabled, _ = self.execute([message()],True,'rag')
        self.assertEqual(a['answer'],b['answer'])
        self.assertEqual([x[:3] for x in base.inputs],[x[:3] for x in enabled.inputs])
        self.assertNotIn('source_access_status',b)

    def test_candidates_are_not_semantic_coverage_and_final_citations_stay_current(self):
        responses = [tool('search_pages',{'query':'q'}),tool('read_page',{'page_id':'a:1'}),
                     {'content':'done'},message([{'text':'fact','evidence_page_ids':['a:1']}])]
        result, provider, events = self.execute(responses,True)
        self.assertTrue(result['completed'])
        self.assertEqual(result['model_requests'],4)
        self.assertEqual(len(result['source_access_status']),2)
        self.assertEqual(result['source_access_status'][1]['read_pages'],[])
        for messages, _, _, _ in provider.inputs:
            self.assertLessEqual(sum(m.get('content','').startswith('SOURCE_ACCESS_STATUS')
                for m in messages if isinstance(m.get('content'),str)),1)
        self.assertIn(EVIDENCE_COVERAGE_POLICY,json.dumps(provider.inputs[0][0]))
        self.assertTrue(any(e['event']=='source_access_status' for e in events))
        self.assertNotIn('b:23',provider.inputs[-1][0][-1]['content'])

    def test_repeat_and_failed_reads_do_not_grant_access(self):
        result, _, _ = self.execute([tool('read_page',{'page_id':'a:1'}),
            tool('read_page',{'page_id':'a:1'}),tool('read_page',{'page_id':'unknown'}),
            {'content':'done'},message()], True)
        self.assertEqual(result['distinct_pages'],1)
        self.assertEqual(len(result['source_access_status'][0]['read_pages']),1)
        self.assertTrue(any(t.get('status')=='error' for t in result['trace']))

    def test_bad_flag_is_rejected_before_provider(self):
        for value in (None,0,1,'true',[]):
            config = dict(self.config,evidence_coverage_search=value)
            with self.assertRaises(ValueError):
                ask('q','agent',None,{},None,self.pages,config,FakeProvider([]))

    def test_same_document_name_uses_distinct_document_ids(self):
        state=source_access_status({p['page_id']:p for p in self.pages},['a:1','a:1','b:23'],['a:1','a:1'])
        self.assertEqual([s['doc_id'] for s in state],['a','b'])
        self.assertEqual(len(state[0]['read_pages']),1)

    def test_exact_job_counts_and_alternating_session_blocks(self):
        tasks, confirm = task_plan()
        jobs = ordered_jobs(tasks)
        self.assertEqual(len(jobs),24)
        self.assertEqual(len({digest(j['key']) for j in jobs}),24)
        self.assertEqual([j['arm'] for j in jobs[:4]],['baseline','baseline','candidate','candidate'])
        self.assertEqual([j['arm'] for j in jobs[4:8]],['candidate','candidate','baseline','baseline'])
        self.assertEqual(sum(t['language']=='zh' for t in tasks),6)
        self.assertEqual(sum(t['language']=='zh' for t in confirm),3)

    def test_context_restores_own_successful_retry_not_other_arm_or_session(self):
        tasks, _ = task_plan()
        jobs=ordered_jobs(tasks)
        previous, current = jobs[:2]
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'runs/attempts'/digest(previous['key'])
            first={'key':previous['key'],'attempt':1,'state':'finished','answer_digest':'failed',
                   'result':{'completed':False}}
            second={'key':previous['key'],'attempt':2,'state':'finished','answer_digest':'retry',
                    'result':{'completed':True,'question':'q','retrieval_query':'q','answer':{}}}
            atomic_json(folder/'attempt-0001.json',first)
            self.assertEqual(conversation_context(temporary,jobs,current),([],[]))
            atomic_json(folder/'attempt-0002.json',second)
            context,sources=conversation_context(temporary,jobs,current)
            self.assertEqual(len(context),1); self.assertEqual(sources[0]['attempt'],2)
            self.assertEqual(conversation_context(temporary,jobs,jobs[3]),([],[]))

    def test_review_null_and_empty_answer_coverage_rules(self):
        key={'suite':'x','condition':'hybrid-agent','domain':'jrc','task_id':'t','language':'zh','repeat':0}
        answer={'claims':[],'insufficient_evidence':True,'unresolved':['missing']}
        record={'key':key,'attempt':1,'state':'finished','answer_digest':digest(answer),
                'task':{'answerable':False,'reference_facts':['explanation']},
                'result':{'answer':answer,'completed':True,'timing':{'task_seconds':1}}}
        judgment={'task':'t','language':'zh','answer_digest':digest(answer),'claims':[],
                  'facts':[{'score':0,'reason':'Missing explanation'}],'insufficiency':.5,
                  'insufficiency_reason':'partial','language_ok':True,'diagnosis':'missing', 'source_pages':[]}
        with tempfile.TemporaryDirectory() as temporary:
            save_judgments([record],[judgment],temporary)
            saved=read_json(next(Path(temporary).glob('*.json')))
            self.assertIsNone(saved['fact_correctness']); self.assertEqual(saved['answer_coverage'],0)
            with self.assertRaises(ValueError): save_judgments([record],[judgment],temporary)
            record['task']['reference_facts']=[]
            judgment['facts']=[]
            save_judgments([record],[judgment],Path(temporary)/'null')
            self.assertIsNone(read_json(next((Path(temporary)/'null').glob('*.json')))['answer_coverage'])


if __name__=='__main__': unittest.main()
