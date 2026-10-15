import copy
import json
import threading
import unittest

from foliorecall.generation import TaskStopped, validate_agent_config
from foliorecall.io import read_json
from foliorecall.multi_agent_research import validate_plan, calculation_checks
from foliorecall.qa_runtime import can_recover
from tests.test_agent import tool
from tests import test_multi_agent as basic


def plan(dependency=False):
    return {'requirements':[{'id':'r1','text':'Report one method'},{'id':'r2','text':'Report two method'}],
        'tasks':[{'id':f't{i}','name':f'Report {i} evidence','question':f'Find method {i}',
                  'query':f'method {i}','scope':f'document {i}','requirement_ids':[f'r{i}'],
                  'depends_on':['t1'] if dependency and i==2 else []} for i in (1,2)]}


def draft(version=0):
    return {'answer':{'claims':[{'text':'Supported methods','evidence_page_ids':['p1','p2']}],
                     'insufficient_evidence':False,'unresolved':[]},
            'coverage':[{'requirement_id':f'r{i}','covered':True,'note':'original page'} for i in (1,2)],
            'calculations':[]}


def verification(version=0, issue=None, supported=True):
    return {'claims':[{'id':f'v{version}-c1','supported':supported,'reason':'Actual original pages checked'}],
            'issues':[] if issue is None else [issue]}


class ResearchProvider(basic.ScriptedProvider):
    @staticmethod
    def default(c):
        node, role = c['node'], c['agent_id']
        if node=='research-query':
            return {'content':'{"query":"standalone methods"}'}
        if node=='research-plan':
            return {'content':json.dumps(plan())}
        if node.startswith('research-analysis-v'):
            return {'content':json.dumps(draft(int(node[-1])))}
        if node.startswith('research-verify-v'):
            return {'content':json.dumps(verification(int(node[-1])))}
        step = int(node.split('-')[-1])
        if step==1:
            return tool('search_pages',{'query':role+' methods'})
        if step==2:
            return tool('read_page',{'page_id':'p1' if role=='worker-1' else 'p2'})
        return {'content':'Evidence received; report-specific scope retained.'}


class ResearchTests(unittest.TestCase):
    def setUp(self):
        basic.MultiAgentTests.setUp(self)
        self.config.update(collaboration_workflow='research',max_requests=16,max_searches=6,
                           max_pages=10,max_output_tokens=16000,task_timeout=300)

    def run_task(self, provider=None, **kwargs):
        return basic.MultiAgentTests.run_task(self,provider or ResearchProvider(),**kwargs)

    def test_research_parallel_roles_and_actual_pages(self):
        p = ResearchProvider(barrier=threading.Barrier(2))
        value = self.run_task(p)
        self.assertTrue(value['completed'],value)
        self.assertEqual(p.active['peak'],2)
        self.assertEqual(value['research']['verification_status'],'passed')
        self.assertEqual(value['actual_model_requests'],9)
        self.assertEqual(len(value['research']['tasks']),2)
        self.assertEqual(value['research']['pages'][0]['received_by'],['worker-1'])
        for node in ('research-analysis-v0','research-verify-v0'):
            self.assertIn('base64',json.dumps(next(c for c in p.inputs if c['context']['node']==node)['messages']))
        self.assertNotIn('base64',json.dumps(value))
        self.assertEqual(read_json(self.root/'task'/'final-evidence.json')['page_ids'],['p1','p2'])

    def test_research_dependency_is_sequential_and_owned(self):
        p = ResearchProvider(lambda c,m,n:{'content':json.dumps(plan(True))} if c['node']=='research-plan' else None)
        value = self.run_task(p)
        self.assertTrue(value['completed'],value)
        nodes = [c['context']['node'] for c in p.inputs]
        self.assertLess(nodes.index('research-main-worker-1-3'),nodes.index('research-main-worker-2-1'))
        self.assertEqual(p.active['peak'],1)

    def test_research_analysis_only_rework_and_no_free_final_rewrite(self):
        issue = dict(requirement_id='r1',task_id='t1',kind='condition',detail='Preserve the survey period',needs_retrieval=False)
        p = ResearchProvider(lambda c,m,n:{'content':json.dumps(verification(0,issue,False))}
                             if c['node']=='research-verify-v0' else None)
        value = self.run_task(p)
        self.assertTrue(value['completed'],value)
        self.assertTrue(value['research']['rework']['triggered'])
        self.assertFalse(value['research']['rework']['retrieval'])
        self.assertEqual(value['actual_model_requests'],11)
        self.assertEqual(p.inputs[-1]['context']['node'],'research-verify-v1')

    def test_research_retrieval_rework_two_calls_then_analysis(self):
        issue = dict(requirement_id='r1',task_id='t1',kind='missing_evidence',detail='Missing survey scope',
                     needs_retrieval=True,query='survey scope')
        def script(c,m,n):
            if c['node']=='research-verify-v0':
                return {'content':json.dumps(verification(0,issue))}
            if c['node']=='research-rework-worker-1-1':
                return tool('search_pages',{'query':'survey additional scope'})
            if c['node']=='research-rework-worker-1-2':
                return tool('read_page',{'page_id':'p2'})
        p = ResearchProvider(script)
        value = self.run_task(p)
        self.assertTrue(value['completed'],value)
        self.assertEqual(value['actual_model_requests'],13)
        self.assertNotIn('research-rework-worker-1-3',p.counts)
        self.assertEqual(len(value['research']['verifications']),2)

    def test_research_no_rework_prunes_without_rewriting(self):
        p = ResearchProvider(lambda c,m,n:{'content':json.dumps(verification(0,supported=False))}
                             if c['node']=='research-verify-v0' else None)
        value = self.run_task(p,config=dict(self.config,verification_rework=False))
        self.assertTrue(value['completed'],value)
        self.assertEqual(value['answer']['claims'],[])
        self.assertTrue(value['answer']['insufficient_evidence'])
        self.assertEqual(value['research']['verification_status'],'limited')
        self.assertNotIn('research-analysis-v1',p.counts)

    def test_research_verification_network_failure_can_resume_without_search(self):
        p = ResearchProvider(lambda c,m,n:TaskStopped('api_transport_error')
                             if c['node']=='research-verify-v0' and n==1 else None)
        first = self.run_task(p)
        self.assertFalse(first['completed'])
        self.assertEqual(first['termination_reason'],'api_transport_error')
        self.assertEqual(first['research']['verification_status'],'incomplete')
        self.assertEqual(first['answer']['claims'],[])
        self.assertEqual(len(first['research']['drafts']),1)
        self.assertEqual(first['resume_requirements']['requests'],1)
        second = self.run_task(p,resume=True)
        self.assertTrue(second['completed'],second)
        self.assertEqual(p.counts['research-analysis-v0'],1)
        self.assertEqual(p.counts['research-verify-v0'],2)
        third = self.run_task(p,resume=True)
        self.assertEqual(second,third)

    def test_research_bad_verification_never_repairs_format(self):
        p = ResearchProvider(lambda c,m,n:{'content':'{"claims":[],"issues":[]}'}
                             if c['node']=='research-verify-v0' else None)
        result = self.run_task(p)
        self.assertFalse(result['completed'])
        self.assertEqual(result['termination_reason'],'verification_invalid_output')
        self.assertEqual(p.counts['research-verify-v0'],1)

    def test_research_new_provider_restores_only_unfinished_verification(self):
        first_provider = ResearchProvider(lambda c,m,n:TaskStopped('api_timeout')
                                         if c['node']=='research-verify-v0' else None)
        self.assertFalse(self.run_task(first_provider)['completed'])
        new_provider = ResearchProvider()
        restored = self.run_task(new_provider,resume=True)
        self.assertTrue(restored['completed'],restored)
        self.assertEqual(list(new_provider.counts),['research-verify-v0'])

    def test_research_failed_predecessor_blocks_other_worker(self):
        def script(c,m,n):
            if c['node']=='research-plan':
                return {'content':json.dumps(plan(True))}
            if c['node']=='research-main-worker-1-1':
                return TaskStopped('api_timeout')
        p = ResearchProvider(script)
        result = self.run_task(p)
        self.assertFalse(result['completed'])
        self.assertEqual(result['termination_reason'],'api_timeout')
        self.assertTrue(result['research']['workers']['worker-2']['blocked'])
        self.assertFalse(any('worker-2' in k for k in p.counts))

    def test_research_original_unread_citation_rejected(self):
        def script(c,m,n):
            if c['node']=='research-analysis-v0':
                value = draft()
                value['answer']['claims'][0]['evidence_page_ids']=['history-only-page']
                return {'content':json.dumps(value)}
        result = self.run_task(ResearchProvider(script))
        self.assertFalse(result['completed'])
        self.assertEqual(result['termination_reason'],'invalid_answer_or_evidence')

    def test_research_full_initial_allocation_preserves_four_rework_calls(self):
        issue = dict(requirement_id='r1',task_id='t1',kind='missing_evidence',detail='Check the scope',
                     needs_retrieval=True,query='additional scope')
        def script(c,m,n):
            node = c['node']
            if node=='research-verify-v0':
                return {'content':json.dumps(verification(0,issue))}
            if node.startswith('research-main') and node.endswith('-3'):
                return tool('search_pages',{'query':c['agent_id']+' second question'})
            if node.startswith('research-main') and node.endswith('-4'):
                return tool('read_page',{'page_id':'p1' if c['agent_id']=='worker-1' else 'p2'})
        result = self.run_task(ResearchProvider(script),question='比较这两份报告的方法')
        self.assertTrue(result['completed'],result)
        self.assertEqual(result['actual_model_requests'],16)
        self.assertEqual(result['searches'],5)

    def test_research_shared_page_is_not_new_page_or_new_permission(self):
        def script(c,m,n):
            if c['node']=='research-main-worker-2-2':
                return tool('read_page',{'page_id':'p1'})
            if c['node']=='research-analysis-v0':
                value = draft(); value['answer']['claims'][0]['evidence_page_ids']=['p1']
                return {'content':json.dumps(value)}
        p = ResearchProvider(script)
        result = self.run_task(p)
        self.assertTrue(result['completed'],result)
        self.assertEqual(result['distinct_pages'],1)
        self.assertEqual(result['research']['pages'][0]['received_by'],['worker-1','worker-2'])
        for role in ('worker-1','worker-2'):
            call = next(c for c in p.inputs if c['context']['node']==f'research-main-{role}-3')
            self.assertIn('base64',json.dumps(call['messages']))


class ResearchValidationTests(unittest.TestCase):
    def test_plan_cycle_duplicate_and_exact_merge(self):
        bad = plan(True); bad['tasks'][0]['depends_on']=['t2']
        with self.assertRaises(ValueError): validate_plan(bad)
        one = {'requirements':[{'id':'r1','text':'One requirement'}], 'tasks':[plan()['tasks'][0]]}
        duplicate = dict(one['tasks'][0],id='t2')
        one['tasks'].append(duplicate)
        self.assertEqual(len(validate_plan(one)[1]),1)

    def test_calculation_result_is_not_source_verification(self):
        rows = calculation_checks([dict(operation='percent',a=51,b=66,result=77.27,
                                         evidence_page_ids=['p1'])],{'p1'})
        self.assertTrue(rows[0]['arithmetic_ok'])
        rows = calculation_checks([dict(operation='divide',a=1,b=0,result=0,
                                         evidence_page_ids=['p1'])],{'p1'})
        self.assertFalse(rows[0]['arithmetic_ok'])
        with self.assertRaises(ValueError):
            calculation_checks([dict(operation='eval',a=1,b=1,result=1)],{'p1'})

    def test_config_and_resume_stage_minimum(self):
        config = read_json('configs/agent-multi-demo.json')
        for field,value in [('collaboration_workflow','unknown'),('verification_rework','false')]:
            with self.assertRaises(ValueError): validate_agent_config(dict(config,**{field:value}))
        budget = dict(budget_snapshot_complete=True,requests_used=11,known_output_tokens=10,
                      unknown_output_tokens_reserved=0,searches_used=4,active_seconds_used=1,distinct_page_ids=[])
        result = dict(mode='multi-agent',termination_reason='api_timeout',budget_state=budget,
                      resume_requirements=dict(requests=2,output_tokens=3072))
        self.assertFalse(can_recover(result,{'question':'q'},config))
        result['resume_requirements']['requests']=1
        self.assertTrue(can_recover(result,{'question':'q'},config))
