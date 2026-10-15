import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.io import read_json
from foliorecall.qa_storage import atomic_json, digest
from scripts import run_multi_agent_experiment as experiment


def tasks():
    rows = []
    for i in range(1,13):
        rows.append({'task_id':f'cov-{i:02d}','domain':'demo42' if i in (1,2,11) else 'jrc',
                     'session':'D1' if i<=2 else 'D2' if i<=4 else f'S{i}',
                     'language':'zh' if i in (2,4,6,8,10,12) else 'en','question':f'question {i}',
                     'reference_facts':[],'source_task_id':f'jrc-{i:02d}'})
    confirmation = [dict(rows[4],task_id=f'confirm-{i:02d}',session='C1' if i<=2 else f'C{i}',
                         language='zh' if i in (2,4,6) else 'en',question=f'confirmation {i}') for i in range(1,7)]
    return rows,confirmation


def record(job,attempt=1,completed=True):
    result = {'question':job['question'],'retrieval_query':'query','answer':{'claims':[],
              'insufficient_evidence':True,'unresolved':[]},'completed':completed,
              'termination_reason':'answered' if completed else 'api_timeout'}
    return {'key':job['key'],'task':job['task'],'attempt':attempt,'state':'finished',
            'answer_digest':digest(result['answer']),'result':result}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = experiment.expand(*tasks())

    def test_counts_order_and_diagnostic(self):
        self.assertEqual(len(self.jobs),54)
        self.assertEqual(len({digest(j['key']) for j in self.jobs}),54)
        self.assertEqual([j['arm'] for j in self.jobs[:12]],['A','A','B','B','C','C','B','B','C','C','A','A'])
        self.assertEqual(sum(j['stage']=='development' for j in self.jobs),36)
        self.assertEqual(sum(j['stage']=='confirmation' for j in self.jobs),18)
        self.assertEqual(sum(j['task']['language']=='zh' and j['session'] not in ('D1','D2','C1') for j in self.jobs),18)
        self.assertEqual(sum(j['task']['language']=='zh' and j['session'] in ('D1','D2','C1') for j in self.jobs),9)
        diagnostic = experiment.expand(*tasks(),True)
        self.assertEqual(len(diagnostic),8)
        self.assertEqual([j['arm'] for j in diagnostic],['serial','C','C','serial','serial','C','C','serial'])
        self.assertTrue(all(j['task']['language']=='en' for j in diagnostic))
        self.assertTrue(all(j['mode']=='multi-agent' for j in diagnostic))

    def test_history_isolated_and_successful_retry_only(self):
        first,follow = self.jobs[:2]
        rows = [record(first,completed=False),record(first,attempt=2),record(self.jobs[2])]
        history,sources = experiment.history_for(rows,self.jobs,follow)
        self.assertEqual(len(history),1)
        self.assertEqual(sources[0]['attempt'],2)
        self.assertEqual(experiment.history_for(rows[:1],self.jobs,follow),([],[]))
        self.assertEqual(experiment.history_for(rows,self.jobs,self.jobs[3])[1][0]['key']['suite'],self.jobs[2]['key']['suite'])

    def test_browser_wrong_question_and_context_do_not_send(self):
        job = next(j for j in self.jobs if j['stage']=='confirmation' and j['arm']=='C')
        callback = experiment.BrowserExecutor(self.root,[{}]*8,{},self.jobs)
        state = {'conversation_id':'cid','turns':[]}
        with patch.object(experiment,'check_identity'),patch.object(experiment,'attempts',return_value=[]), \
             patch.object(experiment,'pending',return_value=[job]),patch.object(experiment,'execute_plan') as execute:
            with self.assertRaisesRegex(ValueError,'固定问题原文'):
                callback('other','multi-agent',[],None,state,{})
            with self.assertRaisesRegex(ValueError,'会话'):
                callback(job['question'],'multi-agent',[record(job)['result']],None,state,{})
            self.assertFalse(execute.called)

    def test_browser_duplicate_returns_terminal_without_model(self):
        job = next(j for j in self.jobs if j['stage']=='confirmation' and j['arm']=='C')
        saved = record(job)
        atomic_json(self.root/'browser-bindings.json',{'cid:1':job['key']})
        callback = experiment.BrowserExecutor(self.root,[{}]*8,{},self.jobs)
        with patch.object(experiment,'check_identity'),patch.object(experiment,'attempts',return_value=[saved]), \
             patch.object(experiment,'execute_plan') as execute:
            result = callback(job['question'],'multi-agent',[],None,{'conversation_id':'cid','turns':[]},{})
            self.assertEqual(result,saved['result'])
            self.assertFalse(execute.called)

    def test_browser_frozen_cache_and_execution_identity(self):
        job = next(j for j in self.jobs if j['stage']=='confirmation' and j['arm']=='C' and j['task']['language']=='zh' and j['session']!='C1')
        job = dict(job,translation={'query':'cached','requests':1})
        units = [{'student':{},'agent':{}} for _ in range(8)]
        callback = experiment.BrowserExecutor(self.root,units,{},self.jobs)
        class Provider:
            context = {}
        provider = Provider()
        resources = dict(provider=provider,model=None,index=None,pages=[],retrieval=None,texts={},query_lock=None)
        def execute_plan(output,identity,jobs,factory,configs,**kwargs):
            factory()(dict(job,execution_id='live-id',graph_output=str(self.root/'graph')),None,1)
        with patch.object(experiment,'check_identity'),patch.object(experiment,'attempts',return_value=[]), \
             patch.object(experiment,'pending',return_value=[job]),patch.object(experiment,'require_reviews'), \
             patch.object(experiment,'execute_plan',side_effect=execute_plan), \
             patch('foliorecall.multi_agent.ask_multi',return_value=record(job)['result']) as ask:
            result = callback(job['question'],'multi-agent',[],lambda e:None,{'conversation_id':'cid','turns':[]},resources)
            self.assertEqual(ask.call_args.kwargs['translation'],job['translation'])
            self.assertEqual(provider.context['execution_id'],'live-id')
            self.assertEqual(provider.context['task_id'],job['key']['task_id'])
            self.assertTrue(result['browser_confirmed'])
            self.assertIsNone(ask.call_args.kwargs['cooldown_seconds'])

    def test_client_overlap(self):
        job = experiment.expand(*tasks(),True)[1]
        folder = self.root/'diagnostic/attempts'/digest(job['key'])/'graph/operations'
        for i,(end,seconds) in enumerate([(13,3),(14,3)]):
            atomic_json(folder/f'{i}.json',{'kind':'request','role':f'worker-{i}',
                'finished_at':end,'outcome':{'record':{'seconds':seconds,'cost_status':'estimated_from_usage'}}})
        result = experiment.overlap(self.root,job)
        self.assertEqual(result['max_client_calls'],2)
        self.assertEqual(result['overlap_seconds'],2)

    def test_gradio_chat_uses_experiment_adapter(self):
        import gradio as gr
        from foliorecall.demo import serve_demo
        config = read_json('configs/agent-multi-demo.json')
        atomic_json(self.root/'agent.json',config)
        query = read_json('configs/student-ml-cpu.json')
        pages = [{'page_id':f'p{i}','doc_id':'d','page_number':i,'document_name':'demo.pdf'} for i in (1,2)]
        apps = []
        result = dict(record(self.jobs[0])['result'],evidence=[],timing={'task_seconds':.01})
        def adapter(q,mode,context,event,state,resources):
            self.assertEqual(q,'fixed question')
            self.assertEqual(context,[])
            self.assertIn('model',resources)
            event({'event':'coordinate','status':'completed','role':'coordinator'})
            return copy.deepcopy(result)
        with patch('foliorecall.generation.Bailian',return_value=object()), \
             patch('foliorecall.search.load_index',return_value=(None,pages)), \
             patch('foliorecall.agent.load_public_corpus',return_value={}), \
             patch('foliorecall.demo.load_search_session',return_value=(object(),None,pages,None)), \
             patch.object(gr.Blocks,'launch',lambda app,**kw:apps.append(app)):
            serve_demo(self.root,{},query,output=self.root/'ui',agent_config_path=self.root/'agent.json',
                       multi_agent_config_path=self.root/'agent.json',public_corpus=self.root/'public.json',
                       _chat_executor=adapter)
            chat = next(fn.fn for fn in apps[0].fns.values() if fn.fn and fn.fn.__name__=='chat')
            updates = list(chat('fixed question','multi-agent',None))
            self.assertEqual(updates[-1][3]['answer'],result['answer'])
            self.assertTrue(list((self.root/'ui').glob('qa-*.json')))

    def test_reviews_require_every_attempt(self):
        row = record(self.jobs[0])
        with self.assertRaisesRegex(ValueError,'缺1'):
            experiment.require_reviews(self.root,[row])
        atomic_json(self.root/'scoring/review.json',dict(key=row['key'],attempt=1))
        experiment.require_reviews(self.root,[row])
        with self.assertRaisesRegex(ValueError,'缺1'):
            experiment.require_reviews(self.root,[row,record(self.jobs[0],2)])


if __name__=='__main__':
    unittest.main()
