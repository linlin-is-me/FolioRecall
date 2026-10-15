from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.qa_storage import atomic_json, digest
from scripts import multi_agent_research_experiment as research
from scripts import run_multi_agent_experiment as legacy
from tests.test_multi_agent_experiment import tasks, record


class ResearchExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dev,confirm = tasks()
        for t in dev+confirm:
            t['category']='example'
        self.jobs = research.expand(dev,confirm)

    def test_exact_order_counts_and_cache_followup_strata(self):
        jobs = self.jobs
        self.assertEqual(len(jobs),60)
        self.assertEqual(len({digest(j['key']) for j in jobs}),60)
        self.assertEqual(sum(j['stratum']=='main' for j in jobs),54)
        self.assertEqual(sum(j['stage']=='development' for j in jobs),40)
        self.assertEqual(sum(j['stratum']=='ablation' for j in jobs),6)
        self.assertEqual([j['arm'] for j in jobs[6:14]],['M1','M1','N','N','M2','M2','S','S'])
        zh = [j for j in jobs if j['key']['language']=='zh']
        self.assertEqual(sum(j['session'] in ('D1','D2','C1') for j in zh),11)
        self.assertEqual(sum(j['session'] not in ('D1','D2','C1') for j in zh),19)
        self.assertTrue(all(j['mode']=='agent' if j['arm']=='S' else j['mode']=='multi-agent' for j in jobs))

    def test_ablation_successful_history_is_own_not_full_workflow(self):
        n = [j for j in self.jobs if j['arm']=='N' and j['session']=='D2']
        m = next(j for j in self.jobs if j['arm']=='M2' and j['session']=='D2')
        rows = [record(n[0],completed=False),record(m)]
        self.assertEqual(legacy.history_for(rows,self.jobs,n[1]),([],[]))
        rows.append(record(n[0],attempt=2))
        history,sources = legacy.history_for(rows,self.jobs,n[1])
        self.assertEqual(len(history),1)
        self.assertEqual(sources[0]['attempt'],2)
        self.assertEqual(sources[0]['key'],n[0]['key'])

    def test_first_segment_and_browser_boundary_are_not_ablation_stages(self):
        units = [{'agent':{}}]*8
        with patch.object(research,'load_plan',return_value=(units,{},self.jobs)), \
             patch.object(legacy,'check_identity'),patch.object(research,'budget',return_value={}):
            status = research.run(self.root,dry_run=True)
            self.assertEqual(status['automatic_tasks'],14)
            self.assertFalse(status['browser_required'])
        browser = next(j for j in self.jobs if j['stage']=='confirmation' and j['arm']=='M2')
        with patch.object(research,'load_plan',return_value=(units,{},[browser])), \
             patch.object(legacy,'check_identity'),patch.object(research,'budget',return_value={}):
            status = research.run(self.root,phase='confirmation',dry_run=True)
            self.assertEqual(status['automatic_tasks'],0)
            self.assertTrue(status['browser_required'])

    def test_partial_report_pairs_and_deduplicates_translation_sources(self):
        atomic_json(self.root/'plan.json',{'main':self.jobs})
        atomic_json(self.root/'selection.json',{'experiment':'research-v1','scope':'test'})
        for arm in research.ARMS:
            atomic_json(self.root/f'agent-{arm}.json',{})
        source = dict(api_calls=[{'ledger_id':'original','cost_cny':.01}])
        atomic_json(self.root/'translations.json',{'a':source,'b':source})
        for arm in ('M2','N'):
            job = next(j for j in self.jobs if j['arm']==arm and j['key']['task_id']=='cov-05')
            row = record(job)
            row['result'].update(timing={'task_seconds':1},model_requests=1,actual_model_requests=1,actual_searches=1,
                                 estimated_cost_cny=.01,api_calls=[])
            atomic_json(self.root/'main/attempts'/digest(job['key'])/'attempt-0001.json',row)
        with patch.object(research,'budget',return_value={}):
            report = research.report(self.root)
        self.assertEqual(report['historical_translation_cost_cny'],.01)
        self.assertEqual(len(report['pairs']['M2-vs-N']['rows']),1)
        self.assertNotIn('translation_allocated_cost_cny',report['summary'])
        self.assertEqual(report['status'],'partial')
