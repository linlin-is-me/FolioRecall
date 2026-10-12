import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foliorecall.generation import (NEW_MODEL, MODEL, Bailian, CostLedger, TaskStopped,
                                   pricing_rule, request_reservation, usage_cost)
from foliorecall.io import read_json
from foliorecall.qa_translation import translation_identity, validate_translation
from foliorecall.qa_storage import atomic_json
from scripts.prepare_qa_acceptance import migration_budget, CAP, ordered_keys
from scripts.run_qa_development_segment import check_budget, cache_gate
from scripts.report_qa_acceptance import reconcile


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = read_json('configs/agent-bailian.json')
        self.config.update(ledger=str(self.root/'cost.sqlite3'), phase_budget_cny=CAP)
        self.ledger = CostLedger(self.config['ledger'], CAP)

    def test_price_boundaries_and_old_prices(self):
        self.assertEqual(usage_cost({'prompt_tokens':256000,'completion_tokens':100},NEW_MODEL), .5128)
        self.assertEqual(usage_cost({'prompt_tokens':256001,'completion_tokens':100},NEW_MODEL), 1.538406)
        for prompt, rate in ((32768,1),(32769,1.5),(131073,3)):
            self.assertEqual(usage_cost({'prompt_tokens':prompt,'completion_tokens':100},MODEL),
                             (prompt*rate+100*rate*10)/1e6)
        self.assertEqual(request_reservation(NEW_MODEL,2048),6)
        self.assertEqual(pricing_rule(NEW_MODEL)['maximum_input_tokens'],991808)

    def test_budget_addition_and_frozen_resume(self):
        self.ledger.reserve(3)
        before = migration_budget(self.config, CAP)
        self.assertEqual(before['absolute_cap_cny'],round(3+CAP,6))
        migrated = dict(self.config,phase_budget_cny=before['absolute_cap_cny'])
        self.assertTrue(check_budget(migrated,before,20)['passed'])
        self.ledger.reserve(2)
        after = check_budget(migrated,before,20)
        self.assertEqual(after['remaining_cny'],before['absolute_cap_cny']-5)
        self.assertEqual(before['old_absolute_cap_cny'],CAP)
        with self.assertRaises(ValueError): migration_budget(self.config,60)
        with self.assertRaises(TaskStopped): self.ledger.reserve(100)
        self.ledger.fail(self.ledger.records()[0]['id'],'api_transport_error')
        self.assertEqual(self.ledger.total(),5)

    def test_request_uses_new_snapshot_and_supported_parameters(self):
        import httpx
        config = dict(self.config,model=NEW_MODEL)
        with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'mock','DASHSCOPE_BASE_URL':'https://dashscope.aliyuncs.com/compatible-mode/v1'}):
            provider = Bailian(config)
        sent=[]
        class Client:
            def __init__(self,**kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            async def post(self,*args,**kwargs):
                sent.append(kwargs['json'])
                return httpx.Response(200,json={'model':NEW_MODEL,'usage':{'prompt_tokens':100,'completion_tokens':10},
                    'choices':[{'message':{'role':'assistant','content':'{}'},'finish_reason':'stop'}]})
        with patch('httpx.AsyncClient',Client):
            r=provider.complete([{'role':'user','content':'test'}],None,2048,60)
        self.assertEqual(sent[0]['model'],NEW_MODEL)
        self.assertFalse(sent[0]['enable_thinking'])
        self.assertNotIn('vl_high_resolution_images',sent[0])
        self.assertEqual(r['response_model'],NEW_MODEL)
        self.assertEqual(r['request_model'],NEW_MODEL)
        self.assertEqual(r['cost_cny'],.00028)
        self.assertEqual(r['reservation_cny'],6)

    def test_original_cache_identity_and_budget_are_reused(self):
        cache=dict(identity=translation_identity('问题',self.config),query='question',requests=1,
            output_tokens=20,seconds=2.,cost_cny=.003,
            api_calls=[{'ledger_id':'old','usage':{'prompt_tokens':10,'completion_tokens':20}}])
        config=dict(self.config,model=NEW_MODEL,translation_source_config=self.config)
        before=json.dumps(cache,sort_keys=True)
        self.assertTrue(validate_translation(cache,'问题',config))
        jobs=[{'question':'问题','translation':cache if k['language']=='zh' else None,'unit':0,'key':k} for k in ordered_keys('v2')]
        cache_gate([{'agent':config}],jobs)
        self.assertEqual(len([j for j in jobs if j['key']['language']=='zh']),120)
        self.assertEqual(json.dumps(cache,sort_keys=True),before)
        jobs[3]['translation']=None
        with self.assertRaises(ValueError):cache_gate([{'agent':config}],jobs)
        with self.assertRaises(ValueError):validate_translation(cache,'wrong',config)

    def test_historical_translation_and_new_probe_deduplicate(self):
        def record(execution,key,usage,model=MODEL):
            ledger=CostLedger(self.config['ledger'],100,model)
            lid=ledger.reserve(request_reservation(model,128),dict(execution_id=execution,attempt=1,**key))
            cost=ledger.settle(lid,usage)
            return dict(ledger_id=lid,usage=usage,cost_cny=cost,reservation_cny=request_reservation(model,128))
        old=record('translation',{'task_id':'jrc-01'},{'prompt_tokens':100,'completion_tokens':10})
        frozen=migration_budget(self.config,CAP)
        new=record('probe',{'probe':'image'},{'prompt_tokens':100,'completion_tokens':10},NEW_MODEL)
        atomic_json(self.root/'agent.json',dict(self.config,model=NEW_MODEL,phase_budget_cny=frozen['absolute_cap_cny']))
        atomic_json(self.root/'budget-before.json',frozen)
        atomic_json(self.root/'translations/manifest.json',{'execution_id':'translation','identity_digest':'old'})
        atomic_json(self.root/'translations/translations.json',{'jrc-01':{'api_calls':[old]}})
        atomic_json(self.root/'probe/result.json',{'execution_id':'probe','identity_digest':'new','records':[dict(new,probe='image')]})
        result=reconcile(self.root)
        self.assertEqual(len(result['ledger_ids']),2)
        self.assertAlmostEqual(result['historical_translation_cny'],old['cost_cny'])
        self.assertAlmostEqual(result['round_incremental_occupied_cny'],new['cost_cny'])
