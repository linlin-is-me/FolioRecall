"""Real CPU retrieval and Gradio callbacks, with a strictly offline model double."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import time
from unittest.mock import patch

from foliorecall.agent import load_public_corpus
from foliorecall.io import read_json, write_json
from foliorecall.query import load_search_session
from foliorecall.retrieval import read_retrieval_config


class OfflineProvider:
    """No network, credentials, paid ledger or answer-quality claims."""
    def __init__(self):
        self.context, self.calls = {}, []

    def complete(self, messages, tools, max_tokens, timeout):
        context = dict(self.context)
        self.calls.append(context)
        node = context['node']
        if node.startswith('query-'):
            value = {'query':'Evidence pages and analysis periods for population and employment in Europe'}
        elif node == 'coordinate-plan':
            value = {'tasks':[{'question':'Population projections','query':'working age population baseline migration projections'},
                              {'question':'Employment observations','query':'employment EU recent employment growth'}]}
        elif node == 'coordinate-merge':
            value = {'gap':None}
        elif node == 'coordinate-final':
            allowed = json.loads(messages[-1]['content'].split('allowed_evidence_page_ids=')[-1])
            chinese = any('Write every claims[].text and unresolved[] explanation in Chinese' in str(m.get('content')) for m in messages)
            question = messages[1]['content']
            value = {'claims':([] if 'Saturn' in question else [
                        {'text':'模拟生成器已接收所引页面，仅验证执行流程。' if chinese else
                         'The offline generator received the cited pages; this validates application flow only.',
                         'evidence_page_ids':allowed}]),
                     'insufficient_evidence':True,
                     'unresolved':['模拟输出，不评价回答质量。' if chinese else 'Simulated output; answer quality is not evaluated.']}
        else:
            tools_seen = [json.loads(m['content']) for m in messages if m.get('role')=='tool']
            if not tools_seen:
                task = json.loads(messages[1]['content'])
                return self._tool('search_pages',{'query':task['query']})
            if isinstance(tools_seen[-1],list):
                return self._tool('read_page',{'page_id':tools_seen[-1][0]['page_id']})
            value = {'findings':'Offline tool path completed.','unresolved':['Quality was not evaluated.']}
        return {'message':{'content':json.dumps(value,ensure_ascii=False)},
                'usage':{'prompt_tokens':0,'completion_tokens':0},'cost_cny':0.,'simulation':True}

    @staticmethod
    def _tool(name,params):
        return {'message':{'content':None,'tool_calls':[{'id':'offline-tool','type':'function',
            'function':{'name':name,'arguments':json.dumps(params)}}]},
            'usage':{'prompt_tokens':0,'completion_tokens':0},'cost_cny':0.,'simulation':True}


class OfflineResearchProvider(OfflineProvider):
    def complete(self,messages,tools,max_tokens,timeout):
        node = self.context['node']
        if node=='research-query':
            value = {'query':'Population and employment evidence pages and analysis periods'}
        elif node=='research-plan':
            value = {'requirements':[{'id':'r1','text':'Population evidence'},{'id':'r2','text':'Employment evidence'}],
                'tasks':[{'id':'t1','name':'人口预测取证','question':'Find population evidence',
                          'query':'working age population baseline migration projections','scope':'population',
                          'requirement_ids':['r1'],'depends_on':[]},
                         {'id':'t2','name':'就业观测取证','question':'Find employment evidence',
                          'query':'employment EU recent employment growth','scope':'employment',
                          'requirement_ids':['r2'],'depends_on':[]}]}
        elif node.startswith('research-analysis-'):
            allowed = json.loads(messages[0]['content'].split('allowed_evidence_page_ids=')[-1])
            payload = json.loads(messages[1]['content'])
            chinese = 'Write every claims[].text and unresolved[] explanation in Chinese' in messages[0]['content']
            value = {'answer':{'claims':[] if 'Saturn' in payload['question'] else [{
                'text':'模拟回答仅验证实际原页传递。' if chinese else 'Simulated answer validates actual page delivery only.',
                'evidence_page_ids':allowed}], 'insufficient_evidence':True,
                'unresolved':['模拟生成，不评价质量。' if chinese else 'Simulated generation; quality not assessed.']},
                'coverage':[{'requirement_id':'r1','covered':False},{'requirement_id':'r2','covered':False}],
                'calculations':[]}
        elif node.startswith('research-verify-'):
            payload = json.loads(messages[1]['content'])
            value = {'claims':[{'id':c['id'],'supported':True,'reason':'Simulation only.'}
                               for c in payload['draft']['numbered_claims']],'issues':[]}
        else:
            return super().complete(messages,tools,max_tokens,timeout)
        self.calls.append(dict(self.context))
        return {'message':{'content':json.dumps(value,ensure_ascii=False)},
                'usage':{'prompt_tokens':0,'completion_tokens':0},'cost_cny':0.,'simulation':True}


def ledger_snapshot(path):
    if not path:
        return None
    with sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True) as db:
        count, occupied = db.execute('SELECT COUNT(*),COALESCE(SUM(cny),0) FROM calls').fetchone()
        return {'calls':count,'occupied_cny':occupied}


def validate(bundle,query_config,output,ledger=None,workflow='basic'):
    import gradio as gr
    from foliorecall.demo import serve_demo, display_request
    root, bundle = Path(output).resolve(), Path(bundle).resolve()
    if root.exists():
        raise ValueError('离线验证使用新的输出目录')
    root.mkdir(parents=True)
    before = ledger_snapshot(ledger)
    teacher = read_json(bundle/'configs/baseline.json')
    query = read_json(query_config)
    retrieval_config = read_retrieval_config(bundle/'hybrid.json')
    started = time.monotonic()
    model,index,pages,retrieval = load_search_session(bundle,teacher,query,retrieval_config=retrieval_config)
    loading = time.monotonic()-started
    texts = load_public_corpus(bundle/'public-corpus.json',pages)
    multi = read_json('configs/agent-multi-demo.json')
    if workflow=='research':
        multi.update(collaboration_workflow='research',max_requests=16,max_searches=6,max_pages=10,
                     max_output_tokens=16000,task_timeout=300)
    multi['ledger'] = str(root/'simulated-costs.sqlite3')
    original = read_json('configs/agent-demo.json')
    original['ledger'] = multi['ledger']
    write_json(root/'multi.json',multi)
    write_json(root/'agent.json',original)
    providers, apps = [], []
    def provider_factory(config):
        provider = OfflineResearchProvider() if workflow=='research' else OfflineProvider()
        providers.append(provider)
        return provider
    def launch(app,*args,**kwargs):
        apps.append(app)
    # Even an accidental path to the real client's complete() cannot send a request.
    with patch('foliorecall.generation.Bailian.complete',side_effect=AssertionError('Real API disabled')), \
         patch('foliorecall.generation.Bailian',side_effect=provider_factory), \
         patch('foliorecall.demo.load_search_session',return_value=(model,index,pages,retrieval)) as resident, \
         patch.object(gr.Blocks,'launch',launch):
        serve_demo(bundle,teacher,query,output=root/'browser',retrieval_config=retrieval_config,
                   agent_config_path=root/'agent.json',public_corpus=bundle/'public-corpus.json',
                   multi_agent_config_path=root/'multi.json')
        assert resident.call_count == 1
        app = apps[0]
        functions = {fn.fn.__name__:fn.fn for fn in app.fns.values() if fn.fn}
        config = app.get_config_file()
        radio = next(c for c in config['components'] if c['type']=='radio')
        assert any('multi-agent' in str(choice) for choice in radio['props']['choices'])
        state, turns = None, []
        for n, question in enumerate([
                'What do the demographic and employment reports say about changes in the EU?',
                '分别列出证据页，并说明分析时间范围。',
                'What is the orbital period of Saturn?'],1):
            if n == 3:
                state = functions['reset_session']()[4]
            updates = list(functions['chat'](question,'multi-agent',state))
            result, state = updates[-1][3], updates[-1][4]
            assert result['completed'], result
            assert result['context_turns'] == (1 if n==2 else 0)
            assert result['mode']=='multi-agent'
            assert ('核验Agent' if workflow=='research' else '检索Agent') in updates[-1][2]
            task = root/'browser'/'collaboration'/f'{state["conversation_id"]}-turn{len(state["turns"])}'
            saved = read_json(task/'result.json')
            assert saved['answer']==result['answer']
            assert saved['conversation_id']==state['conversation_id']
            turns.append({'question':question,'completed':result['completed'],'context_turns':result['context_turns'],
                          'conversation_id':state['conversation_id'],'turn_index':result['turn_index'],
                          'simulated_requests':result['actual_model_requests'],'searches':result['actual_searches'],
                          'evidence_page_ids':[p['page_id'] for p in result['evidence']],
                          'progress_events':len(result['progress']),'record':str(task/'result.json')})
    # Free retrieval uses the same resident model and the original display path.
    free = []
    for mode, state in [('visual',None),('hybrid',retrieval)]:
        _,table,download,_ = display_request(model,query,index,pages,
            'baseline no migration working-age population scenarios',5,root,retrieval=state)
        assert table and Path(download).exists()
        free.append({'mode':mode,'top_page':table[0][2],'json':str(download)})
    after = ledger_snapshot(ledger)
    assert before == after, '共享费用账本在离线验证期间发生变化'
    result = {'status':'passed','source':'working tree; recorded before the implementation commit',
              'scope':'offline functional validation; real CPU ONNX retrieval, simulated generation, no quality/performance experiment',
              'workflow':workflow,'pages':len(pages),'model_loads':1,'model_loading_seconds':loading,'backend':query.get('backend'),
              'turns':turns,'free_retrieval':free,'gradio_callbacks':True,'actual_browser_clicks':False,
              'real_generation_requests':0,'real_generation_cost_cny':0,'ledger_before':before,'ledger_after':after,
              'dependencies':{name:importlib.metadata.version(name) for name in (
                  'torch','sentence-transformers','transformers','onnxruntime','httpx','gradio','langgraph','langgraph-checkpoint-sqlite')}}
    write_json(root/'summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--query-config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--ledger',help='可选，只读核对已有共享账本')
    parser.add_argument('--workflow',choices=('basic','research'),default='basic')
    args = parser.parse_args()
    print(json.dumps(validate(args.bundle,args.query_config,args.output,args.ledger,args.workflow),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
