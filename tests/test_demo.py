import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from foliorecall.demo import display_request, qa_display


class DemoTests(unittest.TestCase):
    def test_all_qa_examples_fill_only_question_and_keep_selected_mode(self):
        import gradio as gr
        from foliorecall.demo import serve_demo
        repo = Path(__file__).resolve().parents[1]
        apps = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = [{'question': 'First', 'mode': 'agent'},
                        {'question': 'Follow up', 'mode': 'rag', 'followup': True},
                        {'question': 'New topic', 'mode': 'agent', 'new_conversation': True}]
            (root/'qa-examples.json').write_text(json.dumps(examples))
            pages = [{'page_id':f'p{i}', 'doc_id':'d', 'document_name':'doc.pdf', 'page_number':i}
                     for i in (1,2)]
            with patch('foliorecall.demo.load_search_session',return_value=(None,None,pages,None)), \
                 patch('foliorecall.search.load_index',return_value=(None,pages)), \
                 patch('foliorecall.agent.load_public_corpus',return_value={}), \
                 patch('foliorecall.generation.Bailian') as provider, \
                 patch('foliorecall.demo.process_memory',return_value={}), \
                 patch.object(gr.Blocks,'launch',lambda app,**kwargs:apps.append(app)):
                serve_demo(root,{}, {'model_id':'test','device':'cpu','dtype':'float32'},
                    output=root/'records', agent_config_path=repo/'configs/agent-demo.json',
                    multi_agent_config_path=repo/'configs/agent-research-demo.json',public_corpus='unused')
                self.assertFalse(provider.return_value.complete.called)
            config = apps[0].get_config_file()
            question = next(c['id'] for c in config['components'] if c['type']=='textbox'
                            and c['props'].get('label')=='英文或中文问题')
            radio = next(c for c in config['components'] if c['type']=='radio')
            self.assertEqual(radio['props']['value'],'multi-agent')
            datasets = {c['id'] for c in config['components'] if c['type']=='dataset'}
            events = [d for d in config['dependencies'] if any(t[0] in datasets for t in d['targets'])]
            self.assertEqual(len(events),3)
            self.assertTrue(all(d['outputs']==[question] for d in events))

    def test_research_verification_distinguishes_reworked_history(self):
        result = {'completed': True,
                  'answer': {'claims': [], 'insufficient_evidence': True, 'unresolved': []},
                  'evidence': [], 'timing': {'task_seconds': .1},
                  'research': {'tasks': [], 'verification_status': 'passed',
                    'verifications': [{'issues': [{'detail': '<missing policy>'}]}, {'issues': []}],
                    'rework': {'triggered': True}}}
        rendered = qa_display(result, [])
        self.assertIn('首次核验（返工前）', rendered)
        self.assertLess(rendered.index('&lt;missing policy&gt;'), rendered.index('末次核验'))
        self.assertLess(rendered.index('末次核验'), rendered.index('未报告问题。'))
        self.assertIn('&lt;missing policy&gt;', rendered)
        self.assertIn('未经独立人工确认', rendered)

    def test_failed_empty_answer_and_missing_preview_remain_visible(self):
        result = {'completed': False, 'termination_reason': 'api_transport_error',
                  'answer': {'claims': [], 'insufficient_evidence': True, 'unresolved': []},
                  'evidence': [{'page_id': 'p1', 'page_number': 2}],
                  'timing': {'task_seconds': .1}}
        with patch('foliorecall.agent.read_evidence', side_effect=OSError('missing')):
            rendered = qa_display(result, [{'page_id': 'p1', 'doc_id': 'd'}])
        self.assertIn('任务未完成', rendered)
        self.assertIn('api_transport_error', rendered)
        self.assertIn('id="qa-page-p1"', rendered)
        self.assertIn('证据预览暂不可读', rendered)
        self.assertNotIn('回答已生成', rendered)

    def test_qa_citation_fragment_resolves_unicode_and_punctuation(self):
        from html.parser import HTMLParser
        from urllib.parse import unquote
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids, self.hrefs = [], []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs: self.ids.append(attrs['id'])
                if 'href' in attrs: self.hrefs.append(attrs['href'])
        pid = 'JRC141952:页&1'
        filename = 'employment_and_social_developments_in_europe-KE0125067ENN.pdf'
        result = {'answer': {'claims': [{'text': '<script>unsafe</script>',
                   'evidence_page_ids': [pid]}], 'unresolved': []},
                  'evidence': [{'page_id': pid, 'page_number': 1}],
                  'timing': {'task_seconds': .1}}
        pages = [{'page_id': pid, 'doc_id': 'document', 'document_name': filename, 'page_number': 1}]
        with patch('foliorecall.agent.read_evidence', return_value=(
                {'image_url': {'url': 'data:image/png;base64,AA=='}}, {})):
            rendered = qa_display(result, pages)
        parsed = Links()
        parsed.feed(rendered)
        self.assertEqual(unquote(parsed.hrefs[0][1:]), parsed.ids[0])
        self.assertNotIn('<script>', rendered)
        self.assertIn('Employment and social developments in europe · 第 1 页</a>', rendered)
        self.assertIn(f'title="{filename}"', rendered)
        self.assertIn(f'<figcaption>{filename} · 物理第 1 页</figcaption>', rendered)
        self.assertEqual(pages[0]['document_name'], filename)
        self.assertGreaterEqual(result['timing']['task_and_display_seconds'], .1)

    def test_display_reuses_ranking_and_reports_missing_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            Image.new('RGB', (20, 20)).save(folder / 'page.png')
            rows = [{'page_id': 'a', 'doc_id': 'd', 'document_name': 'doc.pdf', 'page_number': 4,
                     'score': .8, 'source': 'https://example.org/doc.pdf', 'preview': str(folder / 'page.png')},
                    {'page_id': 'b', 'doc_id': 'd', 'page_number': 5, 'score': .7,
                     'source': 'same', 'preview': str(folder / 'missing.png')}]
            timing = {'encode_seconds': .01, 'query_seconds': .02, 'request_seconds': .03}
            with patch('foliorecall.demo.retrieve', return_value=(rows, json.dumps(rows), timing)) as retrieve:
                gallery, table, download, message = display_request(None, {}, None, [], 'query', 2, folder)
            retrieve.assert_called_once()
            self.assertEqual(len(gallery), 1)
            self.assertEqual(table[0][2], 4)
            self.assertEqual(table[0][1], 'doc.pdf')
            self.assertIn('Doc · 物理第 4 页', gallery[0][1])
            self.assertEqual(json.loads(Path(download).read_text()), rows)
            self.assertIn('预览不可读', message)


if __name__ == '__main__':
    unittest.main()
