import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from foliorecall.agent import ask
from foliorecall.chat import new_session, recent_context, progress_html, stream_chat
from foliorecall.generation import TaskStopped
from foliorecall.qa_storage import atomic_json as write_json
from tests.test_agent import FakeProvider, message


def result(question='q', completed=True):
    return {'question': question, 'retrieval_query': question, 'completed': completed,
            'termination_reason': 'answered' if completed else 'api_transport_error',
            'answer': {'claims': [], 'insufficient_evidence': True, 'unresolved': ['Missing evidence']},
            'evidence': [], 'timing': {'task_seconds': .01}}


class ChatTests(unittest.TestCase):
    def run_ask(self, responses, history=None, events=None):
        config = {'provider':'bailian', 'region':'cn-beijing', 'model':'qwen3.7-plus-2026-05-26',
                  'temperature':0,'enable_thinking':False,'max_searches':3,'max_pages':6,
                  'max_requests':8,'max_output_tokens':8000,'request_timeout':60,
                  'task_timeout':180,'max_pixels':1310720,'phase_budget_cny':20,'ledger':'unused'}
        pages = [{'page_id':'p1','doc_id':'d','page_number':1,'document_name':'one.pdf','preview':'unused'}]
        provider = FakeProvider(responses)
        with patch('foliorecall.agent.retrieve', return_value=(pages,'',{'request_seconds':.01})), \
             patch('foliorecall.agent.read_evidence', return_value=({'type':'image_url','image_url':{'url':'dummy'}}, {'seconds':.01})):
            answer = ask('它的趋势呢？' if history else 'What is the trend?', 'rag', None, {}, None,
                         pages, config, provider, conversation=history,
                         on_event=events.append if events is not None else None)
        return answer, provider

    def test_followup_translates_and_resolves_in_one_budgeted_request(self):
        history = [result(str(i)) for i in range(5)] + [result('failed', False)]
        events = []
        value, provider = self.run_ask([{'content':'{"query":"standalone English trend"}'},
                                       message([{'text':'趋势','evidence_page_ids':['p1']}])], history, events)
        self.assertTrue(value['completed'])
        self.assertEqual(value['model_requests'], 2)
        self.assertEqual(value['context_turns'], 3)
        self.assertEqual(value['translation_status'], 'contextual_rewrite')
        context = json.loads(provider.inputs[0][0][1]['content'])
        self.assertEqual([t['question'] for t in context['previous_turns']], ['2','3','4'])
        self.assertEqual(value['api_calls'][0]['phase'], 'query_rewrite')
        self.assertEqual(events[-1]['event'], 'terminal')
        self.assertEqual(events[-1]['status'], 'completed')
        self.assertLess(next(i for i,e in enumerate(events) if e['event']=='model_request' and e['status']=='started'),
                        next(i for i,e in enumerate(events) if e['event']=='model_request' and e['status']=='completed'))

    def test_historical_citation_does_not_grant_current_permission(self):
        old = result(); old['answer']['claims'] = [{'text':'old','evidence_page_ids':['old-page']}]
        value, _ = self.run_ask([{'content':'{"query":"standalone"}'},
                                message([{'text':'x','evidence_page_ids':['old-page']}])], [old])
        self.assertFalse(value['completed'])
        self.assertEqual(value['failure']['output_diagnostic']['code'], 'unread_citation')

    def test_empty_history_needs_no_rewrite_and_error_has_terminal_event(self):
        events = []
        value, provider = self.run_ask([TaskStopped('api_transport_error')], [], events)
        self.assertEqual(len(provider.inputs), 1)
        self.assertEqual(events[-1]['status'], 'failed')
        self.assertNotIn('query_rewrite', [e['event'] for e in events])

    def test_sessions_reset_and_failed_context_is_excluded(self):
        a, b = new_session(), new_session()
        a['turns'] = [result(str(i)) for i in range(4)] + [result('failure', False)]
        self.assertNotEqual(a['conversation_id'], b['conversation_id'])
        self.assertEqual([v['question'] for v in recent_context(a)], ['1','2','3'])
        self.assertEqual(recent_context(b), [])

    def test_consumer_disconnect_keeps_one_worker_and_terminal_record(self):
        proceed, saved = threading.Event(), threading.Event()
        calls = []
        def execute(q, mode, history, event):
            calls.append(q)
            event({'event':'model_request','status':'started'})
            proceed.wait(2)
            return result(q)
        def save(path, value):
            write_json(path, value)
            saved.set()
        with tempfile.TemporaryDirectory() as folder, patch('foliorecall.chat.write_json', side_effect=save):
            stream = stream_chat('q', 'rag', None, execute, [], threading.Lock(), folder)
            self.assertTrue(next(stream)[-1])
            stream.close()
            proceed.set()
            self.assertTrue(saved.wait(3))
            record = json.loads(next(Path(folder).glob('qa-*.json')).read_text())
            self.assertEqual(record['turn_index'], 1)
            self.assertTrue(record['completed'])
            self.assertEqual(calls, ['q'])

    def test_stream_updates_before_result_and_does_not_mutate_input_session(self):
        state = new_session()
        def execute(q, mode, history, event):
            event({'event':'search_pages','status':'completed','query':q})
            return result(q)
        updates = list(stream_chat('question', 'agent', state, execute, [], threading.Lock()))
        self.assertTrue(any('搜索页面' in u[2] and u[-1] for u in updates))
        self.assertFalse(updates[-1][-1])
        self.assertEqual(len(updates[-1][4]['turns']), 1)
        self.assertEqual(state['turns'], [])

    def test_progress_escapes_document_and_query_text(self):
        filename = 'employment_and_social_developments_in_europe-KE0125067ENN.pdf'
        event = {'event':'read_page','status':'completed','page_id':'p1','page_number':3,'document_name':filename}
        value = progress_html([{'event':'search_pages','status':'completed','query':'<script>x</script>'}, event])
        self.assertNotIn('<script>', value)
        self.assertIn('&lt;script&gt;', value)
        self.assertIn(f'title="{filename}"', value)
        self.assertIn('Employment and social developments in europe · 物理第3页', value)
        self.assertEqual(event['document_name'], filename)


if __name__ == '__main__':
    unittest.main()
