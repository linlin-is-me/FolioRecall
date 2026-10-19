import copy
import json
from pathlib import Path

import tempfile
import unittest

from scripts.prepare_agent_showcase import (CASES, STYLE, beijing_time, prepare, sanitize_result,
                                            saved_verifications, select_attempts)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def fixture_source(tmp_path):
    source = tmp_path/'source'
    repo = tmp_path/'repository'
    demo = repo/'outputs/releases/v0.1.0-final-candidate/demo-bundle'
    jrc = repo/'outputs/extensions/A/indexes/jrc-original'
    for folder in (demo, jrc):
        folder.mkdir(parents=True)
        (folder/'preview.png').write_bytes(b'fixture preview')
        (folder/'pages.jsonl').write_text(json.dumps({'page_id': 'doc:1', 'doc_id': 'doc',
            'document_name': 'report.pdf', 'page_number': 1, 'preview': 'preview.png',
            'source': 'https://example.org/report'})+'\n', encoding='utf-8')
    (demo/'SOURCES.md').write_text('# Existing attribution\n', encoding='utf-8')
    for name, _, _ in CASES:
        key = {'suite': 'test-M2', 'task_id': name, 'domain': 'demo42', 'language': 'en',
               'condition': 'hybrid-agent', 'repeat': 0}
        answer = {'claims': [{'text': 'Fact <script>alert(1)</script>', 'evidence_page_ids': ['doc:1']}],
                  'insufficient_evidence': False, 'unresolved': []}
        result = {'question': 'Question', 'completed': name != 'cov-11',
                  'termination_reason': 'answered' if name != 'cov-11' else 'verification_invalid_output',
                  'answer': answer, 'timing': {'task_seconds': 12.5}, 'actual_model_requests': 3,
                  'actual_searches': 1, 'api_calls': [{'api_key': 'private-secret', 'ledger_id': 'private-id'}],
                  'evidence': [{'page_id': 'doc:1', 'document_name': 'report.pdf', 'page_number': 1,
                                'text': 'Text', 'preview': '/home/private.png'}],
                  'graph': {'checkpoint': '/home/private.sqlite3'},
                  'progress': [{'event': 'model_request', 'status': 'completed', 'role': 'worker-1',
                                'elapsed_seconds': 3.5, 'raw_response': 'private-secret'}],
                  'research': {'verification_status': 'completed' if name != 'cov-11' else 'failed',
                      'tasks': [{'id': 't1', 'name': 'Evidence', 'depends_on': []}], 'workers': {},
                      'drafts': [{'version': 0, 'answer': answer, 'numbered_claims': []}],
                      'verifications': [], 'pages': []}}
        attempt = {'key': key, 'state': 'finished', 'attempt': 1, 'answer_digest': 'saved-digest',
                   'started_at': '2026-10-15', 'finished_at': '2026-10-15', 'result': result}
        write(source/'main/attempts'/name/'attempt-0001.json', attempt)
        write(source/'scoring'/f'{name}-attempt-0001.json', {'key': key, 'attempt': 1,
              'answer_digest': 'saved-digest', 'reviewer_type': 'assistant', 'human': 'not_checked',
              'diagnosis': 'Existing diagnosis', 'fact_correctness': 0.5, 'api_key': 'private-secret'})
        if name != 'cov-11':
            write(source/'main/attempts'/name/'graph/verification-v0.json', {'claims': [], 'issues': []})
    return source, repo


class AgentShowcaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)

    def test_exact_first_attempt_selection_and_failure_retained(self):
        tmp_path = self.tmp_path
        source, _ = fixture_source(tmp_path)
        selected = select_attempts(source)
        assert list(selected) == sorted(case[0] for case in CASES)
        assert len(selected) == 4
        assert not selected['cov-11'][1]['result']['completed']
        duplicate = copy.deepcopy(selected['cov-01'][1])
        write(source/'main/attempts/another/attempt-0001.json', duplicate)
        with self.assertRaisesRegex(ValueError, '不唯一'):
            select_attempts(source)


    def test_whitelist_does_not_mutate_source_or_timing(self):
        tmp_path = self.tmp_path
        source, _ = fixture_source(tmp_path)
        path, attempt = select_attempts(source)['cov-01']
        result = attempt['result']
        result['research']['tasks'][0]['scope'] = 'debug C:\\Users\\name\\private.txt /home/user/secret'
        result['research']['workers']['worker-1'] = {'done': True, 'messages': [{'content': 'private-secret'}]}
        old = copy.deepcopy(result)
        exported = sanitize_result(result, path)
        assert result == old
        assert exported['timing'] == result['timing']
        assert 'api_calls' not in exported
        text = json.dumps(exported)
        assert 'private-secret' not in text and 'private.sqlite3' not in text
        assert '/home/user/' not in text and 'Users' not in text
        assert exported['research']['verifications'][0]['version'] == 0


    def test_verification_version_comes_from_filename_or_explicit_event(self):
        tmp_path = self.tmp_path
        path = tmp_path/'attempt-0001.json'
        original = {'claims': [{'id': 'v1-c1', 'supported': True, 'reason': 'ok'}], 'issues': []}
        write(tmp_path/'graph/verification-v1.json', original)
        result = {'research': {'verifications': [original]}, 'progress': []}
        assert saved_verifications(path, result)[0]['version'] == 1
        (tmp_path/'graph/verification-v1.json').unlink()
        result['progress'] = [{'event': 'research_verify', 'status': 'completed', 'version': 1, 'issues': []}]
        assert saved_verifications(path, result)[0]['version'] == 1
        result['progress'].append({'event': 'research_verify', 'status': 'completed', 'version': 0, 'issues': []})
        assert 'version' not in saved_verifications(path, result)[0]


    def test_prepare_offline_links_images_and_safe_json(self):
        tmp_path = self.tmp_path
        source, repo = fixture_source(tmp_path)
        output = tmp_path/'showcase'
        originals = {p: p.read_bytes() for p in source.rglob('*.json')}
        summary = prepare(source, output, repo)
        assert summary['case_count'] == 4 and summary['preview_count'] == 1
        assert summary['new_model_calls'] == 0
        assert all(p.read_bytes() == content for p, content in originals.items())
        assert len(list((output/'cases').glob('*.html'))) == 4
        assert len(list((output/'cases').glob('*.json'))) == 4
        content = (output/'cases/cov-01.html').read_text(encoding='utf-8')
        assert 'href="#page-doc%3A1"' in content
        assert 'id="page-doc:1"' in content
        assert 'src="../pages/doc-1.png"' in content
        assert '<script>alert' not in content and '&lt;script&gt;' in content
        assert 'fetch(' not in content and '<script src' not in content
        for path in (output/'cases').glob('*.json'):
            text = path.read_text(encoding='utf-8')
            assert 'private-secret' not in text and '/home/' not in text
        failure = json.loads((output/'cases/cov-11.json').read_text(encoding='utf-8'))
        assert failure['result']['completed'] is False
        assert failure['assistant_review']['human'] == 'not_checked'
        preserved = output/'validation/browser.json'
        write(preserved, {'existing_validation': True})
        (output/'notes.txt').write_text('user notes', encoding='utf-8')
        (output/'cases/cov-01.html').write_text('outdated display', encoding='utf-8')
        prepare(source, output, repo)
        assert '<!doctype html>' in (output/'cases/cov-01.html').read_text(encoding='utf-8')
        assert json.loads(preserved.read_text(encoding='utf-8')) == {'existing_validation': True}
        assert (output/'notes.txt').read_text(encoding='utf-8') == 'user notes'


    def test_review_binding_rejected_before_output(self):
        tmp_path = self.tmp_path
        source, repo = fixture_source(tmp_path)
        path = source/'scoring/cov-01-attempt-0001.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['answer_digest'] = 'wrong-answer'
        write(path, value)
        with self.assertRaisesRegex(ValueError, '不匹配'):
            prepare(source, tmp_path/'showcase', repo)
        assert not (tmp_path/'showcase').exists()


    def test_rework_calculation_and_raw_answer_are_retained(self):
        tmp_path = self.tmp_path
        source, _ = fixture_source(tmp_path)
        path, attempt = select_attempts(source)['confirm-03']
        result = attempt['result']
        result['research']['rework'] = {'triggered': True, 'retrieval': True, 'task_id': 't2'}
        result['research']['drafts'][0]['calculations'] = [{'operation': 'divide', 'a': 51, 'b': 66,
            'result': 0.7727, 'computed_result': 51/66, 'arithmetic_ok': True, 'evidence_page_ids': ['doc:1']}]
        exported = sanitize_result(result, path)
        assert exported['research']['rework'] == result['research']['rework']
        assert exported['research']['drafts'][0]['calculations'] == result['research']['drafts'][0]['calculations']
        assert exported['answer'] == result['answer']

    def test_refresh_validates_source_before_replacing_existing_display(self):
        tmp_path = self.tmp_path
        source, repo = fixture_source(tmp_path)
        output = tmp_path/'showcase'
        prepare(source, output, repo)
        previous = (output/'index.html').read_bytes()
        review = source/'scoring/cov-01-attempt-0001.json'
        value = json.loads(review.read_text(encoding='utf-8'))
        value['answer_digest'] = 'different'
        write(review, value)
        with self.assertRaisesRegex(ValueError, '不匹配'):
            prepare(source, output, repo)
        assert (output/'index.html').read_bytes() == previous
        with self.assertRaisesRegex(ValueError, '历史源记录'):
            prepare(source, source/'showcase', repo)

    def test_history_commit_state_timezone_and_scoped_header_style(self):
        tmp_path = self.tmp_path
        source, repo = fixture_source(tmp_path)
        write(source/'closeout-summary.json', {'runtime_commit': 'original-run-commit'})
        path = source/'main/attempts/cov-01/attempt-0001.json'
        attempt = json.loads(path.read_text(encoding='utf-8'))
        attempt['finished_at'] = 1792055119.9904828
        attempt['result']['progress'].append({'event': 'worker_state', 'role': 'worker-1',
                                              'state': 'blocked', 'status': 'completed'})
        write(path, attempt)
        output = tmp_path/'showcase'
        summary = prepare(source, output, repo)
        assert summary['source_runtime_commit'] == 'original-run-commit'
        assert summary['render_kind'] == 'offline_display_of_historical_records'
        case = json.loads((output/'cases/cov-01.json').read_text(encoding='utf-8'))
        assert case['result']['progress'][-1]['state'] == 'blocked'
        assert case['finished_at'] == 1792055119.9904828
        page = (output/'cases/cov-01.html').read_text(encoding='utf-8')
        assert '2026-10-15 17:05:19 北京时间' in page
        assert beijing_time('2026-10-15T09:05:19+00:00') == '2026-10-15 17:05:19 北京时间'
        assert '.showcase-header{margin:' in STYLE
        assert '}header{' not in STYLE
