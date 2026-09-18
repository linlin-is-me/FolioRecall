import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from foliorecall.demo import display_request, qa_display


class DemoTests(unittest.TestCase):
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
        result = {'answer': {'claims': [{'text': '<script>unsafe</script>',
                   'evidence_page_ids': [pid]}], 'unresolved': []},
                  'evidence': [{'page_id': pid, 'page_number': 1}],
                  'timing': {'task_seconds': .1}}
        pages = [{'page_id': pid, 'doc_id': 'document', 'page_number': 1}]
        with patch('foliorecall.agent.read_evidence', return_value=(
                {'image_url': {'url': 'data:image/png;base64,AA=='}}, {})):
            rendered = qa_display(result, pages)
        parsed = Links()
        parsed.feed(rendered)
        self.assertEqual(unquote(parsed.hrefs[0][1:]), parsed.ids[0])
        self.assertNotIn('<script>', rendered)
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
            self.assertEqual(json.loads(Path(download).read_text()), rows)
            self.assertIn('预览不可读', message)


if __name__ == '__main__':
    unittest.main()
