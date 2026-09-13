import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from foliorecall.demo import display_request


class DemoTests(unittest.TestCase):
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
