import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_release import public_result, record
from export_stage4_models import add_release_links
from foliorecall.io import read_json, write_json


class ReleaseAssetsTests(unittest.TestCase):
    def test_public_result_keeps_rankings_and_raw_times_without_task_text(self):
        raw = {'results': [{'query_id': 'q', 'query': 'not for redistribution',
                           'nDCG@10': 0.125, 'ranking': [{'page_id': 'p', 'score': 0.3,
                                                         'preview': '/local/image.png'}]}],
               'requests': [{'query_id': 'q', 'repeat': 2, 'request_seconds': 0.004}],
               'queries': 1}
        result = public_result(raw)
        self.assertNotIn('query', result['results'][0])
        self.assertNotIn('preview', result['results'][0]['ranking'][0])
        self.assertEqual(result['requests'], raw['requests'])
        self.assertEqual(result['results'][0]['nDCG@10'], 0.125)
        self.assertEqual(result['results'][0]['ranking'][0], {'page_id': 'p', 'score': 0.3})
        self.assertIn('query', raw['results'][0])
        self.assertEqual(public_result({'queries': ['private training text'], 'parameters_equal': 104}),
                         {'queries': 1, 'parameters_equal': 104})

    def test_actual_config_verbatim_and_model_link_update_preserve_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'encoding.json'
            write_json(config, {'adapter': '.', 'revision': 'fixed'})
            original = config.read_bytes()
            sources = []
            record(config, root / 'evidence/config.json', sources, verbatim=True)
            self.assertEqual((root / 'evidence/config.json').read_bytes(), original)
            write_json(root / 'source.json', {'checkpoint': 'old/checkpoint-750'})
            (root / 'README.md').write_text('Model card\n')
            add_release_links(root, 'v0.1.0rc2')
            self.assertEqual(config.read_bytes(), original)
            source = read_json(root / 'source.json')
            self.assertEqual(source['checkpoint'], 'old/checkpoint-750')
            self.assertTrue(source['evidence_url'].endswith('/v0.1.0rc2/experiment-evidence.zip'))


if __name__ == '__main__':
    unittest.main()
