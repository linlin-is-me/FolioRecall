import unittest
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from foliorecall.benchmark_data import prepare_benchmark, validate_task
from foliorecall.io import read_json, write_json


class BenchmarkDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = {'name': 'fixture', 'dataset': 'fixture/pages', 'revision': 'fixed',
                     'pages': 2, 'english_queries': 1}
        self.config = {'tasks': [self.task]}

    def fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        image = io.BytesIO()
        Image.new('RGB', (2, 2)).save(image, format='PNG')
        rows = {
            'corpus/test.parquet': [dict(corpus_id=p, doc_id='doc', page_number_in_doc=i,
                image={'bytes': image.getvalue()}, markdown='text') for i, p in enumerate(['a', 'b'])],
            'queries/test.parquet': [{'query_id': 'q', 'query': 'ＡBC  query', 'language': 'en'}],
            'qrels/test.parquet': [{'query_id': 'q', 'corpus_id': 'a', 'score': 2.0}]}
        for name, records in rows.items():
            target = self.root / 'upstream' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(records), target)
        return rows

    def prepare(self, training=None):
        return prepare_benchmark(self.config, self.root / 'prepared', training)[0]

    def fresh(self, training=None):
        files = self.fixture()
        with patch('huggingface_hub.HfApi.list_repo_files', return_value=list(files)), \
             patch('huggingface_hub.try_to_load_from_cache', return_value=None), \
             patch('huggingface_hub.hf_hub_download',
                   side_effect=lambda repo, file, **kw: str(self.root / 'upstream' / file)):
            return self.prepare(training)

    def split(self, query):
        directory = self.root / 'train'
        write_json(directory / 'split.json', {'train': [{'query': query}]})
        return directory

    def test_optional_training_new_parquet_task(self):
        result = self.fresh()
        self.assertEqual(result['training_overlap']['status'], 'not_checked')
        self.assertIsNone(result['normalized_training_query_overlap'])
        self.assertEqual(result['candidates'], 2)

    def test_invalid_explicit_training_fails_before_download(self):
        with patch('huggingface_hub.HfApi.list_repo_files') as download:
            with self.assertRaises(ValueError):
                self.prepare(self.root / 'missing')
            directory = self.root / 'invalid'
            for split in [{}, {'train': [{'query': 3}]}]:
                write_json(directory / 'split.json', split)
                with self.assertRaises(ValueError):
                    self.prepare(directory)
            download.assert_not_called()
        self.assertFalse((self.root / 'prepared').exists())

    def test_cached_overlap_recomputed_without_rewriting_source(self):
        initial = self.fresh(self.split('unrelated'))
        self.assertEqual(initial['normalized_training_query_overlap'], [])
        source = self.root / 'prepared/fixture/source.json'
        original = source.read_bytes()
        with patch('huggingface_hub.HfApi.list_repo_files') as download:
            changed = self.prepare(self.split(' abc\tQUERY '))
            self.assertEqual(changed['training_overlap']['status'], 'checked')
            self.assertEqual(changed['normalized_training_query_overlap'], ['q'])
            omitted = self.prepare()
            self.assertIsNone(omitted['normalized_training_query_overlap'])
            self.assertEqual(omitted['training_overlap']['status'], 'not_checked')
            download.assert_not_called()
        self.assertEqual(source.read_bytes(), original)
        self.assertNotEqual(initial['training_overlap']['record'], changed['training_overlap']['record'])

    def test_unchecked_and_legacy_sources_can_be_checked(self):
        self.fresh()
        source = self.root / 'prepared/fixture/source.json'
        legacy = read_json(source)
        legacy.pop('training_overlap')
        legacy['normalized_training_query_overlap'] = []
        write_json(source, legacy)
        original = source.read_bytes()
        result = self.prepare(self.split('abc query'))
        self.assertEqual(result['normalized_training_query_overlap'], ['q'])
        self.assertEqual(source.read_bytes(), original)

    def test_reused_task_still_rejects_revision_and_candidate_mismatch(self):
        self.fresh()
        self.config['tasks'] = [dict(self.task, revision='changed')]
        with self.assertRaises(ValueError):
            self.prepare()
        self.config['tasks'] = [self.task]
        write_json(self.root / 'prepared/fixture/qrels.json', {'q': {'outside': 1}})
        with self.assertRaises(ValueError):
            self.prepare()

    def test_complete_candidates_and_graded_labels(self):
        pages = [{'page_id': 'a'}, {'page_id': 'negative'}]
        queries = [{'query_id': 'q'}]
        expected = {'pages': 2, 'english_queries': 1}
        validate_task(pages, queries, {'q': {'a': 2, 'negative': 0}}, expected)
        for bad_pages, bad_qrels in [(pages[:1], {'q': {'a': 2}}),
                                      ([pages[0], pages[0]], {'q': {'a': 2}}),
                                      (pages, {'q': {'outside': 2}}),
                                      (pages, {'q': {'a': 0}}),
                                      (pages, {'q': {'a': float('inf')}})]:
            with self.assertRaises(ValueError):
                validate_task(bad_pages, queries, bad_qrels, expected)


if __name__ == '__main__':
    unittest.main()
