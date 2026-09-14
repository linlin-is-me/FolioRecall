import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.__main__ import main
from foliorecall.benchmark_data import load_dataset_provenance
from foliorecall.io import read_json, write_json, write_rows


class DatasetProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.task = {'name': 'fixture', 'dataset': 'fixture/pages', 'revision': 'fixed',
                     'pages': 2, 'english_queries': 1}
        self.source = {'task': self.task, 'scope': 'fixed fixture',
                       'normalized_training_query_overlap': []}
        self.pages = [{'page_id': 'a'}, {'page_id': 'b'}]
        write_json(self.data / 'source.json', self.source)
        self.original = (self.data / 'source.json').read_bytes()
        write_rows(self.data / 'pages.jsonl', self.pages)
        write_rows(self.data / 'texts.jsonl', [{'page_id': 'a', 'text': 'one two'},
                                               {'page_id': 'b', 'text': 'three'}])
        write_rows(self.data / 'queries.jsonl', [{'query_id': 'q', 'query': 'one'}])
        write_json(self.data / 'qrels.json', {'q': {'a': 2}})

    def audit(self, name, checked=True, overlap=None):
        run = self.data / 'runs' / name
        report = {'status': 'checked' if checked else 'not_checked',
                  'training_file': '/fixture/train/split.json' if checked else None,
                  'training_query_count': 1 if checked else None,
                  'normalized_training_query_overlap': ([] if overlap is None else overlap) if checked else None}
        write_json(run / 'run.json', {'config': {'task': self.task, 'training_file': report['training_file']}})
        write_json(run / 'training-overlap.json', report)
        return run / 'training-overlap.json'

    def test_historical_source_without_completed_audit_is_unchanged(self):
        self.assertEqual(load_dataset_provenance(self.data), self.source)
        self.audit('notes')  # Non-run folders do not participate in selection.
        run = self.data / 'runs/20'
        write_json(run / 'run.json', {'config': {'task': self.task}})
        write_json(run / 'training-overlap.json.tmp', {'incomplete': True})
        self.assertEqual(load_dataset_provenance(self.data), self.source)
        self.assertEqual((self.data / 'source.json').read_bytes(), self.original)

    def test_numeric_latest_checked_empty_and_unchecked_audits(self):
        self.audit('2', overlap=['q'])
        newest = self.audit('10')
        result = load_dataset_provenance(self.data)
        self.assertEqual(result['training_overlap']['status'], 'checked')
        self.assertEqual(result['normalized_training_query_overlap'], [])
        self.assertEqual(result['training_overlap']['record'], str(newest.resolve()))
        newest = self.audit('11', checked=False)
        result = load_dataset_provenance(self.data)
        self.assertEqual(result['training_overlap']['status'], 'not_checked')
        self.assertIsNone(result['normalized_training_query_overlap'])
        self.assertEqual(result['training_overlap']['record'], str(newest.resolve()))
        self.assertEqual((self.data / 'source.json').read_bytes(), self.original)

    def test_latest_malformed_audit_never_falls_back(self):
        self.audit('1')
        newest = self.audit('2')
        for report in [[], {}, {'status': 'checked'},
                       dict(read_json(newest), normalized_training_query_overlap=['outside']),
                       dict(read_json(newest), normalized_training_query_overlap=['q', 'q']),
                       dict(read_json(newest), training_query_count=True),
                       dict(read_json(newest), training_file='/other/split.json'),
                       dict(read_json(newest), status='not_checked')]:
            with self.subTest(report=report):
                write_json(newest, report)
                with self.assertRaises(ValueError):
                    load_dataset_provenance(self.data)
        newest.write_text('{', encoding='utf-8')
        with self.assertRaises(ValueError):
            load_dataset_provenance(self.data)

    def test_audit_task_and_candidate_bindings_are_checked(self):
        newest = self.audit('10')
        write_json(newest.parent / 'run.json', {'config': {'task': dict(self.task, revision='other')}})
        with self.assertRaisesRegex(ValueError, '任务配置'):
            load_dataset_provenance(self.data)
        self.audit('10')
        write_rows(self.data / 'pages.jsonl', self.pages[:1])
        with self.assertRaisesRegex(ValueError, '任务数量'):
            load_dataset_provenance(self.data)
        write_rows(self.data / 'pages.jsonl', self.pages)
        write_json(self.data / 'qrels.json', {'q': {'outside': 1}})
        with self.assertRaisesRegex(ValueError, '候选库外'):
            load_dataset_provenance(self.data)

    def test_neural_cli_saves_latest_audit_and_rejects_corruption_before_loading(self):
        newest = self.audit('10', overlap=['q'])
        config = self.root / 'config.json'
        write_json(config, {'device': 'cpu'})
        output = self.root / 'neural'
        args = ['foliorecall', 'evaluate', '--config', str(config), '--data', str(self.data),
                '--index', str(self.root / 'index'), '--output', str(output)]
        with patch('sys.argv', args), patch('foliorecall.search.load_index', return_value=(object(), self.pages)), \
             patch('foliorecall.query.load_query_encoder') as load_model, \
             patch('foliorecall.evaluation.evaluate', return_value={'scope': 'legacy', 'metrics': {}}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            load_model.assert_called_once()
            result = read_json(output / 'result.json')
            self.assertEqual(result['dataset']['normalized_training_query_overlap'], ['q'])
            self.assertEqual(result['dataset']['training_overlap']['record'], str(newest.resolve()))
            write_json(newest, {'broken': True})
            load_model.reset_mock()
            with self.assertRaisesRegex(ValueError, '交集检查记录格式'):
                main()
            load_model.assert_not_called()
        self.assertEqual((self.data / 'source.json').read_bytes(), self.original)

    def test_bm25_cli_saves_latest_unchecked_audit_before_request_timing(self):
        self.audit('1', overlap=['q'])
        newest = self.audit('10', checked=False)
        config = self.root / 'config.json'
        write_json(config, {'bm25': {'k1': 1.5, 'b': .75, 'epsilon': .25},
                            'protocol': {'warmups': 1, 'repeats': 1}})
        output = self.root / 'bm25'
        args = ['foliorecall', 'evaluate-text', '--config', str(config), '--data', str(self.data),
                '--output', str(output)]
        with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
        result = read_json(output / 'result.json')
        self.assertEqual(result['dataset']['training_overlap']['status'], 'not_checked')
        self.assertIsNone(result['dataset']['normalized_training_query_overlap'])
        self.assertEqual(result['dataset']['training_overlap']['record'], str(newest.resolve()))
        self.assertEqual(len(result['requests']), 1)
        self.assertEqual((self.data / 'source.json').read_bytes(), self.original)
        write_json(newest, {'broken': True})
        args[-1] = str(self.root / 'bm25-invalid')
        with patch('sys.argv', args), patch('rank_bm25.BM25Okapi') as build:
            with self.assertRaisesRegex(ValueError, '交集检查记录格式'):
                main()
            build.assert_not_called()
        self.assertFalse((self.root / 'bm25-invalid').exists())


if __name__ == '__main__':
    unittest.main()
