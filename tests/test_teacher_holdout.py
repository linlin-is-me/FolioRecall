import io
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from foliorecall.benchmark_data import prepare_benchmark, load_dataset_provenance
from foliorecall.documents import validate_pages
from foliorecall.io import read_json, read_rows, write_json, write_rows
from scripts.run_teacher_holdout import prepare, commands, report, audit_data, main, SEEDS, ORDER


class V2DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = {'name': 'fixture', 'format': 'vidore-v2', 'dataset': 'fixture/v2',
                     'revision': 'fixed', 'pages': 2, 'english_queries': 1}

    def fixture(self, esg=False, missing_doc=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        image = io.BytesIO()
        Image.new('RGB', (3, 2), 'white').save(image, format='PNG')
        source = {'pdf_url': 'https://example.org/report.pdf', 'company': 'Test', 'date': '2025'} if esg else {'doc-id': 'lecture'}
        if missing_doc:
            source = {}
        data = {'corpus/test-00000.parquet': [dict(source, **{'corpus-id': i, 'image': {'bytes': image.getvalue()}}) for i in (0, 1)],
                'queries/test-00000.parquet': [{'query-id': 0, 'query': 'Question'}],
                'qrels/test-00000.parquet': [{'query-id': 0, 'corpus-id': 0, 'score': 3},
                                          {'query-id': 0, 'corpus-id': 1, 'score': 1}],
                'corpus/train-00000.parquet': [{'not': 'a test row'}],
                'queries/train-00000.parquet': [{'query-id': 99, 'query': 'Excluded train'}]}
        for name, rows in data.items():
            target = self.root / 'upstream' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), target)
        return data

    def prepare(self, esg=False, missing_doc=False):
        files = self.fixture(esg, missing_doc)
        def download_file(url, target, size):
            # Copies frozen fixture bytes, with the same API as the shared downloader.
            relative = url.split('/fixed/')[1]
            shutil.copyfile(self.root/'upstream'/relative, target)
            return size
        with patch('huggingface_hub.HfApi.list_repo_files', return_value=list(files)), \
             patch('huggingface_hub.HfApi.get_paths_info', return_value=[SimpleNamespace(
                 path=f, size=(self.root/'upstream'/f).stat().st_size) for f in files]), \
             patch('foliorecall.data_preparation.download_shard', side_effect=download_file), \
             patch('huggingface_hub.try_to_load_from_cache', return_value=None), \
             patch('huggingface_hub.hf_hub_download', side_effect=lambda repo, file, **kw: str(self.root/'upstream'/file)) as download:
            result = prepare_benchmark({'tasks': [self.task]}, self.root/'data')[0]
            self.assertTrue(all('/train-' not in call.args[1] for call in download.call_args_list))
        return result

    def test_v2_full_corpus_graded_labels_and_unknown_physical_page(self):
        result = self.prepare()
        data = self.root/'data/fixture'
        pages = validate_pages(read_rows(data/'pages.jsonl'))
        self.assertEqual([r['page_id'] for r in pages], ['fixture:0', 'fixture:1'])
        self.assertTrue(all(r['page_number'] is None for r in pages))
        self.assertTrue(all(r['doc_id'] == 'fixture:lecture' for r in pages))
        self.assertEqual(read_json(data/'qrels.json'), {'fixture:0': {'fixture:0': 3., 'fixture:1': 1.}})
        self.assertEqual((result['documents'], result['queries'], result['candidates']), (1, 1, 2))
        self.assertEqual(load_dataset_provenance(data)['revision'], 'fixed')
        import numpy as np
        from foliorecall.search import save_index, load_index
        save_index(np.eye(2, dtype=np.float32), pages, {'dimension': 2}, self.root/'index')
        self.assertEqual(load_index(self.root/'index', {'dimension': 2})[0].ntotal, 2)
        pages[0].pop('page_number_kind')
        with self.assertRaises(ValueError):
            validate_pages(pages)

    def test_esg_uses_actual_pdf_source(self):
        self.prepare(esg=True)
        from foliorecall.io import read_rows
        page = read_rows(self.root/'data/fixture/pages.jsonl')[0]
        self.assertEqual(page['doc_id'], 'fixture:https://example.org/report.pdf')
        self.assertEqual(page['source'], 'https://example.org/report.pdf')

    def test_completed_v2_reuses_data_without_download_or_image_rewrite(self):
        self.prepare()
        directory = self.root/'data/fixture'
        original = (directory/'source.json').read_bytes()
        images = {p.name:p.stat().st_mtime_ns for p in (directory/'images').iterdir()}
        with patch('huggingface_hub.HfApi.list_repo_files') as network:
            prepare_benchmark({'tasks':[self.task]}, self.root/'data')
            network.assert_not_called()
        self.assertEqual(original, (directory/'source.json').read_bytes())
        self.assertEqual(images, {p.name:p.stat().st_mtime_ns for p in (directory/'images').iterdir()})

    def test_missing_source_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, '来源文档'):
            self.prepare(missing_doc=True)

    def test_audit_retains_overlap_and_independence_limits(self):
        self.prepare(esg=True)
        write_json(self.root/'manifest.json', {'data':str(self.root/'data'),
            'suite':{'tasks':[self.task]}, 'training_data':str(self.root/'training')})
        write_rows(self.root/'training/pages.jsonl', [])
        write_json(self.root/'old/split.json', {'train':[{'query':' question '}], 'dev':[]})
        write_rows(self.root/'metadata.jsonl', [{'query':'Unrelated'}])
        write_rows(self.root/'domains/old/pages.jsonl', [{'doc_id':'old', 'source':'https://example.org/report.pdf'}])
        write_rows(self.root/'domains/old/queries.jsonl', [])
        result = audit_data(self.root, self.root/'old', self.root/'domains', self.root/'metadata.jsonl')
        self.assertEqual((result['queries'],result['pages']), (1,2))
        self.assertEqual(result['tasks'][0]['normalized_query_overlap']['train_3000'], ['fixture:0'])
        self.assertEqual(result['tasks'][0]['exact_known_source_matches'], ['fixture:https://example.org/report.pdf'])
        self.assertTrue(result['limitations'])


class HoldoutEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'training-data'
        self.suite = self.root/'suite-input.json'
        write_json(self.data/'split.json', {'train': [{'query_id':str(i), 'query':f'q{i}'} for i in range(256)]})
        write_rows(self.data/'pages.jsonl', [])
        write_rows(self.data/'dev/pages.jsonl', [])
        write_json(self.suite, {'tasks': [{'name': 'a', 'pages': 3, 'english_queries': 2},
                                         {'name': 'b', 'pages': 2, 'english_queries': 1}]})

    def prepared(self):
        def batches(ds, size, seed):
            order = list(range(256))
            random.Random(seed).shuffle(order)
            return [order[i:i+4] for i in range(0, 256, 4)]
        with patch('foliorecall.trainer.validate_training_data', return_value={}), \
             patch('foliorecall.trainer.training_dataset', return_value=[]), \
             patch('foliorecall.trainer.batch_plan', side_effect=batches), \
             patch('foliorecall.trainer.load_encoder') as load, \
             patch('scripts.run_teacher_holdout.provenance'):
            result = prepare(self.root, self.suite, self.data)
            load.assert_not_called()
        return result

    def test_six_configs_pair_only_loss_changes_and_no_encoder(self):
        self.assertEqual(self.prepared()['training_runs'], 6)
        m = read_json(self.root/'manifest.json')
        self.assertEqual(m['training_order'], [f'{a}-{b}' for a,b in ORDER])
        for seed in SEEDS:
            a = read_json(self.root/f'configs/ordinary-{seed}.json')
            b = read_json(self.root/f'configs/llave-{seed}.json')
            self.assertEqual(a['seed'], seed)
            self.assertEqual(b['train'].pop('hardness_mode'), 'in_batch_negatives')
            self.assertEqual(b['train'].pop('hardness_strength'), 9)
            self.assertIsNone(a['train'].pop('hardness_mode'))
            self.assertEqual(a['train'].pop('hardness_strength'), 0)
            self.assertEqual(a, b)
        plans = read_json(self.root/'batches.json')['batches_by_seed']
        self.assertEqual(len({tuple(v[0]) for v in plans.values()}), 3)

    def test_list_commands_does_not_execute_and_retains_identity(self):
        self.prepared()
        with patch('subprocess.run') as run:
            training = commands(self.root, 'train')
            indexes = commands(self.root, 'index')
            evaluation = commands(self.root, 'evaluate')
            run.assert_not_called()
        self.assertEqual((len(training), len(indexes), len(evaluation)), (6, 14, 14))
        self.assertIn('--checkpoint-only', training[0])
        self.assertEqual(indexes[2][indexes[2].index('--config')+1],
                         str(self.root.resolve()/'training/ordinary-42/checkpoint-64/encoding.json'))
        self.assertNotIn('--benchmark', evaluation[0])
        resumed = commands(self.root, 'train', 'llave-123', resume=self.root/'checkpoint-32')
        self.assertIn('--resume', resumed[0])
        with self.assertRaises(ValueError):
            commands(self.root, 'train', 'missing')

    def test_prepare_refuses_changed_inputs_without_overwriting(self):
        self.prepared()
        original = (self.root/'manifest.json').read_bytes()
        suite = read_json(self.suite)
        suite['tasks'][0]['pages'] += 1
        write_json(self.suite, suite)
        with self.assertRaises(ValueError):
            self.prepared()
        self.assertEqual((self.root/'manifest.json').read_bytes(), original)

    def test_index_commands_resume_work_directory_and_preserve_completed_index(self):
        self.prepared()
        out = self.root/'indexes/original/a'
        self.assertNotIn('--resume', commands(self.root, 'index', 'original', 'a')[0])
        write_json(out.with_name('a.work')/'manifest.json', {'fixture': True})
        self.assertIn('--resume', commands(self.root, 'index', 'original', 'a')[0])
        write_json(out/'build.json', {'complete': True})
        before = (out/'build.json').read_bytes()
        self.assertEqual(commands(self.root, 'index', 'original', 'a'), [])
        self.assertEqual((out/'build.json').read_bytes(), before)

    def test_only_explicit_run_dispatches_selected_cli_command(self):
        self.prepared()
        write_json(self.root/'data-preparation.json', {'tasks':[{'normalized_query_overlap':{'train':[]}}]})
        args = ['holdout', 'commands', '--root', str(self.root), '--stage', 'train', '--model', 'ordinary-42']
        with patch('sys.argv', args), patch('subprocess.run') as run, patch('builtins.print'):
            main()
            run.assert_not_called()
        args[1] = 'run'
        with patch('sys.argv', args), patch('subprocess.run') as run, patch('builtins.print'):
            main()
            run.assert_called_once()
            self.assertIn('train', run.call_args.args[0])
            self.assertIn(str(self.root/'configs/ordinary-42.json'), run.call_args.args[0])

    def results(self):
        self.prepared()
        m = read_json(self.root/'manifest.json')
        for model in m['models']:
            if model['method'] == 'original':
                continue
            config = read_json(model['train_config'])
            write_json(self.root/'training'/model['id']/'settings.json', {'config':config})
            write_json(self.root/'training'/model['id']/'result.json', {'complete':True, 'completed_steps':64,
                'loss': {'name':'CachedMultipleNegativesRankingLoss', 'scale':20.,
                         'hardness_mode':config['train']['hardness_mode'],
                         'hardness_strength':config['train']['hardness_strength']}})
        for t in m['suite']['tasks']:
            queries = [{'query_id': f'{t["name"]}:{i}', 'query': f'Query {i}'} for i in range(t['english_queries'])]
            write_rows(Path(m['data'])/t['name']/'queries.jsonl', queries)
            for model in m['models']:
                seed_index = list(SEEDS).index(model['seed']) if model['seed'] is not None else 0
                score = .5 + .1 * seed_index + (.1 if model['method'] == 'llave' else 0) + (.1 if t['name'] == 'a' else 0)
                rows = [dict(q, ranking=[{'page_id':t['name']+':0'}], **{k:score for k in ('nDCG@10','Recall@5','Recall@10')}) for q in queries]
                write_json(self.root/'evaluations'/model['id']/t['name']/'result.json',
                           {'candidates':t['pages'], 'queries':len(rows), 'results':rows})

    def test_report_seed_sample_std_pairing_and_actual_denominators(self):
        self.results()
        result = report(self.root)
        self.assertEqual((result['queries'], result['query_executions']), (3, 21))
        stat = result['seed_summary']['ordinary']['domain_equal']['nDCG@10']
        self.assertAlmostEqual(stat['sample_std'], .1)
        self.assertAlmostEqual(result['comparisons']['llave-42-minus-ordinary-42']['nDCG@10']['mean_delta'], .1)
        self.assertNotEqual(result['models']['original']['domain_equal'], result['models']['original']['query_weighted'])
        paired_rows = read_json(self.root/'paired/llave-42-minus-ordinary-42.json')
        self.assertEqual(len(paired_rows), 3)
        self.assertEqual(result['cross_seed_consistency']['llave-ordinary']['nDCG@10']['all_seeds_improved'], 3)
        self.assertEqual(len(read_json(self.root/'cases/llave-ordinary.json')['improvements']), 3)

    def test_report_rejects_wrong_training_seed_or_incomplete_checkpoint(self):
        self.results()
        path = self.root/'training/ordinary-42/settings.json'
        settings = read_json(path)
        settings['config']['seed'] = 999
        write_json(path, settings)
        with self.assertRaisesRegex(ValueError, '实际种子'):
            report(self.root)
        settings['config']['seed'] = 42
        write_json(path, settings)
        path = self.root/'training/ordinary-42/result.json'
        training = read_json(path)
        training['completed_steps'] = 32
        write_json(path, training)
        with self.assertRaisesRegex(ValueError, '训练未完成'):
            report(self.root)

    def test_missing_or_duplicate_results_cannot_produce_complete_report(self):
        self.results()
        p = self.root/'evaluations/llave-2026/b/result.json'
        value = read_json(p)
        value['results'] += value['results']
        write_json(p, value)
        with self.assertRaises(ValueError):
            report(self.root)
        self.assertFalse((self.root/'comparison.json').exists())
        p.unlink()
        with self.assertRaises(ValueError):
            report(self.root)


if __name__ == '__main__':
    unittest.main()
