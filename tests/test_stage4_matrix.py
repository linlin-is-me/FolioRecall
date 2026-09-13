from pathlib import Path
import tempfile
import unittest

from foliorecall.io import read_json, write_json
from foliorecall.stage4_results import index_costs, paired_interval, summarize_matrix


class MatrixTests(unittest.TestCase):
    def test_index_cost_retains_old_scope_and_adds_failed_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / 'index'
            index.mkdir()
            (index / 'index.faiss').write_bytes(b'index')
            (index / 'pages.jsonl').write_bytes(b'pages')
            (index / 'config.json').write_bytes(b'config')
            write_json(index / 'build.json', {'seconds': 8, 'timing_scope': 'excludes loading'})
            cells = [{'task': 'a', 'teacher': 'original', 'index': str(index)}]*2
            budget = root / 'budget'
            for name, seconds, code in [('failed', 3, 1), ('resumed', 8, 0)]:
                write_json(budget / 'runs' / name / 'process.json', {'command': ['index', '--output', str(index)], 'seconds': seconds, 'returncode': code})
            result = index_costs(cells, budget)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['gpu_process_seconds'], 11)
            self.assertEqual(result[0]['build']['timing_scope'], 'excludes loading')
            self.assertNotIn('model_loading_seconds', result[0]['build'])

    def test_missing_domains_do_not_produce_macro_and_query_ids_are_task_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [{'name': name, 'pages': 1, 'english_queries': 1} for name in ['a', 'b']]
            cells = []
            for task in tasks:
                for candidate in ['original', 'ml']:
                    file = root / task['name'] / candidate / 'result.json'
                    score = 1. if candidate == 'ml' else .5
                    metrics = {'nDCG@10': score, 'Recall@5': 1., 'Recall@10': 1.}
                    t = {'p50': .01, 'p95': .02}
                    write_json(file, {'queries': 1, 'candidates': 1, 'device': 'cuda', 'dtype': 'bfloat16',
                        'warmups': 5, 'repeats': 3, 'metrics': metrics, 'request_seconds': t,
                        'rounds': [{'request_seconds': t}]*3,
                        'results': [{'query_id': 'same-id', 'query': 'query', 'ranking': [{'page_id': 'p'}], **metrics}]})
                    cells.append({'task': task['name'], 'device': 'gpu', 'candidate': candidate, 'result': str(file)})
            cells.append({'task': 'a', 'device': 'gpu', 'candidate': 'lora750', 'result': str(root / 'missing.json')})
            manifest = root / 'jobs.json'
            write_json(manifest, {'tasks': tasks, 'matrix': cells})
            result = summarize_matrix(manifest, root / 'summary')
            self.assertFalse(result['complete_matrix'])
            self.assertNotIn('gpu/lora750', result['macro'])
            self.assertEqual(result['macro']['gpu/ml']['nDCG@10'], 1.)
            changes = read_json(root / 'summary/ml-vs-original-queries.json')
            self.assertEqual(len(changes), 2)
            self.assertEqual({r['task'] for r in changes}, {'a', 'b'})

    def test_paired_constant_differences_have_exact_interval(self):
        r = paired_interval([[.25, .25], [.25]], repeats=100)
        self.assertEqual((r['difference'], r['lower_95'], r['upper_95']), (.25, .25, .25))


if __name__ == '__main__':
    unittest.main()
