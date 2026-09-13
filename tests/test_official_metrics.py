import unittest

from foliorecall.evaluation import metrics
from foliorecall.text_baseline import tokenize


class OfficialMetricsTests(unittest.TestCase):
    def test_trec_linear_grades_multi_positive_and_short_rankings(self):
        import pytrec_eval
        relevance = {'2': 2, '11': 1, '9': 2, 'missing': 1}
        for scores in ({'2': 1., '11': 1., '9': 1., 'none': 0.}, {'none': 1.}, {'2': 1.}):
            expected = pytrec_eval.RelevanceEvaluator({'q': relevance}, {'ndcg_cut_10', 'recall_5', 'recall_10'}).evaluate({'q': scores})['q']
            ids = sorted(scores, key=lambda pid: (scores[pid], pid), reverse=True)
            actual = metrics(ids, relevance)
            for own, official in [('nDCG@10', 'ndcg_cut_10'), ('Recall@5', 'recall_5'), ('Recall@10', 'recall_10')]:
                self.assertAlmostEqual(actual[own], expected[official], places=12)

    def test_invalid_rankings_and_labels(self):
        for ids, relevance in [(['a', 'a'], {'a': 1}), (['a'], {'a': 0}),
                               (['a'], {'a': 1, 'b': -1}), (['a'], {'a': float('nan')})]:
            with self.assertRaises(ValueError):
                metrics(ids, relevance)

    def test_faiss_top10_tie_boundary_matches_trec(self):
        import faiss
        import numpy as np
        import pytrec_eval
        from foliorecall.search import search
        ids = ['2', '11', '9', '8', '7', '6', '5', '4', '3', '1', '0', '10']
        relevance = {'0': 2, '1': 1, '2': 2, '9': 1, '11': 2}
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0]] * len(ids), dtype=np.float32))
        ranking = search(index, [{'page_id': pid} for pid in ids], [[1, 0]], 10)[0]
        actual = metrics([row['page_id'] for row in ranking], relevance)
        expected = pytrec_eval.RelevanceEvaluator({'q': relevance},
            {'ndcg_cut_10', 'recall_5', 'recall_10'}).evaluate({'q': dict.fromkeys(ids, 1.)})['q']
        for own, official in [('nDCG@10', 'ndcg_cut_10'), ('Recall@5', 'recall_5'), ('Recall@10', 'recall_10')]:
            self.assertAlmostEqual(actual[own], expected[official], places=12)

    def test_shared_text_normalization_preserves_numbers(self):
        self.assertEqual(tokenize('ＡＢＣ 2025, D0 + pH!'), ['abc', '2025', 'd0', 'ph'])


if __name__ == '__main__':
    unittest.main()
