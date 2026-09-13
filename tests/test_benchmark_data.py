import unittest

from foliorecall.benchmark_data import validate_task


class BenchmarkDataTests(unittest.TestCase):
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
