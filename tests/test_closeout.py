import sys
import unittest
from unittest.mock import patch

from foliorecall.evaluation import validate_candidate_corpus


class CloseoutTests(unittest.TestCase):
    def test_complete_candidates_allow_reordering_but_reject_missing_extra_duplicates(self):
        corpus = [{"page_id": "positive"}, {"page_id": "negative"}]
        validate_candidate_corpus(list(reversed(corpus)), corpus)
        for pages in (corpus[:1], corpus + [{"page_id": "extra"}], corpus + corpus[:1], []):
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                validate_candidate_corpus(pages, corpus)
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_candidate_corpus(corpus, corpus + corpus[:1])

    def test_cli_rejects_missing_nonrelevant_candidate_before_model_loading(self):
        from foliorecall.__main__ import main
        args = ["foliorecall", "evaluate", "--index", "unused-index", "--output", "unused-output"]
        # Any attempt to import/load the encoder fails this metadata-only check.
        with patch.object(sys, "argv", args), \
             patch("foliorecall.__main__.read_json", return_value={}), \
             patch("foliorecall.__main__.read_rows", return_value=[{"page_id": "positive"}, {"page_id": "negative"}]), \
             patch("foliorecall.search.load_index", return_value=(None, [{"page_id": "positive"}])), \
             patch("foliorecall.__main__.provenance") as provenance, \
             patch.dict(sys.modules, {"foliorecall.encoding": None}):
            with self.assertRaisesRegex(ValueError, "索引缺少 1 页"):
                main()
            provenance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
