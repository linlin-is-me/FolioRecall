import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from foliorecall.documents import import_documents
from foliorecall.evaluation import metrics
from foliorecall.io import read_rows
from foliorecall.search import save_index, load_index, search


class CoreTests(unittest.TestCase):
    def test_persisted_index_preserves_source_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8), "white").save(root / "page.png")
            rows = [{"page_id": pid, "doc_id": "document", "page_number": number,
                     "source": "original.pdf", "preview": str(root / "page.png")}
                    for pid, number in (("b", 10), ("a", 2), ("c", 7))]
            config = {"dimension": 3, "model_id": "test", "normalize": True}
            vectors = np.eye(3, dtype=np.float32)
            index = save_index(vectors, rows, config, root / "index")
            before = search(index, rows, vectors[:1])[0]
            restored, metadata = load_index(root / "index", config)
            self.assertEqual(before, search(restored, metadata, vectors[:1])[0])
            self.assertEqual(before[0]["page_number"], 10)
            self.assertEqual(before[0]["source"], "original.pdf")
            with self.assertRaisesRegex(ValueError, "不匹配"):
                load_index(root / "index", dict(config, model_id="different"))
            with self.assertRaisesRegex(ValueError, "空索引"):
                save_index(np.empty((0, 3)), [], config, root / "empty")

    def test_graded_multi_positive_metrics(self):
        result = metrics(["irrelevant", "a", "b"], {"a": 2, "b": 1})
        expected = (2 / math.log2(3) + 1 / math.log2(4)) / (2 + 1 / math.log2(3))
        self.assertAlmostEqual(result["nDCG@10"], expected)
        self.assertEqual(result["Recall@5"], 1)
        self.assertEqual(metrics(["a"], {"a": 2, "b": 1})["Recall@5"], 0.5)

    def test_bad_pdf_does_not_discard_good_image_and_reimport_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8), "red").save(root / "good.png")
            (root / "broken.pdf").write_bytes(b"not a PDF")
            inputs = [root / "broken.pdf", root / "good.png"]
            result = import_documents(inputs, root / "output")
            self.assertEqual(result["pages"], 1)
            self.assertEqual(len(result["errors"]), 1)
            first = read_rows(result["manifest"])
            again = import_documents(inputs, root / "output")
            self.assertEqual(first, read_rows(again["manifest"]))


if __name__ == "__main__":
    unittest.main()
