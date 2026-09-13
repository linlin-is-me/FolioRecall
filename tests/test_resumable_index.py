import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch  # Import before the temporary sys.modules overlay; never initialize CUDA.
from PIL import Image

from foliorecall.indexing import build_index
from foliorecall.io import read_json, write_rows
from foliorecall.search import load_index


class Model:
    max_seq_length = 32
    def preprocess(self, *args, **kwargs):
        return {'input_ids': np.zeros((1, 2)), 'image_grid_thw': np.array([[1, 2, 2]])}
    def __getitem__(self, key):
        return SimpleNamespace(pooling_mode='lasttoken')


class ResumableIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = []
        for i in range(3):
            image = self.root / f'{i}.png'
            Image.new('RGB', (8, 8), 'white').save(image)
            self.rows.append(dict(page_id=str(i), doc_id='d', page_number=i+1, source='test', preview=str(image)))
        self.pages = self.root / 'pages.jsonl'
        write_rows(self.pages, self.rows)
        self.config = dict(model_id='test', dimension=2, device='cpu', prompt='', normalize=True)
        self.encoded = []
        self.pause = False
        def encode(model, rows, config):
            self.encoded.extend(r['page_id'] for r in rows)
            if self.pause:
                self.pause = False
                os.kill(os.getpid(), signal.SIGINT)
            return np.array([[1, 0] if int(r['page_id']) % 2 == 0 else [0, 1] for r in rows], dtype=np.float32)
        self.load = unittest.mock.Mock(return_value=Model())
        self.fake = SimpleNamespace(load_encoder=self.load, encode_pages=encode, processing=lambda c: {})
        self.modules = patch.dict(sys.modules, {'foliorecall.encoding': self.fake})
        self.modules.start()
    def tearDown(self):
        self.modules.stop()
        self.tmp.cleanup()

    def test_pause_resume_matches_continuous_and_loads_once_per_segment(self):
        self.pause = True
        out = self.root / 'resumed'
        first = build_index(self.config, self.pages, out, chunk_size=2)
        self.assertFalse(first['complete'])
        self.assertFalse(out.exists())
        self.assertEqual(self.encoded, ['0', '1'])
        result = build_index(self.config, self.pages, out, resume=True, chunk_size=2)
        self.assertTrue(result['complete'])
        self.assertEqual(self.encoded, ['0', '1', '2'])
        self.assertEqual(self.load.call_count, 2)
        build_index(self.config, self.pages, self.root / 'continuous', chunk_size=2)
        a, rows = load_index(out, self.config)
        b, _ = load_index(self.root / 'continuous', self.config)
        np.testing.assert_array_equal(a.reconstruct_n(0, 3), b.reconstruct_n(0, 3))
        self.assertEqual(rows, self.rows)
        self.assertEqual(len(result['segments']), 2)
        with self.assertRaisesRegex(ValueError, '不可覆盖'):
            build_index(self.config, self.pages, out, resume=True)

    def test_changed_input_and_row_order_fail_before_model_load(self):
        self.pause = True
        out = self.root / 'partial'
        build_index(self.config, self.pages, out, chunk_size=2)
        self.load.reset_mock()
        write_rows(self.pages, list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, '已变化'):
            build_index(self.config, self.pages, out, True, 2)
        self.load.assert_not_called()
        write_rows(self.pages, self.rows)
        Image.new('RGB', (12, 12), 'red').save(self.rows[0]['preview'])
        with self.assertRaisesRegex(ValueError, '已变化'):
            build_index(self.config, self.pages, out, True, 2)
        self.load.assert_not_called()

    def test_failed_finalization_does_not_publish_or_reencode(self):
        out = self.root / 'retry'
        with patch('foliorecall.indexing.save_index', side_effect=RuntimeError('interrupted save')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                build_index(self.config, self.pages, out, chunk_size=2)
        self.assertFalse(out.exists())
        self.load.reset_mock()
        result = build_index(self.config, self.pages, out, True, 2)
        self.assertTrue(result['complete'])
        self.load.assert_not_called()
        self.assertEqual(self.encoded, ['0', '1', '2'])

    def test_corrupt_chunk_order_is_rejected_before_load(self):
        self.pause = True
        out = self.root / 'corrupt'
        build_index(self.config, self.pages, out, chunk_size=2)
        from foliorecall.io import write_json
        p = self.root / 'corrupt.work/chunks/00000000.json'
        r = read_json(p)
        r['page_ids'].reverse()
        write_json(p, r)
        self.load.reset_mock()
        with self.assertRaisesRegex(ValueError, '行序'):
            build_index(self.config, self.pages, out, True, 2)
        self.load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
