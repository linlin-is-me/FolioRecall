import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from foliorecall.documents import import_documents
from foliorecall.io import provenance, read_rows
from foliorecall.search import load_index, save_index, search


class DeliveryTests(unittest.TestCase):
    def test_ties_include_all_boundary_candidates_and_use_string_ids(self):
        import faiss
        ids = ['2', '11', '9', '8', '7', '6', '5', '4', '3', '1', '0', '10']
        rows = [{'page_id': pid} for pid in ids]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1, 0]] * len(ids), dtype=np.float32))
        ranked = search(index, rows, [[1, 0]], 10)[0]
        self.assertEqual([r['page_id'] for r in ranked], sorted(ids, reverse=True)[:10])
        self.assertTrue(all(r['score'] == 1 for r in ranked))

    def test_non_git_provenance(self):
        with tempfile.TemporaryDirectory() as temporary, patch('subprocess.check_output', side_effect=FileNotFoundError('git')):
            result = provenance(temporary, {'device': 'cpu'})
            self.assertIsNone(result['commit'])
            self.assertIn('package_version', result)
            self.assertIn('git_unavailable', result)

    def test_changed_input_preserves_old_preview_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'page.png'
            Image.new('RGB', (8, 8), 'red').save(source)
            result = import_documents([source], root / 'imported')
            rows = read_rows(result['manifest'])
            before = Path(rows[0]['preview']).read_bytes()
            manifest = Path(result['manifest']).read_bytes()
            Image.new('RGB', (12, 12), 'blue').save(source)
            with self.assertRaisesRegex(ValueError, '新输出目录'):
                import_documents([source], root / 'imported')
            self.assertEqual(Path(rows[0]['preview']).read_bytes(), before)
            self.assertEqual(Path(result['manifest']).read_bytes(), manifest)

    def test_portable_preview_and_legacy_paths(self):
        from foliorecall.io import write_rows
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'page.png'
            Image.new('RGB', (8, 8)).save(image)
            config = {'model_id': 'test', 'dimension': 2}
            rows = [{'page_id': 'a', 'doc_id': 'd', 'page_number': 3, 'source': 'https://example.org/p.pdf', 'preview': str(image)}]
            save_index([[1, 0]], rows, config, root / 'original')
            self.assertEqual(load_index(root / 'original', config)[1], rows)
            import shutil
            shutil.copy(image, root / 'original' / 'page.png')
            write_rows(root / 'original' / 'pages.jsonl', [dict(rows[0], preview='page.png')])
            shutil.copytree(root / 'original', root / 'moved')
            _, loaded = load_index(root / 'moved', config)
            self.assertEqual(Path(loaded[0]['preview']), root / 'moved' / 'page.png')
            self.assertEqual(loaded[0]['page_number'], 3)


if __name__ == '__main__':
    unittest.main()
