from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.agent import load_public_corpus
from foliorecall.io import read_json, write_json, write_rows, read_rows
from foliorecall.retrieval import load_retrieval, read_retrieval_config
from scripts.prepare_agent_demo import prepare


class AgentBundleTests(unittest.TestCase):
    def test_subset_ids_order_and_relative_text_loading_survive_move(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; source.mkdir()
            (source / 'configs').mkdir()
            (source / 'index.faiss').write_bytes(b'existing-vector-file')
            rows = [dict(page_id=f'p{i}', doc_id='d', page_number=i+1,
                         document_name='doc.pdf', source='https://example.org/doc', preview='page.png') for i in range(42)]
            write_rows(source / 'pages.jsonl', rows)
            write_json(source / 'bundle.json', {'pages':42})
            (source / 'SOURCES.md').write_text('retained license')
            pdf = root / 'doc.pdf'; pdf.write_bytes(b'cached-pdf')
            write_json(root / 'sources.json', [{'doc_name':'doc.pdf','local_path':str(pdf),'url':'https://example.org/doc'}])
            def extract(local_rows, output):
                self.assertEqual([r['page_id'] for r in local_rows], [r['page_id'] for r in rows])
                target = output / 'texts.jsonl'
                write_rows(target, [{'page_id':r['page_id'],'text':f'population evidence page {i}'} for i,r in enumerate(local_rows)])
                return target
            out = root / 'export'
            with patch('scripts.prepare_agent_demo.extract_page_texts', side_effect=extract):
                prepare(source, root / 'sources.json', out)
            moved = root / 'moved'; shutil.move(str(out), moved)
            self.assertEqual(read_rows(moved / 'pages.jsonl'), rows)
            self.assertEqual((moved / 'index.faiss').read_bytes(), b'existing-vector-file')
            self.assertEqual(len(load_public_corpus(moved / 'public-corpus.json', rows)),42)
            self.assertEqual(len(load_retrieval(read_retrieval_config(moved / 'hybrid.json'), rows)['texts']),42)
            config = read_json(moved / 'configs/agent-demo.json')
            self.assertEqual(config['phase_budget_cny'],20)
            self.assertFalse(Path(config['ledger']).is_absolute())


if __name__ == '__main__':
    unittest.main()
