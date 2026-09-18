import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foliorecall.io import write_rows, read_json, write_json
from foliorecall.retrieval import build_text_index, load_retrieval, rrf


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pages = [dict(page_id=str(i), doc_id='doc', page_number=i+1) for i in range(4)]
        self.texts = [dict(page_id=str(i), text=t) for i,t in enumerate(['Cars energy', '', 'WATER water', 'Solar cost'])]
        write_rows(self.root/'pages.jsonl', self.pages)
        write_rows(self.root/'texts.jsonl', self.texts)
        self.config = {'mode': 'hybrid', 'rrf_k': 60, 'candidate_k': 50, 'text_index': str(self.root/'index')}

    def build(self):
        return build_text_index(self.root/'pages.jsonl', self.root/'texts.jsonl', self.root/'index')

    def test_full_corpus_roundtrip_and_empty_page(self):
        result = self.build()
        self.assertEqual(result['coverage'], .75)
        state = load_retrieval(self.config, self.pages)
        self.assertEqual(state['texts']['1'], '')
        self.assertEqual(len(state['model'].get_scores(['water'])), 4)
        with self.assertRaises(ValueError):
            load_retrieval(self.config, self.pages[::-1])
        with self.assertRaises(ValueError):
            self.build()

    def test_corrupt_binding_or_text_fails(self):
        self.build()
        path = self.root/'index/index.json'
        record = read_json(path)
        record['texts'][0] = 'replacement'
        write_json(path, record)
        with self.assertRaises(ValueError):
            load_retrieval(self.config, self.pages)

    def test_empty_and_duplicate_ids_rejected(self):
        for texts in ([dict(r, text='') for r in self.texts], self.texts+[self.texts[0]]):
            write_rows(self.root/'texts.jsonl', texts)
            with self.assertRaises(ValueError):
                self.build()

    def test_mismatch_before_model_load(self):
        from foliorecall.query import load_search_session
        self.build()
        with patch('foliorecall.search.load_index', return_value=(None,self.pages[::-1])), \
             patch('foliorecall.query.load_query_encoder') as model, self.assertRaises(ValueError):
            load_search_session('unused', {}, retrieval_config=self.config)
        model.assert_not_called()

    def test_rrf_dedup_missing_channel_and_string_ties(self):
        row = lambda pid: {'page_id': pid, 'score': .7}
        result = rrf([row('10'),row('2')], [row('2'),row('10')])
        self.assertEqual([r['page_id'] for r in result], ['2','10'])
        self.assertEqual(result[0]['channels']['visual']['rank'], 2)
        self.assertEqual(result[0]['score_kind'], 'rrf')
        self.assertEqual(rrf([], [row('2')])[0]['score'], 1/61)
        with self.assertRaises(ValueError):
            rrf([row('2'),row('2')], [])

    def test_hybrid_requests_top50_and_times_serialization(self):
        from foliorecall.query import retrieve
        self.build()
        state = load_retrieval(self.config, self.pages)
        with patch('foliorecall.query.encode_query_texts', return_value=[[1]]), \
             patch('foliorecall.query.search', return_value=[[dict(self.pages[0], score=.8)]]) as search:
            ranked, payload, times = retrieve(None, {'device':'cpu'}, None, self.pages, 'water', 3, state)
        self.assertEqual(search.call_args.args[-1], 50)
        self.assertEqual(json.loads(payload), ranked)
        self.assertGreaterEqual(times['request_seconds'], times['query_seconds'])


class TextExtractionTests(unittest.TestCase):
    def test_pdf_text_cache_and_change_protection(self):
        import pypdfium2 as pdfium
        from foliorecall.documents import import_documents
        from foliorecall.io import read_rows
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'blank.pdf'
            with pdfium.PdfDocument.new() as pdf:
                page = pdf.new_page(100,100)
                page.close()
                pdf.save(source)
            out = root/'imported'
            import_documents([source], out, extract_text=True)
            rows = read_rows(out/'pages.jsonl')
            preview = Path(rows[0]['preview'])
            previous = preview.stat().st_mtime_ns
            texts = (out/'texts.jsonl').read_bytes()
            import_documents([source], out, extract_text=True)
            self.assertEqual(preview.stat().st_mtime_ns, previous)
            self.assertEqual((out/'texts.jsonl').read_bytes(), texts)
            self.assertEqual(read_rows(out/'texts.jsonl')[0]['text'], '')
            source.write_bytes(source.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, '新输出目录'):
                import_documents([source], out, extract_text=True)
            self.assertEqual(preview.stat().st_mtime_ns, previous)
