"""Prepare the three approved JRC PDFs locally; no OCR or model computation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.documents import import_documents
from foliorecall.io import read_json, read_rows, write_json, write_rows, provenance
from foliorecall.retrieval import build_text_index, page_identity

IDS = ('JRC141212','JRC141952','JRC143054')


def prepare(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('输出目录已存在，请使用新目录')
    files=[source/(name+'.pdf') for name in IDS]
    if any(not p.is_file() for p in files):
        raise ValueError('缺少三份已核对的 JRC PDF；不静默替换资料')
    provenance(output,{'source_pdfs':[str(p) for p in files],'source_ids':IDS,'dpi':150,
                       'scope':'full pages; PDF text layer only; stable JRC ID plus physical page number'})
    imported=output/'imported'
    result=import_documents(files,imported,extract_text=True)
    if result['errors']:
        raise ValueError('PDF 导入不完整，请检查 import-errors.json')
    texts={r['page_id']:r['text'] for r in read_rows(imported/'texts.jsonl')}
    rows,new_texts=[],[]
    for page in read_rows(imported/'pages.jsonl'):
        doc=Path(page['document_name']).stem
        pid=f'{doc}:{page["page_number"]}'
        new_texts.append({'page_id':pid,'text':texts[page['page_id']],'source_kind':'pdf-text-layer'})
        rows.append(dict(page,page_id=pid,doc_id=doc,
                         source=f'https://publications.jrc.ec.europa.eu/repository/handle/{doc}'))
    write_rows(output/'pages.jsonl',rows)
    write_rows(output/'texts.jsonl',new_texts)
    write_json(output/'public-corpus.json',{'visibility':'public','pages':page_identity(rows),'texts':'texts.jsonl',
        'source_urls':[f'https://publications.jrc.ec.europa.eu/repository/handle/{doc}' for doc in IDS],
        'license':'CC BY 4.0 except separately credited third-party materials; local research copy, no redistribution approval implied',
        'license_evidence':'physical page 2 of each PDF'})
    build_text_index(output/'pages.jsonl',output/'texts.jsonl',output/'text-index')
    write_json(output/'hybrid.json',{'mode':'hybrid','text_index':'text-index','rrf_k':60,'candidate_k':50})
    write_json(output/'status.json',{'pages':len(rows),'text_index':'complete','visual_index':'pending_gpu',
                                   'qa_tasks':'pending_reference_freeze','source_pdf_bytes':sum(p.stat().st_size for p in files)})
    print({'pages':len(rows),'output':str(output),'visual_index':'pending_gpu'},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    prepare(args.source,args.output)
