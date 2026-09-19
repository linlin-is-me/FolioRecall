"""Freeze attributed QA reviews; this module never invents reference facts."""
from collections import Counter
from pathlib import Path
import os

from .io import read_json, read_rows, write_rows, provenance
from .qa_storage import atomic_json, digest
from .retrieval import page_identity
from .evaluation import validate_candidate_corpus

CATEGORIES = ('numeric', 'chart', 'multi_page', 'single_page')


def reviewed_task(original, review, pages):
    ids = {p['page_id'] for p in pages}
    if review.get('task_id') != original['task_id'] or review.get('state') != 'reviewed':
        raise ValueError('任务审查尚未完成或 ID 错配')
    for lang in ('en', 'zh'):
        if review.get('source_question_'+lang) != original['question_'+lang]:
            raise ValueError('审查绑定的原问题发生变化')
    attribution = review.get('review', {})
    if (attribution.get('assistant') != 'text_and_image_checked'
            or attribution.get('human') != 'not_checked'
            or not attribution.get('translation')
            or attribution.get('unresolved_review_issues') != []):
        raise ValueError('审查来源或技术阻塞状态不完整')
    category, scope = review.get('category'), review.get('answerability')
    if category not in CATEGORIES or not review.get('category_reason'):
        raise ValueError('缺少核验题型及依据')
    if scope not in ('complete', 'partial', 'insufficient') or review.get('answerable') is not (scope == 'complete'):
        raise ValueError('可回答范围与 answerable 不一致')
    facts, evidence = review.get('reference_facts'), review.get('evidence_page_ids')
    if not isinstance(facts, list) or any(not isinstance(f, str) or not f.strip() for f in facts):
        raise ValueError('参考事实格式错误')
    if not isinstance(evidence, list) or len(set(evidence)) != len(evidence) or not set(evidence).issubset(ids):
        raise ValueError('证据越界或重复')
    if scope in ('complete', 'partial') and (not facts or not evidence):
        raise ValueError('完整或部分可答任务需要有效事实和证据')
    unresolved = review.get('unresolved_requirements')
    if not isinstance(unresolved, list) or bool(unresolved) != (scope != 'complete'):
        raise ValueError('未解决要求与可回答范围不一致')
    if category == 'numeric' and not review.get('calculation'):
        raise ValueError('计算题必须记录公式、输入与单位')
    links = review.get('fact_evidence', [])
    if sorted(f.get('fact_index', -1) for f in links) != list(range(len(facts))):
        raise ValueError('每项事实必须恰有一项证据关联')
    linked = set()
    for fact in links:
        if not fact.get('location') or not fact.get('page_ids') or not set(fact['page_ids']).issubset(evidence):
            raise ValueError('事实证据页或页内位置无效')
        linked.update(fact['page_ids'])
    if linked != set(evidence):
        raise ValueError('存在没有事实关联的证据页')
    if not set(evidence).issubset(attribution.get('image_pages', [])):
        raise ValueError('决定答案的证据页尚未看图')
    if not set(attribution.get('image_pages', [])).issubset(ids):
        raise ValueError('图像核验页越界')
    rubric = review.get('rubric', {})
    if set(rubric) != {'fact_correctness', 'answer_coverage', 'citation_support', 'insufficiency_handling'}:
        raise ValueError('必须冻结四项评分规则')
    if rubric['answer_coverage'].get('required_fact_indices') != list(range(len(facts))):
        raise ValueError('覆盖度规则与必需事实不匹配')
    result = dict(original, **{k:v for k,v in review.items() if k not in ('source_question_en','source_question_zh','state')})
    result.update(upstream=original, scope='retrospective development; assistant-reviewed, not human gold',
                  question_zh=review.get('question_zh_revision') or original['question_zh'])
    result['attributes'] = {'calculation': bool(review.get('calculation')), 'chart': category == 'chart',
                            'multi_page': len(evidence) > 1}
    return result


def select_repeats(tasks):
    ordered = sorted(tasks, key=lambda r:r['task_id'])
    direct = [r for r in ordered if r['answerability']=='complete' and r['category']=='single_page']
    first = (direct or ordered)[0]
    remaining = [r for r in ordered if r != first]
    # Category priority is separate from the primary-category hierarchy.
    for category in ('multi_page','numeric','chart'):
        candidates = [r for r in remaining if r['category']==category]
        if candidates:
            second = candidates[0]
            break
    else:
        second = remaining[0]
    return [{'task_id':r['task_id'], 'category':r['category'], 'answerability':r['answerability']}
            for r in (first,second)]


def freeze_development(selection, reviews, datasets, data_root, output):
    originals, reviews = read_rows(selection), read_rows(reviews)
    tasks_config = read_json(datasets)['tasks']
    if len(originals)!=40 or len({r['task_id'] for r in originals})!=40:
        raise ValueError('必须保留原 40 个唯一任务 ID')
    if len(reviews)!=40 or len({r['task_id'] for r in reviews})!=40 or {r['task_id'] for r in reviews}!={r['task_id'] for r in originals}:
        raise ValueError('审查记录缺失、重复或替换了任务')
    counts = Counter(r['task'] for r in originals)
    if counts != Counter({c['name']:8 for c in tasks_config}):
        raise ValueError('开发参考要求固定五域各八题')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('冻结输出已存在，请使用新目录')
    mapping = {r['task_id']:r for r in reviews}
    prepared = []
    for spec in tasks_config:
        root = Path(data_root).resolve()/spec['name']
        source, pages, texts = read_json(root/'source.json'), read_rows(root/'pages.jsonl'), read_rows(root/'texts.jsonl')
        if (source.get('task') != spec or source.get('revision') != spec['revision']
                or source.get('dataset') != spec['dataset'] or source.get('candidates')!=spec['pages'] or len(pages)!=spec['pages']):
            raise ValueError('冻结语料的领域、revision 或完整候选数错配')
        validate_candidate_corpus(pages,texts)
        rows = [reviewed_task(r,mapping[r['task_id']],pages) for r in originals if r['task']==spec['name']]
        # Open only actually reviewed pages, not all 12,969 images.
        from PIL import Image
        by_id = {p['page_id']:p for p in pages}
        for pid in {p for r in rows for p in r['review']['image_pages']}:
            preview = Path(by_id[pid]['preview'])
            if not preview.is_absolute():
                preview = root/preview
            with Image.open(preview) as im:
                im.load()
        prepared.append((spec,root,pages,rows,select_repeats(rows)))
    # No output is published before every review and referenced image passes.
    temporary = output.with_name(output.name+'.preparing')
    if temporary.exists():
        raise ValueError('存在未完成冻结目录，请检查后选择新输出位置')
    provenance(temporary, {'selection':str(selection), 'reviews':str(reviews), 'datasets':str(datasets),
                          'source_digest':digest(originals),'review_digest':digest(reviews),
                          'scope':'assistant source review; no API/model execution'})
    repeat_rows, summaries = [], {}
    for spec,root,pages,rows,repeated in prepared:
        domain = temporary/spec['name']
        domain.mkdir()
        write_rows(domain/'tasks.jsonl',rows)
        status = {'schema_version':2,'split':'development','task':spec['name'],'dataset':spec['dataset'],
                  'revision':spec['revision'],'qa_reference_frozen':True,'evaluation_allowed':True,
                  'human_gold':False,'corpus_pages':page_identity(pages),'tasks_digest':digest(rows),
                  'source_record':str(root/'source.json')}
        atomic_json(domain/'status.json',status)
        atomic_json(domain/'public-corpus.json',{'visibility':'public','task':spec['name'],
            'dataset':spec['dataset'],'revision':spec['revision'],'pages':page_identity(pages),
            'source_urls':[f'https://huggingface.co/datasets/{spec["dataset"]}/tree/{spec["revision"]}'],
            'texts':os.path.relpath(root/'texts.jsonl',output/spec['name']),
            'scope':'Only fixed public benchmark pages; page IDs are domain-local'})
        summaries[spec['name']] = {'tasks':len(rows),'pages':len(pages),'categories':dict(Counter(r['category'] for r in rows)),
                                  'answerability':dict(Counter(r['answerability'] for r in rows))}
        repeat_rows.extend(dict(r,domain=spec['name']) for r in repeated)
    atomic_json(temporary/'repeat-tasks.json',{'selection_before_system_answers':True,
                'rule':'complete single-page first; multi-page, numeric, chart priority second; string ID tie-break',
                'tasks':repeat_rows})
    atomic_json(temporary/'summary.json',{'tasks':40,'pages':sum(c['pages'] for c in tasks_config),
                'domains':summaries,'human':'not_checked','status':'frozen_assistant_reference'})
    temporary.rename(output)
    return read_json(output/'summary.json')
