"""Persist explicit attributed judgments; arithmetic is not a semantic judge."""
from pathlib import Path
import math

from .qa_storage import atomic_json, digest
from .qa_evaluation import summarize_attempts


def save_judgments(attempts, judgments, output):
    prepared = []
    for judgment in judgments:
        matches = [a for a in attempts if a['key']['task_id'] == judgment['task']
                   and a['key']['language'] == judgment['language']
                   and a['key']['condition'] == judgment.get('condition', 'hybrid-agent')
                   and a['attempt'] == judgment.get('attempt', 1)
                   and (not judgment.get('suite') or a['key']['suite'] == judgment['suite'])]
        if len(matches) != 1:
            raise ValueError('评分必须唯一指定suite、任务、语言和尝试')
        a = matches[0]
        if a['state'] != 'finished' or a['answer_digest'] != judgment['answer_digest']:
            raise ValueError('评分答案绑定错误')
        claims, facts = judgment['claims'], judgment['facts']
        if len(claims) != len(a['result']['answer']['claims']) or len(facts) != len(a['task']['reference_facts']):
            raise ValueError('逐声明或必需事实评分数量错误')
        for item, fields in [(c, ('fact', 'citation')) for c in claims] + [(f, ('score',)) for f in facts]:
            if not item.get('reason') or any(type(item.get(k)) not in (int, float)
                or not math.isfinite(item[k]) or not 0 <= item[k] <= 1 for k in fields):
                raise ValueError('评分需要有限0–1数值和依据')
        na = {}
        if not claims:
            na.update(fact_correctness='No assessable factual claims.', citation_support='No cited factual claims.')
        if not facts:
            na['answer_coverage'] = 'No required reference facts.'
        if judgment['insufficiency'] is None:
            na['insufficiency_handling'] = judgment['insufficiency_reason']
        record = {'key': a['key'], 'attempt': a['attempt'], 'answer_digest': a['answer_digest'],
            'reviewer_type': 'assistant', 'human': 'not_checked',
            'claim_reviews': [dict(c, claim_index=i) for i, c in enumerate(claims)],
            'required_fact_reviews': [dict(f, fact_index=i) for i, f in enumerate(facts)],
            'fact_correctness': sum(c['fact'] for c in claims)/len(claims) if claims else None,
            'citation_support': sum(c['citation'] for c in claims)/len(claims) if claims else None,
            'answer_coverage': sum(f['score'] for f in facts)/len(facts) if facts else None,
            'insufficiency_handling': judgment['insufficiency'],
            'insufficiency_reason': judgment['insufficiency_reason'], 'language_ok': judgment['language_ok'],
            'language_output_is_system_fallback': not a['result']['completed'], 'not_applicable': na,
            'diagnosis': judgment['diagnosis'], 'source_pages_checked': judgment['source_pages'],
            'scope': 'Explicit assistant source review; human=not_checked.'}
        summarize_attempts([r for r in attempts if r['key'] == a['key']], 1, [record])
        target = Path(output)/f'{digest(a["key"])}-attempt-{a["attempt"]:04d}.json'
        if target.exists():
            raise ValueError('评分已存在，不覆盖旧记录')
        prepared.append((target, record))
    if len({p for p, _ in prepared}) != len(prepared):
        raise ValueError('评分记录重复')
    for target, record in prepared:
        atomic_json(target, record)
    return {'saved': len(prepared)}
