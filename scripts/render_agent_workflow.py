"""Render bilingual README workflow SVGs without browser-side Mermaid."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(language):
    zh = language == 'zh'
    labels = (
        [['问题与最近成功上下文'], ['规划协调', '提取要求、分配责任与前置依赖'],
         ['取证 A', '第一组责任'], ['取证 B', '另一组责任，按依赖调度'],
         ['共享原页与带范围的候选事实'], ['分析', '比较、计算与答案草稿'],
         ['核验', '对照原页检查声明、遗漏与条件'], ['定向返工', '原责任取证或分析修订'],
         ['末次核验'], ['保留模型核验支持的声明', '单列模型报告的未解决事项'],
         ['保存答案、引用、轨迹与费用']]
        if zh else
        [['Question and successful context'], ['Plan', 'Requirements, responsibilities and dependencies'],
         ['Evidence A', 'First responsibility'], ['Evidence B', 'Another responsibility', 'Honor dependencies'],
         ['Shared original pages and scoped findings'], ['Analyze', 'Compare, calculate and draft'],
         ['Verify', 'Check claims and gaps against original pages'], ['Targeted rework', 'Retrieval or analysis revision'],
         ['Final verification'], ['Keep model-supported claims', 'List model-reported unresolved items'],
         ['Save answer, citations, trace and costs']]
    )
    title = 'Agent执行流程' if zh else 'Agent workflow'
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1040" viewBox="0 0 1000 1040" role="img" aria-labelledby="title desc">
<title id="title">{title}</title>
<desc id="desc">{escape('规划、分工取证、共享证据、分析、核验，最多一次定向返工后保存。' if zh else 'Plan, gather evidence, analyze and verify, with at most one targeted rework before saving.')}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#52796b"/></marker></defs>
<rect width="1000" height="1040" rx="20" fill="#ffffff"/>
<g font-family="Arial, Noto Sans CJK SC, Microsoft YaHei, sans-serif" text-anchor="middle" fill="#19372c">''']
    boxes = [(70,30,560,64), (70,134,560,82), (40,270,290,86), (370,270,290,86),
             (70,410,560,64), (70,522,560,82), (70,660,560,82),
             (704,660,272,82), (704,818,272,82), (70,818,560,82), (70,950,560,64)]
    for i, ((x,y,w,h), lines) in enumerate(zip(boxes, labels)):
        fill = '#edf7f2' if i in (1,5,6,8) else '#f8faf9'
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#9bb8aa" stroke-width="1.5"/>')
        font = 18 if i in (2,3,7,8) else 20
        for j,line in enumerate(lines):
            ty = y+h/2 + (j-(len(lines)-1)/2)*28 + font*.35
            parts.append(f'<text x="{x+w/2}" y="{ty}" font-size="{font}" font-weight="{600 if j == 0 else 400}">{escape(line)}</text>')
    paths = ['M350 94V134','M350 216V242H185V270','M350 242H515V270',
             'M185 356V383H350V410','M515 356V383H350',
             'M350 474V522','M350 604V660','M350 742V818',
             'M630 701H704','M840 742V818','M704 859H630','M350 900V950']
    for path in paths:
        marker = '' if path == 'M515 356V383H350' else ' marker-end="url(#arrow)"'
        parts.append(f'<path d="{path}" fill="none" stroke="#52796b" stroke-width="2.5" stroke-linejoin="round"{marker}/>')
    annotations = [(450,784,'无需返工' if zh else 'No rework'),
                   (802,640,'最多一次返工' if zh else 'At most one rework'),
                   (824,302,'最多两路远程请求并发' if zh else 'Up to two concurrent calls'),
                   (824,332,'存在前置依赖时等待' if zh else 'Wait for dependencies')]
    for x,y,label in annotations:
        parts.append(f'<text x="{x}" y="{y}" font-size="17" fill="#526b60">{escape(label)}</text>')
    parts.append('</g></svg>\n')
    return '\n'.join(parts)


if __name__ == '__main__':
    for language in ('zh','en'):
        (ROOT/f'doc/assets/agent-workflow-{language}.svg').write_text(render(language),encoding='utf-8')
