"""Prepare at most fifteen fixed-ranking visual cases; never alter official labels."""
import argparse
from pathlib import Path
import sys
import textwrap
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foliorecall.io import provenance, read_json, read_rows, write_json

parser = argparse.ArgumentParser()
parser.add_argument('--summary', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise ValueError('请保留旧复核目录')
summary = read_json(args.summary)
if not summary['complete_matrix']:
    raise ValueError('等待完整固定矩阵，再选择全局最大变化案例')
provenance(output, {'summary': args.summary, 'selection': 'five largest absolute nDCG changes per fixed comparison',
    'scope': 'official labels and images only; assistant visual judgments must be recorded separately'})
data = {}
cases = []
known_hr = {r['query_id'] for r in read_json('outputs/stage4/summary/hr-error-evidence.json')['largest_changes']}
for comparison in summary['comparisons']:
    for row in comparison['review_five']:
        task, query_id = row['task'], row['query_id']
        if task not in data:
            folder = Path('outputs/stage4/data') / task
            data[task] = ({p['page_id']: p for p in read_rows(folder / 'pages.jsonl')}, read_json(folder / 'qrels.json'))
        pages, qrels = data[task]
        grades = qrels[query_id]
        reference, candidate = row['reference_ranking'], row['candidate_ranking']
        position = next(i for i, (a, b) in enumerate(zip(reference, candidate)) if a['page_id'] != b['page_id'])
        positive = max(grades, key=lambda p: (grades[p], p))
        selected = [('Reference', reference[position]['page_id']), ('Candidate', candidate[position]['page_id']), ('Highest official grade', positive)]
        canvas = Image.new('RGB', (1800, 1050), 'white')
        draw = ImageDraw.Draw(canvas)
        heading = f"{comparison['name']} | {task}/{query_id} | nDCG delta {row['delta']['nDCG@10']:+.6f} | first changed rank {position+1}"
        draw.text((15, 10), heading, fill='black')
        draw.multiline_text((15, 35), '\n'.join(textwrap.wrap(row['query'], 140)), fill='black', spacing=4)
        evidence = []
        for column, (side, page_id) in enumerate(selected):
            page = pages[page_id]
            with Image.open(page['preview']) as source:
                image = source.convert('RGB')
                image.thumbnail((580, 920))
                canvas.paste(image, (column*600+10, 120))
            label = f"{side}: id={page_id}, physical page={page['page_number']}, grade={grades.get(page_id, 0)}"
            draw.multiline_text((column*600+10, 80), '\n'.join(textwrap.wrap(label, 72)), fill='black')
            evidence.append(dict(side=side, official_grade=grades.get(page_id, 0), page=page))
        name = f'{len(cases)+1:02d}-{comparison["name"]}-{task}-{query_id}'
        image_path = output / (name + '.png')
        canvas.save(image_path)
        cases.append(dict(row, comparison=comparison['name'], first_changed_rank=position+1,
            official_qrels=grades, selected_page_evidence=evidence, image=str(image_path),
            existing_hr_evidence='outputs/stage4/summary/hr-error-evidence.json' if task == 'hr' and query_id in known_hr and comparison['name'] == 'distilled94-vs-ml' else None))
write_json(output / 'cases.json', cases)
print(f'{len(cases)} cases prepared; images require visual review; labels unchanged')
