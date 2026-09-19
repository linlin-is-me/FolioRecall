from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from foliorecall.io import read_rows, read_json, write_rows, write_json
from foliorecall.qa_freeze import reviewed_task, freeze_development, select_repeats
from foliorecall.qa_evaluation import validate_tasks
from foliorecall.agent import load_public_corpus


def fixture(domain='hr', number=0):
    original={'task_id':f'{domain}:{number}','task':domain,'question_en':'What?', 'question_zh':'什么？',
              'reference_answer':'upstream unchanged','category':'numeric','evidence_qrels':{'1':2.,'2':1.}}
    review={'task_id':original['task_id'],'state':'reviewed','source_question_en':'What?',
        'source_question_zh':'什么？','category':'single_page','category_reason':'direct reading, no calculation',
        'answerability':'complete','answerable':True,'reference_facts':['The figure is five.'],
        'evidence_page_ids':['1'],'fact_evidence':[{'fact_index':0,'page_ids':['1'],'location':'table row 1'}],
        'unresolved_requirements':[], 'calculation':None,
        'review':{'assistant':'text_and_image_checked','human':'not_checked','translation':'checked',
                 'unresolved_review_issues':[],'image_pages':['1']},
        'rubric':{'fact_correctness':'source','answer_coverage':{'required_fact_indices':[0]},
                  'citation_support':'page','insufficiency_handling':'missing'}}
    return original,review


class FreezeTests(unittest.TestCase):
    def test_review_scope_and_category(self):
        original,review=fixture(); pages=[{'page_id':'1'},{'page_id':'2'}]
        row=reviewed_task(original,review,pages)
        self.assertEqual(row['category'],'single_page') # two qrels are not multi-page reasoning
        self.assertFalse(row['attributes']['calculation'])
        self.assertEqual(row['upstream'],original)
        partial=deepcopy(review); partial.update(answerable=False,answerability='partial',unresolved_requirements=['missing scope'])
        self.assertFalse(reviewed_task(original,partial,pages)['answerable'])
        for changes in ({'state':'blocked'}, {'category':'numeric'}, {'evidence_page_ids':['3']},
                        {'fact_evidence':[]}, {'source_question_en':'Changed'}, {'answerable':False}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                reviewed_task(original,dict(review,**changes),pages)
        blocked=deepcopy(partial); blocked['review']['unresolved_review_issues']=['missing image']
        with self.assertRaises(ValueError): reviewed_task(original,blocked,pages)

    def test_real_format_freeze_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; specs=[]; originals=[]; reviews=[]
            for d in ('hr','computer_science','finance_en','industrial','pharmaceuticals'):
                folder=data/d; folder.mkdir(parents=True)
                preview=folder/'p.png'; Image.new('RGB',(10,10)).save(preview)
                spec={'name':d,'dataset':'public/'+d,'revision':'r1','pages':2,'english_queries':8}; specs.append(spec)
                pages=[dict(page_id=str(n),doc_id=d,page_number=n,preview=str(preview)) for n in (1,2)]
                write_rows(folder/'pages.jsonl',pages);write_rows(folder/'texts.jsonl',[dict(p,text='text') for p in pages])
                write_json(folder/'source.json',dict(task=spec,dataset=spec['dataset'],revision='r1',candidates=2))
                for n in range(8):
                    o,r=fixture(d,n); originals.append(o); reviews.append(r)
            write_rows(root/'selection.jsonl',originals);write_rows(root/'reviews.jsonl',reviews)
            write_json(root/'datasets.json',{'tasks':specs})
            def freeze(out):return freeze_development(root/'selection.jsonl',root/'reviews.jsonl',root/'datasets.json',data,root/out)
            result=freeze('frozen'); self.assertEqual(result['tasks'],40)
            self.assertEqual(read_json(root/'frozen/run.json')['config']['reviews'],str(root/'reviews.jsonl'))
            self.assertEqual(len(read_json(root/'frozen/repeat-tasks.json')['tasks']),10)
            for d in specs:
                name=d['name'];pages=read_rows(data/name/'pages.jsonl');tasks=root/'frozen'/name/'tasks.jsonl'
                self.assertEqual(len(validate_tasks(tasks,pages)[0]),8)
                self.assertEqual(len(load_public_corpus(tasks.parent/'public-corpus.json',pages)),2)
                for wrong in (pages[:1],list(reversed(pages)),[dict(p,doc_id='wrong') for p in pages]):
                    with self.assertRaises(ValueError): validate_tasks(tasks,wrong)
            with self.assertRaises(ValueError): freeze('frozen')
            write_rows(root/'reviews.jsonl',reviews[:-1]+[reviews[0]])
            with self.assertRaises(ValueError): freeze('bad-review')
            self.assertFalse((root/'bad-review').exists())
            write_rows(root/'reviews.jsonl',reviews)
            source=read_json(data/'hr/source.json');source['revision']='wrong';write_json(data/'hr/source.json',source)
            with self.assertRaises(ValueError): freeze('bad-revision')
            with self.assertRaises(ValueError): validate_tasks(root/'frozen/hr/tasks.jsonl',read_rows(data/'hr/pages.jsonl'))

    def test_repeat_priority_and_string_ties(self):
        rows=[{'task_id':str(n),'category':c,'answerability':'complete'} for n,c in
              [(2,'single_page'),(10,'single_page'),(3,'chart'),(4,'numeric'),(5,'multi_page')]]
        self.assertEqual([r['task_id'] for r in select_repeats(rows)],['10','5'])

    def test_background_notes_are_not_required_facts(self):
        original,review=fixture()
        review['background_notes']=[{'text':'Unrequested background detail.',
                                    'page_ids':['1'],'location':'table footnote',
                                    'required_for_coverage':False}]
        row=reviewed_task(original,review,[{'page_id':'1'}])
        self.assertEqual(row['reference_facts'],['The figure is five.'])
        self.assertEqual(row['rubric']['answer_coverage']['required_fact_indices'],[0])
        self.assertEqual(row['background_notes'],review['background_notes'])
        self.assertEqual(row['upstream'],original)
