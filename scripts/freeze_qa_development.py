"""Consume completed reviews; no automatic answer generation."""
import argparse
import json
from foliorecall.qa_freeze import freeze_development

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True); p.add_argument('--reviews',required=True)
    p.add_argument('--datasets',default='configs/stage4-evaluation.json')
    p.add_argument('--data-root',default='outputs/stage4/data'); p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(freeze_development(a.selection,a.reviews,a.datasets,a.data_root,a.output),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
