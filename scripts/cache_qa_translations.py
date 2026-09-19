"""Prepare quality-only query conversions; incurs API costs unless --dry-run."""
import argparse
import uuid
from pathlib import Path
from foliorecall.generation import Bailian
from foliorecall.io import provenance
from foliorecall.qa_runtime import prepare_suite
from foliorecall.qa_storage import atomic_json, digest, output_lock
from foliorecall.qa_translation import cache_translation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    units,identity=prepare_suite(a.suite)
    if a.dry_run:
        print({'queries':len({j['key']['task_id'] for u in units for j in u['jobs']})}); return
    output=Path(a.output)
    with output_lock(output):
        if output.exists():
            raise ValueError('译文缓存目录已存在；失败记录保留，不覆盖')
        provenance(output,identity)
        records={}
        execution_id=uuid.uuid4().hex
        atomic_json(output/'execution.json',{'execution_id':execution_id})
        provider=Bailian(units[0]['agent'])
        for unit in units:
            for job in unit['jobs']:
                task_id=job['key']['task_id']
                if job['key']['language']!='zh' or task_id in records:
                    continue
                provider.context={'suite':identity['suite']['suite_id'],'task_id':task_id,'operation':'translation_cache',
                                  'execution_id':execution_id,'attempt':1,'request_sequence':1}
                atomic_json(output/'current.json',{'task_id':task_id,'state':'running'})
                try:
                    record=cache_translation(job['question'],unit['agent'],provider)
                except Exception as exc:
                    atomic_json(output/'failure.json',{'task_id':task_id,'type':type(exc).__name__,
                                'call':getattr(exc,'record',{}),'scope':'Stopped; check durable ledger before another cache run'})
                    raise
                records[task_id]=record
                atomic_json(output/'records'/f'{digest(task_id)}.json',record)
                atomic_json(output/'translations.json',records)


if __name__ == '__main__':
    main()
