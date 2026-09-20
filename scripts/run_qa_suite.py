"""Serial QA suite. Run from the checkout root; no automatic API retries."""
import argparse
import json
import time
from foliorecall.qa_runtime import prepare_suite, execute_plan, provider_factory, suite_jobs


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--translation-cache')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--max-seconds',type=float)
    a = p.parse_args()
    units, identity = prepare_suite(a.suite,a.translation_cache)
    result = execute_plan(a.output,identity,suite_jobs(units,identity['suite']),lambda:provider_factory(units),
        [u['agent'] for u in units],a.resume,a.stop_after,a.max_seconds,a.dry_run,started)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['stopped'] is None else 75


if __name__ == '__main__':
    raise SystemExit(main())
