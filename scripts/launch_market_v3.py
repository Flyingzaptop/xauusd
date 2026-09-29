"""Execute fixed independent model/seed fits with separate logs and outputs."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.workers<=5:
        raise ValueError('Use one to five independent one-thread workers')
    cfg=json.loads(Path('configs/market-v3.json').read_text())
    logs=Path(cfg['output'])/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    jobs=[(kind,seed) for kind in ['c','no_transfer','no_open','a','b'] for seed in cfg['seeds']]
    started=time.time()
    def run(job):
        kind,seed=job
        path=logs/f'{kind}-{seed}.log'
        with path.open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,'scripts/run_market_v3.py','--phase','fit',
                                   '--seed',str(seed),'--model',kind],stdout=out,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            raise RuntimeError(f'{kind}/{seed} failed; see {path}')
        return dict(model=kind,seed=seed,log=str(path))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            item=future.result()
            results.append(item)
            print(json.dumps(dict(completed=len(results),total=len(jobs),**item)),flush=True)
    Path('reports/v3/execution.json').write_text(json.dumps(dict(workers=args.workers,
                process_threads=cfg['threads'],wall_seconds=time.time()-started,completed=results),indent=2))


if __name__=='__main__':
    main()
