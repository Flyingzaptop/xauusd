"""Fixed v3.1 fit matrix; independent single-thread processes, optional resume."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=9)
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    if not 1<=args.workers<=9:raise ValueError('Use one to nine workers')
    cfg=json.loads(Path('configs/market-v31.json').read_text())
    logs=Path(cfg['output'])/'logs'; logs.mkdir(parents=True,exist_ok=True)
    jobs=[(m,s) for m in ['c','no_transfer','no_open','a','b'] for s in cfg['seeds']]
    start=time.time(); outcomes=[]
    def run(job):
        model,seed=job; logfile=logs/f'{model}-{seed}.log'
        cmd=[sys.executable,'scripts/fit_market_v31.py','--model',model,'--seed',str(seed)]
        checkpoint_dir=Path(cfg['output'])/'checkpoints'/f'{model}-{seed}'
        if args.resume and (checkpoint_dir/'last.pt').exists():cmd.append('--resume')
        elif args.resume and Path(f'reports/v31/fit-{model}-{seed}.json').exists():
            raise RuntimeError(f'Fit report exists but recovery checkpoint is missing: {model}/{seed}')
        with logfile.open('a' if args.resume else 'w',encoding='utf-8') as out:
            result=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return dict(model=model,seed=seed,returncode=result.returncode,log=str(logfile))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            outcome=f.result(); outcomes.append(outcome)
            print(json.dumps(dict(finished=len(outcomes),total=15,**outcome)),flush=True)
    report=dict(workers=args.workers,resumed=args.resume,wall_seconds=time.time()-start,jobs=outcomes,
                complete=all(r['returncode']==0 for r in outcomes))
    Path('reports/v31/execution.json').write_text(json.dumps(report,indent=2))
    if not report['complete']:raise SystemExit('One or more fits failed; inspect preserved logs')


if __name__=='__main__':main()
