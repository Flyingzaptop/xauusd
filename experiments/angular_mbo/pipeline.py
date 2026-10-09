"""Run unattended. Stop on any failed stage; never fabricate downstream results."""
import logging
import argparse
import subprocess
import sys
import traceback
from config import *

def run(from_stage='prepare'):
    OUT.mkdir(parents=True,exist_ok=True)
    stages=[('prepare',['prepare.py']),('verification',['verify.py']),
            ('smoke_training',['train.py','--smoke']),('training',['train.py']),('report',['report.py'])]
    if from_stage!='prepare':
        manifest=json.loads((OUT/'manifest.json').read_text())
        verification=json.loads((OUT/'verification.json').read_text())
        assert manifest['status']=='prepared' and verification['status']=='passed'
        stages=stages[[name for name,_ in stages].index(from_stage):]
    try:
        for name,args in stages:
            dump(OUT/'pipeline.json',dict(status='running',stage=name))
            with (OUT/f'{name}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).parent/args[0]),*args[1:]],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        dump(OUT/'pipeline.json',dict(status='complete',stage='report'))
    except Exception:
        dump(OUT/'pipeline.json',dict(status='failed',stage=name,error=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-stage',choices=['prepare','verification','smoke_training','training','report'],default='prepare')
    run(parser.parse_args().from_stage)
