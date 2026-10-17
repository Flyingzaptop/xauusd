"""Small real-data student experiment; refuses to bypass synthetic/meta gates."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from gates import assess,source_hashes
from real_data import load_episodes
from environment import Execution,UnpriceableExit
from model import Actor,Teacher
from meta import ConstantTeacher,ShuffledTeacher
from learning import train,rollout

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--out',required=True);p.add_argument('--teacher');p.add_argument('--updates',type=int,default=250);p.add_argument('--mode',choices=['money','constant','learned','shuffled'],default='money');p.add_argument('--gold-only',action='store_true');p.add_argument('--seed',type=int,default=501);p.add_argument('--reports',default='/workspace/xauusd/reports/sequential_teacher_v4_20261016');args=p.parse_args()
    gate=assess(args.reports)
    if not gate['allowed']:raise SystemExit(json.dumps(gate))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(args.seed)
    episodes,meta=load_episodes(args.panel,args.gold_only);features=episodes['train'][0].observations.shape[-1];actor=Actor(features);execution=Execution(120,.25);teacher=None
    if args.mode=='constant':teacher=ConstantTeacher()
    elif args.mode in ['learned','shuffled']:
        if not args.teacher:raise ValueError('A real-input trained teacher checkpoint is required; synthetic teacher is not substituted')
        ck=torch.load(args.teacher,map_location='cpu',weights_only=True);assert ck['features']==features and ck['future_steps']==120
        teacher=Teacher(features,future_steps=120);teacher.load_state_dict(ck['model']);teacher.eval()
        if args.mode=='shuffled':teacher=ShuffledTeacher(teacher)
    factory=lambda u,b:[episodes['train'][(u*b+j)%len(episodes['train'])] for j in range(b)]
    try:
        logs,_=train(actor,factory,execution,args.seed,updates=args.updates,batch_episodes=4,teacher=teacher)
        torch.save({'model':actor.state_dict(),'features':features,'panel':meta,'execution':execution.__dict__,'source_hashes':source_hashes()},out/'actor.pt')
        (out/'learning.json').write_text(json.dumps(logs,indent=2));results={}
        for split in ['dev','test']:
            paths,metrics,_=rollout(actor,episodes[split],execution,args.seed+888,teacher=None)
            results[split]=metrics
            # Full probabilities and every available decision are retained.
            with (out/(split+'_decisions.jsonl')).open('w') as f:
                for day,path in zip(episodes[split],paths):
                    for r in path:f.write(json.dumps({'episode':day.name,**{k:v for k,v in r.items() if k not in ['observation','mask']}})+'\n')
        (out/'results.json').write_text(json.dumps({'complete':True,'mode':args.mode,'seed':args.seed,'gold_only':args.gold_only,'results':results},indent=2))
    except UnpriceableExit as e:
        (out/'unscorable.json').write_text(json.dumps({'complete':False,'reason':str(e),'policy':'No silent exclusion or overnight extension; fix execution-data contract before reporting performance.'},indent=2));raise
