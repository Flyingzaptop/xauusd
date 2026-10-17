from pathlib import Path
import argparse,json,time
import torch,numpy as np
from environment import Execution,backward_oracle
from model import Actor
from synthetic import make_episode,FEATURES
from learning import train,rollout
from provenance import COMMON,fingerprint
P=Path('/workspace/xauusd/reports/sequential_teacher_v4_20261016');P.mkdir(exist_ok=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=250);parser.add_argument('--tasks',nargs='+',default=['direction','wait','cross_market','null']);parser.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);args=parser.parse_args();torch.set_num_threads(2);results=[];start=time.time();code_hashes=fingerprint(COMMON+["run_gates.py"])
    for task in args.tasks:
        for seed in args.seeds:
            torch.manual_seed(seed);actor=Actor(FEATURES);execution=Execution(2 if task=='wait' else 8,.05)
            factory=lambda u,b,task=task,seed=seed:[make_episode(task,seed*100000+u*b+j) for j in range(b)]
            logs,_=train(actor,factory,execution,seed,updates=args.updates)
            episodes=[make_episode(task,9000000+seed*1000+j) for j in range(128)]
            paths,m,_=rollout(actor,episodes,execution,seed+500);_,greedy,_=rollout(actor,episodes,execution,seed+500,greedy=True)
            net=np.array([v['net'] for v in m]);oracle=np.array([backward_oracle(e,execution)[0][0] for e in episodes]);ratio=float(net.mean()/oracle.mean())
            if task=='wait':passgate=float(np.mean([p[0]['action']==0 for p in paths]))>=.8 and ratio>=.7
            elif task=='null':passgate=float(net.mean()-1.96*net.std(ddof=1)/np.sqrt(len(net)))<=0
            else:passgate=ratio>=.65
            record={'task':task,'seed':seed,'updates':args.updates,'mean_net':float(net.mean()),'net_standard_error':float(net.std(ddof=1)/np.sqrt(len(net))),'oracle_mean':float(oracle.mean()),'oracle_ratio':ratio,'first_WAIT_fraction':float(np.mean([p[0]['action']==0 for p in paths])),'greedy_mean_net':float(np.mean([v['net'] for v in greedy])),'passed':passgate,'final_loss':logs[-1]}
            results.append(record);(P/'synthetic_gates.json').write_text(json.dumps({'code_hashes':code_hashes,'complete':False,'results':results,'elapsed':time.time()-start},indent=2));print(json.dumps(record),flush=True)
            torch.save({'model':actor.state_dict(),'features':FEATURES,'task':task,'seed':seed},P/f'synthetic_{task}_{seed}.pt')
            (P/f'synthetic_{task}_{seed}_curves.json').write_text(json.dumps(logs))
    (P/'synthetic_gates.json').write_text(json.dumps({'code_hashes':code_hashes,'complete':True,'passed':all(r['passed'] for r in results),'results':results,'elapsed':time.time()-start},indent=2))
