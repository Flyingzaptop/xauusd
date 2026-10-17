"""Small outer REINFORCE experiment with reset students and selected-action credit.
No persistent student is used to rank candidate teachers. CPU gate, not a market run.
"""
from pathlib import Path
import argparse,copy,json,time
import numpy as np,torch
from model import Actor,Teacher
from synthetic import make_episode,FEATURES
from environment import Execution
from learning import rollout,train
from provenance import COMMON,fingerprint

class ConstantTeacher(torch.nn.Module):
    def forward(self,x,f):
        a=torch.full((len(x),2),2.,device=x.device);return torch.distributions.Beta(a,a)
class ShuffledTeacher(torch.nn.Module):
    def __init__(self,teacher):super().__init__();self.teacher=teacher
    def forward(self,x,f):
        d=self.teacher(x,f);ix=torch.roll(torch.arange(len(x)),1);return torch.distributions.Beta(d.concentration1[ix],d.concentration0[ix])

def student_trial(teacher,seed,inner_updates,track=False,task='direction'):
    torch.manual_seed(seed);actor=Actor(FEATURES);execution=Execution(4,.05)
    factory=lambda u,b:[make_episode(task,200000+seed*10000+u*b+j,length=16) for j in range(b)]
    val=[make_episode(task,8000000+seed*1000+j,length=16) for j in range(64)]
    _,m,_=rollout(actor,val,execution,600+seed);before=float(np.mean([v['net'] for v in m]))
    logs,credit=train(actor,factory,execution,seed,updates=inner_updates,batch_episodes=8,teacher=teacher,track_teacher=track)
    _,m,_=rollout(actor,val,execution,600+seed);after=float(np.mean([v['net'] for v in m]))
    return after-before,after,credit,actor

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cycles',type=int,default=24);parser.add_argument('--inner-updates',type=int,default=16);args=parser.parse_args()
    P=Path('/workspace/xauusd/reports/sequential_teacher_v4_20261016');torch.set_num_threads(2);torch.manual_seed(571)
    teacher=Teacher(FEATURES,future_steps=4);initial=copy.deepcopy(teacher);opt=torch.optim.AdamW(teacher.parameters(),lr=3e-4);history=[];best=-np.inf;baseline=0.;best_state=None;start=time.time();code_hashes=fingerprint(COMMON+["meta.py"])
    for cycle in range(args.cycles):
        gain,score,credit,_=student_trial(teacher,11+cycle%3,args.inner_updates,True)
        advantage=gain-baseline;baseline=.9*baseline+.1*gain
        if not credit:raise RuntimeError('Teacher received no executed-entry credit')
        loss=-torch.stack(credit).sum()*float(np.clip(advantage,-10,10))/len(credit)
        opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1));opt.step()
        record={'cycle':cycle+1,'student_seed':11+cycle%3,'external_gain':gain,'after_net':score,'policy_loss':float(loss.detach()),'teacher_grad_norm':norm}
        # Evaluate the UPDATED teacher by actually training fresh reset pupils.
        if (cycle+1)%4==0:
            scores=[student_trial(teacher,s,args.inner_updates)[1] for s in [101,102]];record['selection_pupil_scores']=scores
            if np.mean(scores)>best:best=float(np.mean(scores));best_state=copy.deepcopy(teacher.state_dict());record['selected']=True
        history.append(record);(P/'meta_history.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
    if best_state is None:raise RuntimeError('No teacher evaluated')
    teacher.load_state_dict(best_state);torch.save({'model':best_state,'features':FEATURES,'future_steps':4,'inner_updates':args.inner_updates},P/'candidate_synthetic_teacher.pt')
    results={}
    for name,t in [('learned',teacher),('initial',initial),('constant',ConstantTeacher()),('shuffled',ShuffledTeacher(teacher)),('money',None)]:
        results[name]=[student_trial(t,s,args.inner_updates)[1] for s in [201,202,203]]
    # Small-gate criterion; deployment would additionally require held-out market evidence.
    learned=np.array(results['learned']);comparators=['initial','constant','shuffled','money']
    passed=all(float(np.mean(learned-np.array(results[k])))>0 for k in comparators) and all(np.sum(learned>np.array(results[k]))>=2 for k in comparators)
    output={'code_hashes':code_hashes,'complete':True,'passed':bool(passed),'seed_order':[201,202,203],'results':results,'selection_score':best,'cycles':args.cycles,'inner_updates':args.inner_updates,'elapsed':time.time()-start,'meaning':'Small synthetic teacher-transfer gate; not evidence of trading profitability.'}
    (P/'meta_gate.json').write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)
