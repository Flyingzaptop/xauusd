"""Exact policy-gradient meta-learning in a small known finite SMDP.
No action labels: both inner and outer losses are expected episode returns under
causal policies. Hindsight prices define the simulator, not the actor inputs.
This diagnostic enumerates actions; it is not yet a scalable market algorithm.
"""
from pathlib import Path
import argparse,copy,json,time
import numpy as np,torch
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel,SDPBackend
from model import Actor,Teacher
from synthetic import make_episode,future_view,FEATURES
from provenance import COMMON,fingerprint

H=4;N=12;COST=.05;TASK="direction"

def batch(seed,n=8):
    eps=[make_episode(TASK,seed+j,length=N) for j in range(n)]
    x=torch.tensor(np.stack([e.observations[:-1] for e in eps]));future=torch.tensor(np.stack([[future_view(e,i,H) for i in range(N)] for e in eps]));price=torch.tensor(np.stack([e.opens for e in eps]),dtype=torch.float32)
    end=torch.minimum(torch.arange(N)+H,torch.tensor(N));delta=price[:,end]-price[:,:-1];money=torch.stack([torch.zeros_like(delta),delta-COST,-delta-COST],-1)
    return x,future,money

def expected_return(probs,rewards):
    # Integrates every possible WAIT/OPEN trajectory, with the same blocking rule.
    b,n,_=probs.shape;v=[None]*(n+1);v[n]=torch.zeros(b)
    for i in range(n-1,-1,-1):
        j=min(i+H,n);q=torch.stack([v[i+1],rewards[:,i,1]+v[j],rewards[:,i,2]+v[j]],-1);v[i]=(probs[:,i]*q).sum(-1)
    return v[0]

def probability(actor,params,x):
    shape=x.shape;d,_=functional_call(actor,params,(x.reshape(-1,*shape[-2:]),));return d.probs.reshape(shape[0],shape[1],3)

def scores(teacher,x,f):
    b,n=x.shape[:2];s=teacher(x.reshape(-1,*x.shape[-2:]),f.reshape(-1,*f.shape[-2:])).mean.reshape(b,n,2);return torch.cat([torch.zeros(b,n,1),2*s-1],-1)

def adapted(teacher,seed,steps=3,mode='learned',track=False):
    # The global RNG fork prevents resetting the teacher's own random stream.
    with torch.random.fork_rng():
        torch.manual_seed(seed);actor=Actor(FEATURES,width=16,depth=1)
    params=dict(actor.named_parameters());x,f,money=batch(100000+seed*100)
    if mode=='money':reward=money/10
    elif mode=='money_scaled':reward=money
    elif mode=='constant':reward=torch.tensor([0.,.5,.5]).expand(len(x),N,3)
    else:
        reward=scores(teacher,x,f)
        if mode=='shuffled':reward=reward.roll(1,0)
    if not track:reward=reward.detach()
    for _ in range(steps):
        loss=-expected_return(probability(actor,params,x),reward).mean()
        grads=torch.autograd.grad(loss,tuple(params.values()),create_graph=track,allow_unused=True)
        params={k:v if g is None else v-.25*g for (k,v),g in zip(params.items(),grads)}
    vx,vf,vm=batch(8000000+seed*100,n=32)
    result=expected_return(probability(actor,params,vx),vm).mean()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cycles',type=int,default=80);p.add_argument('--task',choices=['direction','wait','cross_market'],default='direction');a=p.parse_args();TASK=a.task;H=2 if TASK=='wait' else 4;N=3 if TASK=='wait' else 12;torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(571)
    P=Path('/workspace/xauusd/reports/sequential_teacher_v4_20261016')/('meta_gradient_'+TASK);P.mkdir(exist_ok=True);teacher=Teacher(FEATURES,width=16,depth=1,future_steps=H);initial=copy.deepcopy(teacher);opt=torch.optim.Adam(teacher.parameters(),lr=.001);history=[];best=-np.inf;best_state=None;start=time.time();code_hashes=fingerprint(COMMON+["meta_gradient.py"])
    with sdpa_kernel(SDPBackend.MATH):
        for cycle in range(a.cycles):
            loss=-adapted(teacher,11+cycle%3,track=True)/10
            opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1));opt.step()
            row={'cycle':cycle+1,'outer_expected_profit':float(-loss.detach()*10),'teacher_grad_norm':norm}
            if (cycle+1)%8==0:
                values=[float(adapted(teacher,s).detach()) for s in [101,102]];row['selection_profit']=values
                if np.mean(values)>best:best=float(np.mean(values));best_state=copy.deepcopy(teacher.state_dict())
                print(json.dumps(row),flush=True)
            history.append(row);(P/'meta_gradient_history.json').write_text(json.dumps(history,indent=2))
        teacher.load_state_dict(best_state);results={}
        for name,t,mode in [('learned',teacher,'learned'),('initial',initial,'learned'),('constant',teacher,'constant'),('shuffled',teacher,'shuffled'),('money',teacher,'money'),('money_scaled',teacher,'money_scaled')]:
            results[name]=[float(adapted(t,s,mode=mode).detach()) for s in [201,202,203]]
        learned=np.array(results['learned']);passed=all(np.mean(learned-np.array(v))>0 for k,v in results.items() if k!='learned')
        output={'code_hashes':code_hashes,'complete':True,'passed':bool(passed),'method':'exact expected-return differentiable meta-gradient in a small known SMDP','task':TASK,'inner_steps':3,'outer_cycles':a.cycles,'results':results,'student_seeds':[201,202,203],'selection_profit':best,'elapsed':time.time()-start,'limitations':['Exact action enumeration is feasible for this tiny fixed-exit simulator; not yet validated as scalable market training.','Three independent student initializations, simple synthetic directional signal; no claim about markets.']}
        torch.save({'model':teacher.state_dict(),'width':16,'depth':1,'features':FEATURES,'future_steps':H},P/'meta_gradient_teacher.pt');(P/'meta_gradient_gate.json').write_text(json.dumps(output,indent=2));print(json.dumps(output),flush=True)
