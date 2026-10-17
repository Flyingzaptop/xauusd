"""Diagnostic: is cross_market solvable by the exact-meta inner learner within its budget,
for ANY reward? Compares money reward vs inner steps, plus DP oracle. Fresh seeds 301-305."""
import sys,json,torch,numpy as np
sys.path.insert(0,'/workspace/xauusd/experiments/sequential_teacher_v4')
import meta_gradient as mg
from model import Actor
from synthetic import FEATURES
from torch.func import functional_call
from torch.nn.attention import sdpa_kernel,SDPBackend
torch.set_num_threads(4);torch.backends.mha.set_fastpath_enabled(False)
def oracle(money):
    b,n,_=money.shape;v=[None]*(n+1);v[n]=torch.zeros(b)
    for i in range(n-1,-1,-1):
        j=min(i+mg.H,n);v[i]=torch.stack([v[i+1],money[:,i,1]+v[j],money[:,i,2]+v[j]],-1).max(-1).values
    return v[0].mean()
def run(task,steps,scale,seed,lr=.25):
    with torch.random.fork_rng():
        torch.manual_seed(seed);actor=Actor(FEATURES,width=16,depth=1)
    p=dict(actor.named_parameters());x,f,m=mg.batch(100000+seed*100);r=m/scale
    for _ in range(steps):
        loss=-mg.expected_return(mg.probability(actor,p,x),r).mean();g=torch.autograd.grad(loss,tuple(p.values()),allow_unused=True)
        p={k:v if gg is None else (v-lr*gg).detach().requires_grad_() for (k,v),gg in zip(p.items(),g)}
    vx,vf,vm=mg.batch(8000000+seed*100,n=32);return float(mg.expected_return(mg.probability(actor,p,vx),vm).mean()),float(oracle(vm))
if __name__=="__main__":
 out={}
 with sdpa_kernel(SDPBackend.MATH):
  for task in ['direction','cross_market']:
    mg.TASK=task;mg.H=4;mg.N=12
    for steps in [3,10,30,100,300]:
      for scale in [10,1]:
        res=[run(task,steps,scale,s) for s in range(301,306)]
        out[f'{task}/steps{steps}/money_div{scale}']=[round(np.mean([a for a,_ in res]),3),round(np.mean([b for _,b in res]),3)]
        print(task,steps,scale,'mean_profit=%.3f oracle=%.3f'%(np.mean([a for a,_ in res]),np.mean([b for _,b in res])),flush=True)
 json.dump(out,open('/workspace/xauusd/experiments/claude_diag/inner_budget.json','w'),indent=1)
