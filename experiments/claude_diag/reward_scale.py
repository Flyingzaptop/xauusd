"""Diagnostic: money reward with different scales at the exact-meta budget (3 inner steps),
on the original test seeds 201-203 (to compare with suite numbers) and fresh 301-305."""
import sys,json,numpy as np,torch
sys.path.insert(0,'/workspace/xauusd/experiments');sys.path.insert(0,'/workspace/xauusd/experiments/sequential_teacher_v4')
from claude_diag.inner_budget import run,mg
from torch.nn.attention import sdpa_kernel,SDPBackend
out={}
with sdpa_kernel(SDPBackend.MATH):
  for task in ['direction','cross_market']:
    mg.TASK=task;mg.H=4;mg.N=12
    for scale in [10,7,5,3,2,1]:
      for name,seeds in [('seen201-203',[201,202,203]),('fresh301-305',range(301,306))]:
        v=[run(task,3,scale,s)[0] for s in seeds];out[f'{task}/money_div{scale}/{name}']=[round(x,3) for x in v]
        print(task,'money/%d'%scale,name,'mean=%.3f'%np.mean(v),[round(x,2) for x in v],flush=True)
json.dump(out,open('/workspace/xauusd/experiments/claude_diag/reward_scale.json','w'),indent=1)
