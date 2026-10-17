"""Exercise actual student_step from train.py without launching the GPU job."""
import ast,math,json,numpy as np,torch
from pathlib import Path
from objective import entropy_coefficient
module=ast.parse((Path(__file__).parent/'train.py').read_text());nodes=[n for n in module.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CFG' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='student_step')]
ns={'torch':torch,'np':np,'math':math,'entropy_coefficient':entropy_coefficient};exec(compile(ast.Module(body=nodes,type_ignores=[]),'train_subset','exec'),ns)
torch.manual_seed(7);m=torch.nn.Linear(4,4);opt=torch.optim.AdamW(m.parameters(),lr=1e-4);opt.param_groups[0]['updates']=0;old=m.weight.detach().clone();alphas=[]
for i in range(16):
 z=m(torch.randn(32,4));policy=torch.distributions.Categorical(logits=z[:,:3]);value=z[:,3];actions=policy.sample();r=torch.randn(32).clamp(-1,1);v=ns['student_step'](m,opt,r,policy,value,actions);assert all(math.isfinite(vv) for vv in v.values());alphas.append(v['entropy_alpha'])
assert not torch.equal(old,m.weight);assert all(a>=b for a,b in zip(alphas,alphas[1:]))
print(json.dumps({'passed':True,'actual_student_step_updates_weights':True,'alpha_monotonic':True,'all_losses_finite':True}))
