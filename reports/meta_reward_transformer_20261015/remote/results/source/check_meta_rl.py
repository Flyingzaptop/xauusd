"""Small true score-function outer RL test; no differentiation through student."""
import torch,json,numpy as np
from torch import nn
torch.set_num_threads(1);torch.manual_seed(19)
teacher=nn.Linear(2,4);nn.init.zeros_(teacher.weight);nn.init.zeros_(teacher.bias);opt=torch.optim.Adam(teacher.parameters(),lr=.015)
xmeta=torch.tensor([[-1.],[1.]]).repeat(64,1);y=xmeta[:,0];history=[];baseline=0.
for outer in range(600):
 student=nn.Linear(1,3,bias=False);nn.init.zeros_(student.weight);sopt=torch.optim.SGD(student.parameters(),lr=.5)
 logs=[]
 for k in range(3):
  x=torch.where(torch.rand(96,1)>.5,1.,-1.);z=torch.cat([x,x],1);v=teacher(z);dist=torch.distributions.Beta(1+torch.nn.functional.softplus(v[:,:2]),1+torch.nn.functional.softplus(v[:,2:]));scores=dist.sample();logs.append(dist.log_prob(scores).sum(-1).sum())
  policy=torch.distributions.Categorical(logits=student(x));a=policy.sample();r=torch.where(a==0,0.,2*scores.gather(1,(a-1).clamp(min=0)[:,None]).squeeze(-1)-1).detach()
  loss=-(policy.log_prob(a)*r).mean();sopt.zero_grad();loss.backward();sopt.step()
 with torch.no_grad():
  p=student(xmeta).softmax(-1);gain=((p[:,1]-p[:,2])*y).mean().item()
 advantage=gain-baseline;baseline=.95*baseline+.05*gain
 tl=-sum(logs)*advantage/(96*3);opt.zero_grad();tl.backward();norm=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),10));opt.step();history.append(gain)
first=float(np.mean(history[:50]));last=float(np.mean(history[-100:]));assert last>first+.02,(first,last)
# A constant external reward has zero score-function advantage and zero gradient.
assert norm>0
result={'passed':True,'first50_mean_independent_return_gain':first,'last100_mean_independent_return_gain':last,'teacher_gradient_norm_last':norm,'outer_updates':600,'student_updates_per_outer':3,'note':'Toy test of feedback direction, not evidence about market predictability.'}
print(json.dumps(result))
