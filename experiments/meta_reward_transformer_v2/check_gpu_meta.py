import torch,json
from model import MarketTransformer
torch.set_num_threads(4);torch.manual_seed(25);teacher=MarketTransformer(teacher=True).cuda().train();student=MarketTransformer().cuda().train();x=torch.randn(4,222,120,96,device='cuda',dtype=torch.bfloat16);c=torch.randn(4,8,device='cuda',dtype=torch.bfloat16);f=torch.randn(4,120,8,device='cuda',dtype=torch.bfloat16);rng=torch.cuda.get_rng_state()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=teacher(x,c,f);scores=p.sample();old=p.log_prob(scores)
with torch.autocast('cuda',dtype=torch.bfloat16):a,v=student(x,c)
action=a.sample();r=torch.where(action==0,0.,2*scores.gather(1,(action-1).clamp(min=0)[:,None]).squeeze(-1)-1);loss=-(a.log_prob(action)*(r-v.detach())).mean()+.5*(r-v).square().mean();loss.backward()
torch.cuda.set_rng_state(rng)
with torch.autocast('cuda',dtype=torch.bfloat16):p2=teacher(x,c,f)
error=float((old-p2.log_prob(scores)).abs().max().detach());assert error<.03,error
outer=-p2.log_prob(scores).mean()*.1;outer.backward();norm=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1));assert norm>0
print(json.dumps({'passed':True,'teacher_likelihood_replay_error':error,'teacher_gradient_norm':norm,'peak_GB':torch.cuda.max_memory_allocated()/1e9}))
