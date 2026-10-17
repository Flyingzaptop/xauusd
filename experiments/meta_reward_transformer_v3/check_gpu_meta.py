import torch,json
from pathlib import Path
from model import MarketTransformer
from objective import centered_teacher_reward

torch.set_num_threads(4);torch.manual_seed(25);teacher=MarketTransformer(teacher=True).cuda().train();parent=torch.load(Path(__file__).parent/'teacher_v2_cycle80.pt',map_location='cpu',weights_only=True);teacher.load_state_dict(parent['model']);del parent
opt=torch.optim.AdamW(teacher.parameters(),lr=3e-5);before=teacher.head.weight.detach().clone();opt.zero_grad(set_to_none=True);error=0.
for k in range(4):
 x=torch.randn(4,222,120,96,device='cuda',dtype=torch.bfloat16);c=torch.randn(4,8,device='cuda',dtype=torch.bfloat16);f=torch.randn(4,120,8,device='cuda',dtype=torch.bfloat16);rng=torch.cuda.get_rng_state()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p=teacher(x,c,f);scores=p.sample();old=p.log_prob(scores);center=p.mean.mean(0)
 rewards=centered_teacher_reward(scores,center,torch.tensor([0,1,2,1],device='cuda'));assert torch.isfinite(rewards).all()
 torch.cuda.set_rng_state(rng)
 with torch.autocast('cuda',dtype=torch.bfloat16):p2=teacher(x,c,f)
 error=max(error,float((old-p2.log_prob(scores)).abs().max().detach()));assert error<.03,error
 loss=-p2.log_prob(scores).sum()*.1/(len(scores)*4);loss.backward();del x,c,f,p,p2,loss,scores
norm=float(torch.nn.utils.clip_grad_norm_(teacher.parameters(),1));assert norm>0;opt.step();assert not torch.equal(before,teacher.head.weight)
print(json.dumps({'passed':True,'warm_start_loaded':True,'replay_error':error,'accumulated_gradient_norm':norm,'four_cycle_update_changed_weights':True,'peak_GB':torch.cuda.max_memory_allocated()/1e9}))
