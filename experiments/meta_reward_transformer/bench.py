import sys,time,json,torch
sys.path.insert(0,'/content/meta_reward');from model import MarketTransformer
r='/content/meta_reward/';torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
out=[]
for batch in [16,32,64,128]:
 try:
  model=MarketTransformer().cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4);x=torch.randn(batch,222,120,96,device='cuda',dtype=torch.bfloat16);c=torch.randn(batch,8,device='cuda',dtype=torch.bfloat16);torch.cuda.reset_peak_memory_stats()
  times=[]
  for k in range(3):
   torch.cuda.synchronize();start=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):p,v=model(x,c);a=p.sample();loss=-(p.log_prob(a)*torch.randn_like(v)).mean()+.5*v.square().mean()
   loss.backward();opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();times.append(time.monotonic()-start)
  out.append({'batch':batch,'seconds':times,'peak_bytes':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in model.parameters())});print(json.dumps(out[-1]),flush=True)
  del model,opt,x,c,p,v,a,loss;torch.cuda.empty_cache()
 except torch.cuda.OutOfMemoryError:
  out.append({'batch':batch,'OOM':True});torch.cuda.empty_cache();break
open(r+'benchmark.json','w').write(json.dumps(out,indent=2))
