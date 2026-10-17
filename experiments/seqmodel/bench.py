"""GPU throughput/memory benchmark for SeqModel configs (used to estimate compute units before a full run).
usage: bench.py BTC.npz ETH.npz"""
import sys,time,json,numpy as np,torch,torch.nn.functional as F
from model import SeqModel;from data import Market
dev=torch.device('cuda');M=Market([sys.argv[1]],[sys.argv[2]],device=dev);rng=np.random.default_rng(0)
cfgs=[dict(backend=b,d=d,depth=dp,batch=bs,Lf=3600,Lc=1440) for b in (sys.argv[3].split(',') if len(sys.argv)>3 else ('attn','mamba')) for d,dp,bs in ((256,8,8),(384,12,8))]
for c in cfgs:
    try:
        m=SeqModel(d=c['d'],depth=c['depth'],heads=8,backend=c['backend'],Lc=c['Lc'],Lf=c['Lf']).to(dev);opt=torch.optim.AdamW(m.parameters(),1e-4)
        warm=c['Lf']+c['Lc']*60+121;torch.cuda.reset_peak_memory_stats();ts=[]
        for st in range(25):
            torch.cuda.synchronize();t=time.time()
            g=torch.tensor(rng.integers(warm,M.T-4000,c['batch']),device=dev);b=M.batch(g,c['Lf'],c['Lc'],32)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pre,y=m(b);loss=F.huber_loss(pre.float(),b['pre_t'].transpose(1,2).float())+y.float().pow(2).mean()
            opt.zero_grad();loss.backward();opt.step();torch.cuda.synchronize();ts.append(time.time()-t)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            m.eval();torch.cuda.synchronize();t=time.time()
            for _ in range(5):m(M.batch(torch.tensor(rng.integers(warm,M.T-4000,64),device=dev),c['Lf'],c['Lc'],32))
            torch.cuda.synchronize();ev=(time.time()-t)/5
        print(json.dumps({**c,'params_M':round(sum(x.numel() for x in m.parameters())/1e6,1),'step_s':round(float(np.median(ts[5:])),3),'eval64_s':round(ev,3),'peak_GB':round(torch.cuda.max_memory_allocated()/1e9,1)}),flush=True)
        del m,opt;torch.cuda.empty_cache()
    except Exception as e:print(json.dumps({**c,'error':repr(e)[:300]}),flush=True);torch.cuda.empty_cache()
