"""Pretrain (next-second summary) + finetune (multi-horizon returns) + evaluation with costs. Months are passed as npz lists."""
import argparse,json,time,math,os,glob,numpy as np,torch,torch.nn.functional as F
import sys;sys.path.insert(0,os.path.dirname(__file__))
from model import SeqModel;from data import Market,HZ
p=argparse.ArgumentParser()
p.add_argument('--train',nargs='+',required=True);p.add_argument('--val',nargs='+',required=True);p.add_argument('--test',nargs='+',default=[])
p.add_argument('--backend',default='attn');p.add_argument('--d',type=int,default=256);p.add_argument('--depth',type=int,default=8);p.add_argument('--heads',type=int,default=8)
p.add_argument('--Lf',type=int,default=3600);p.add_argument('--Lc',type=int,default=1440);p.add_argument('--K',type=int,default=32);p.add_argument('--batch',type=int,default=8)
p.add_argument('--pre_steps',type=int,default=2000);p.add_argument('--ft_steps',type=int,default=4000);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--eval_every',type=int,default=500)
p.add_argument('--device',default='cuda');p.add_argument('--out',default='/content/seq_out');p.add_argument('--eval_stride',type=int,default=30)
a=p.parse_args();dev=torch.device(a.device);os.makedirs(a.out,exist_ok=True);torch.manual_seed(1)
def pairs(lst):
    b=[f for f in lst if 'BTCUSDT' in os.path.basename(f)];e=[f for f in lst if 'ETHUSDT' in os.path.basename(f)];return b,e
TR=Market(*pairs(a.train),device=dev);VA=Market(*pairs(a.val),device=dev);TE=Market(*pairs(a.test),device=dev) if a.test else None
warm=max(a.Lf,a.Lc*60+120)+1;HMAX=max(HZ)+2
m=SeqModel(d=a.d,depth=a.depth,heads=a.heads,backend=a.backend,Lc=a.Lc,Lf=a.Lf).to(dev)
print('params %.1fM'%(sum(x.numel() for x in m.parameters())/1e6),flush=True)
amp=dict(device_type='cuda',dtype=torch.bfloat16) if dev.type=='cuda' else dict(device_type='cpu',enabled=False)
def anchors(M,n,rng):return torch.tensor(rng.integers(warm,M.T-HMAX,n),device=dev)
def evaluate(M,stride):
    m.eval();gs=torch.arange(warm,M.T-HMAX,stride,device=dev);P=[];Y=[];V=[]
    with torch.no_grad(),torch.autocast(**amp):
        for i in range(0,len(gs),a.batch):
            b=M.batch(gs[i:i+a.batch],a.Lf,a.Lc,a.K);_,y=m(b);P.append(y.float().cpu());Y.append(b['y'].cpu());V.append(b['vol'].cpu())
    m.train();P=torch.cat(P).numpy();Y=torch.cat(Y).numpy();V=torch.cat(V).numpy();res={}
    for si,sn in enumerate(['BTC','ETH']):
        for hi,h in enumerate(HZ):
            p_=P[:,si,hi];y_=Y[:,si,hi];ic=float(np.corrcoef(p_,y_)[0,1]);thr=np.quantile(np.abs(p_),.95);sel=np.where(np.abs(p_)>thr)[0]
            step=max(1,h//stride);keep=[];last=-10**9
            for k in sel:
                if k>=last+step:keep.append(k);last=k
            g_=np.sign(p_[keep])*y_[keep]
            res[f'{sn}_{h}s']={'ic':ic,'top5_gross_bp':float(g_.mean()) if len(keep) else 0.0,'n':len(keep),**{f'net{c}':float(g_.mean()-c) if len(keep) else 0.0 for c in (4,7,10)}}
    return res
def scaled(b):
    hs=torch.tensor(HZ,device=dev,dtype=torch.float32).sqrt();return (b['y']/(b['vol'].unsqueeze(-1)*hs)).clamp(-5,5)
opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=0.05);rng=np.random.default_rng(1);log=[];t0=time.time()
def lr_at(st,tot):return a.lr*min(1,st/200)*0.5*(1+math.cos(math.pi*min(st,tot)/tot))
for phase,steps in (('pretrain',a.pre_steps),('finetune',a.ft_steps)):
    best=-1e9
    for st in range(1,steps+1):
        for g_ in opt.param_groups:g_['lr']=lr_at(st,steps)*(1 if phase=='pretrain' else 0.5)
        b=TR.batch(anchors(TR,a.batch,rng),a.Lf,a.Lc,a.K)
        with torch.autocast(**amp):
            pre,y=m(b)
            lp=F.huber_loss(pre.float(),b['pre_t'].transpose(1,2).float())
            ly=F.huber_loss(y.float(),scaled(b))
            loss=lp if phase=='pretrain' else ly+0.1*lp
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        if st%50==0:print(json.dumps({'phase':phase,'step':st,'lp':round(float(lp),4),'ly':round(float(ly),4),'t':round(time.time()-t0)}),flush=True)
        if phase=='finetune' and (st%a.eval_every==0 or st==steps):
            r=evaluate(VA,a.eval_stride);score=float(np.mean([v['ic'] for v in r.values()]));log.append({'step':st,'val':r,'score':score})
            print('VAL',st,json.dumps({k:round(v['ic'],4) for k,v in r.items()}),flush=True)
            if score>best:best=score;torch.save(m.state_dict(),a.out+'/best.pt')
    if phase=='pretrain':torch.save(m.state_dict(),a.out+'/pretrained.pt')
m.load_state_dict(torch.load(a.out+'/best.pt'));final={'val':evaluate(VA,a.eval_stride)}
if TE:final['test']=evaluate(TE,a.eval_stride)
json.dump({'args':vars(a),'log':log,'final':final},open(a.out+'/result.json','w'),indent=1)
for k,v in final.items():
    print('FINAL',k);[print(f'  {kk}: ic {vv["ic"]:+.4f} top5 gross {vv["top5_gross_bp"]:+.2f} bp net@4 {vv["net4"]:+.2f} net@10 {vv["net10"]:+.2f} n {vv["n"]}') for kk,vv in v.items()]
