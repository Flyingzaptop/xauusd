"""Big cross-market Transformer on raw M1 windows of 222 instruments -> gold forward returns (15/30/60/120 min,
1-minute entry latency). Variant 'gold' = same model but only the gold token enters the cross-market stack.
Walk-forward folds; early stopping on a validation half-year; predictions saved for the test period."""
import argparse,json,math,time,os
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as Fn
p=argparse.ArgumentParser()
p.add_argument('--data',default='/content/panel');p.add_argument('--out',default='/content/out');p.add_argument('--fold',type=int,default=1)
p.add_argument('--variant',choices=['all222','gold'],default='all222');p.add_argument('--W',type=int,default=240);p.add_argument('--patch',type=int,default=10)
p.add_argument('--dt',type=int,default=128);p.add_argument('--d',type=int,default=512);p.add_argument('--depth',type=int,default=8);p.add_argument('--heads',type=int,default=8)
p.add_argument('--batch',type=int,default=256);p.add_argument('--steps',type=int,default=20000);p.add_argument('--lr',type=float,default=3e-4)
p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--patience',type=int,default=5);p.add_argument('--stride',type=int,default=2)
p.add_argument('--device',default='cuda');p.add_argument('--max-instruments',type=int,default=0);p.add_argument('--seed',type=int,default=1)
a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed);dev=torch.device(a.device)
D=Path(a.data);meta=json.loads((D/'meta.json').read_text());T=np.load(D/'targets.npz');G=meta['G'];gold=meta['gold_index']
panel_np=np.load(D/'panel_kgc.npy',mmap_mode='r')  # [K,G,3]
K=panel_np.shape[0] if not a.max_instruments else a.max_instruments
inst=np.arange(panel_np.shape[0]);
if a.max_instruments:inst=np.r_[gold,[i for i in range(K) if i!=gold][:K-1]];gold_local=int(np.where(inst==gold)[0][0])
else:gold_local=gold
print('loading panel',flush=True);t0=time.time()
panel=torch.empty((G,len(inst),3),dtype=torch.float16,device=dev)
for j,k_ in enumerate(inst):panel[:,j]=torch.nan_to_num(torch.from_numpy(np.array(panel_np[k_])).to(dev),0.,0.,0.)
print('panel',tuple(panel.shape),'%.0fs'%(time.time()-t0),flush=True)
K=panel.shape[1];gns=T['grid_ns'];vol=T['vol60'];HZ=[15,30,60,120]
Y=np.stack([T[f'y{h}'] for h in HZ],1);Ys=Y/(np.nan_to_num(vol,nan=np.nan)[:,None]*np.sqrt(np.array(HZ))[None]+1e-6);Ys=np.clip(Ys,-5,5)
ts=lambda s:np.datetime64(s,'ns').astype(np.int64)
FOLDS={1:('2020-07-01','2021-01-01','2023-01-01'),2:('2022-07-01','2023-01-01','2025-01-01'),3:('2024-07-01','2025-01-01','2026-09-05')}
v0,te0,te1=[ts(x) for x in FOLDS[a.fold]]
minute=((gns//60_000_000_000)%1440);valid=np.isfinite(Ys).all(1)&T['entry_ok']&(np.arange(G)>=a.W)
trn=np.where(valid&(gns<v0-ts('1970-01-02'))&(minute>=360)&(minute<1200))[0][::a.stride]   # 1-day purge before val
val=np.where(valid&(gns>=v0)&(gns<te0-ts('1970-01-02'))&(minute>=479)&(minute<1079))[0][::5]
dec=valid&(gns>=te0)&(gns<te1)&(minute>=479)&(minute<1079)&(((minute-479)%5)==0);tst=np.where(dec)[0]  # bar t closes -> decision t+1 = 08:00..17:55
print('samples train %d val %d test %d'%(len(trn),len(val),len(tst)),flush=True)
Yt=torch.tensor(np.nan_to_num(Ys),dtype=torch.float32,device=dev)
mod=(minute/1440.0);dow=((gns//86_400_000_000_000)+4)%7
clock=torch.tensor(np.stack([np.sin(2*np.pi*mod),np.cos(2*np.pi*mod),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)],1),dtype=torch.float32,device=dev)
ar=torch.arange(a.W,device=dev)
def window(idx):
    i=torch.as_tensor(idx,device=dev);w=panel[(i[:,None]-a.W+1+ar[None])]  # [B,W,K,3], bars up to and incl. t
    return w,clock[i],Yt[i]
class Model(nn.Module):
    def __init__(s):
        super().__init__();P=a.patch;s.np_=a.W//P
        s.pin=nn.Linear(P*3,a.dt);s.tpos=nn.Parameter(torch.zeros(s.np_,a.dt))
        s.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(a.dt,4,a.dt*4,0.1,'gelu',batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        s.tout=nn.Linear(a.dt*2,a.d);s.inst=nn.Embedding(K,a.d);s.cls=nn.Parameter(torch.zeros(1,1,a.d));s.clk=nn.Linear(4,a.d)
        s.cross=nn.TransformerEncoder(nn.TransformerEncoderLayer(a.d,a.heads,a.d*4,0.1,'gelu',batch_first=True,norm_first=True),a.depth,enable_nested_tensor=False)
        s.norm=nn.LayerNorm(a.d);s.head=nn.Linear(a.d,len(HZ))
    def forward(s,w,c,keep):
        B,W,Kk,C=w.shape;w=w[:,:,keep]  # [B,W,k,3]
        k=w.shape[2];x=w.permute(0,2,1,3).reshape(B*k,s.np_,a.patch*3).float()
        h=s.temporal(s.pin(x)+s.tpos);h=torch.cat([h[:,-1],h.mean(1)],-1);h=s.tout(h).reshape(B,k,a.d)+s.inst(keep)[None]
        present=w[...,2].float().amax(1)>0  # instrument had any bar in window
        z=torch.cat([s.cls.expand(B,1,a.d)+s.clk(c)[:,None],h],1);pad=torch.cat([torch.zeros(B,1,dtype=torch.bool,device=w.device),~present],1)
        z=s.cross(z,src_key_padding_mask=pad);return s.head(s.norm(z[:,0]))
keep=torch.arange(K,device=dev) if a.variant=='all222' else torch.tensor([gold_local],device=dev)
m=Model().to(dev);npar=sum(x.numel() for x in m.parameters());print('params %.1fM'%(npar/1e6),flush=True)
opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=0.05,betas=(0.9,0.98))
sched=lambda st:min(1,st/500)*0.5*(1+math.cos(math.pi*min(st,a.steps)/a.steps))
amp=dict(device_type='cuda',dtype=torch.bfloat16) if dev.type=='cuda' else dict(device_type='cpu',dtype=torch.bfloat16,enabled=False)
def predict(idx,bs=128):
    m.eval();out=[]
    with torch.no_grad(),torch.autocast(**amp):
        for j in range(0,len(idx),bs):w,c,_=window(idx[j:j+bs]);out.append(m(w,c,keep).float().cpu())
    m.train();return torch.cat(out).numpy()
def corr(pv,idx):
    yv=Ys[idx];return [float(np.corrcoef(pv[:,j],yv[:,j])[0,1]) for j in range(len(HZ))]
out=Path(a.out)/f'fold{a.fold}_{a.variant}';out.mkdir(parents=True,exist_ok=True);log=[];best=-9;bad=0;t0=time.time();rng=np.random.default_rng(a.seed)
for st in range(1,a.steps+1):
    for g_ in opt.param_groups:g_['lr']=a.lr*sched(st)
    w,c,y=window(rng.choice(trn,a.batch))
    with torch.autocast(**amp):pr=m(w,c,keep).float()
    loss=Fn.huber_loss(pr,y,delta=1.0);opt.zero_grad(set_to_none=True);loss.backward();gn=float(nn.utils.clip_grad_norm_(m.parameters(),1.0));opt.step()
    if st%100==0:print(json.dumps({'step':st,'loss':round(float(loss.detach()),4),'gn':round(gn,3),'sps':round(st*a.batch/(time.time()-t0)),'mem_gb':round(torch.cuda.max_memory_allocated()/1e9,1) if dev.type=='cuda' else 0}),flush=True)
    if st%a.eval_every==0 or st==a.steps:
        pv=predict(val);cv=corr(pv,val);score=float(np.mean(cv));log.append({'step':st,'val_corr':cv,'score':score,'t':time.time()-t0});print('VAL',json.dumps(log[-1]),flush=True)
        if score>best:best=score;bad=0;torch.save(m.state_dict(),out/'best.pt')
        else:
            bad+=1
            if bad>=a.patience:print('early stop',flush=True);break
m.load_state_dict(torch.load(out/'best.pt'));pt=predict(tst);pv=predict(val)
np.savez(out/'preds.npz',test_idx=tst,test_grid_ns=gns[tst],test_pred=pt,val_idx=val,val_pred=pv,horizons=np.array(HZ))
res={'args':vars(a),'params':npar,'best_val_score':best,'val_log':log,'test_corr':corr(pt,tst),'train_samples':len(trn),'steps_done':st,'seconds':time.time()-t0,'samples_seen':st*a.batch}
(out/'result.json').write_text(json.dumps(res,indent=1));print('RESULT',json.dumps({k:res[k] for k in ['best_val_score','test_corr','steps_done','seconds','params']}),flush=True)
