"""Reference baseline for SeqModel: LightGBM on window aggregates of the same per-second/per-minute summaries, same anchors,
labels, vol-scaling and trade evaluation (val-chosen thresholds applied to test/final).
usage: baseline.py --train ... --val ... [--test ...] [--final ...] --out result.json"""
import argparse,json,os,numpy as np,torch,lightgbm as lgb
from data import Market,HZ
p=argparse.ArgumentParser();[p.add_argument(f'--{k}',nargs='+',default=[]) for k in ('train','val','test','final')]
p.add_argument('--stride',type=int,default=30);p.add_argument('--eval_stride',type=int,default=60);p.add_argument('--out',required=True);a=p.parse_args()
Lf,Lc=3600,1440;warm=Lf+Lc*60+121;HMAX=max(HZ)+2;W=[10,60,300,1800,3600]
def pairs(l):return [f for f in l if 'BTCUSDT' in f],[f for f in l if 'ETHUSDT' in f]
def feats(M,stride):
    g=torch.arange(warm,M.T-HMAX,stride);X=[];Y=[];V=[]
    cs=[torch.cat([torch.zeros(1,6,dtype=torch.float64),t['ssum'].double().cumsum(0)]) for t in M.t]   # prefix sums of summaries
    for si,t in enumerate(M.t):
        for w in W:
            s=(cs[si][g+1]-cs[si][g+1-w])/w;X.append(s[:,[0,1,2,4,5]]);X.append(s[:,3:4]*w)   # mean summaries + summed return
        m=t['msum'];mr=((M.s0+g-Lf+1)//60-1-t['m0']).clamp(0,len(m)-1);mc=torch.cat([torch.zeros(1,6),m.cumsum(0)])
        for wm in (60,360,1440):X.append((mc[mr+1]-mc[(mr+1-wm).clamp(min=0)])[:,[1,3]])                  # flow and return over 1h/6h/24h before
        X.append(((M.s0+g)%86400/3600.).view(-1,1).double())
    for i in range(0,len(g),4096):
        b=M.batch(g[i:i+4096],Lf,Lc,1);Y.append(b['y']);V.append(b['vol'])
    return torch.cat([x.double() for x in X],1).numpy(),torch.cat(Y).numpy(),torch.cat(V).numpy()
def ev(P,Y,stride,thr=None):
    res={}
    for si,sn in enumerate(['BTC','ETH']):
        for hi,h in enumerate(HZ):
            p_=P[:,si,hi];y_=Y[:,si,hi];k_=f'{sn}_{h}s';th=thr[k_]['thr'] if thr else float(np.quantile(np.abs(p_),.95))
            sel=np.where(np.abs(p_)>th)[0];step=max(1,h//stride);keep=[];last=-10**9
            for k in sel:
                if k>=last+step:keep.append(k);last=k
            g_=np.sign(p_[keep])*y_[keep];res[k_]={'ic':float(np.corrcoef(p_,y_)[0,1]),'thr':th,'n':len(keep),'top5_gross_bp':float(g_.mean()) if keep else 0.,**{f'net{c}':float(g_.mean()-c) if keep else 0. for c in (4,7,10)}}
    return res
Xt,Yt,Vt=feats(Market(*pairs(a.train)),a.stride);print('train',Xt.shape,flush=True)
hs=np.sqrt(np.array(HZ));models={}
for si in range(2):
    for hi in range(len(HZ)):
        z=np.clip(Yt[:,si,hi]/(Vt[:,si]*hs[hi]),-5,5)
        models[si,hi]=lgb.train(dict(objective='huber',learning_rate=0.03,num_leaves=31,min_data_in_leaf=2000,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10,verbose=-1),lgb.Dataset(Xt,z),400)
out={}
for nm in ('val','test','final'):
    if not getattr(a,nm):continue
    X,Y,V=feats(Market(*pairs(getattr(a,nm))),a.eval_stride);P=np.zeros_like(Y)
    for (si,hi),mdl in models.items():P[:,si,hi]=mdl.predict(X)
    out[nm]=ev(P,Y,a.eval_stride,out.get('val'))
    print(nm,json.dumps({k:(round(v['ic'],4),round(v['net4'],2)) for k,v in out[nm].items()}),flush=True)
json.dump(out,open(a.out,'w'),indent=1)
