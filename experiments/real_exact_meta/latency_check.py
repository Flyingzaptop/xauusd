"""Same walk-forward gold-only 15-min ridge strategy, but entry delayed by d minutes (Open of t+d) and exit
at Open of first bar >= t+d+15 (same session). Tests whether the edge survives realistic latency."""
import json,numpy as np,pandas as pd
Z=np.load('/home/user/data_prep/intermarket.npz',mmap_mode='r');names=np.array(Z['names']);day=Z['day']
sub=np.arange(len(day))[::3];yr=np.array([int(d[:4]) for d in day[sub]]);dd=day[sub]
gold=np.array([n.startswith('XAUUSD_41:') for n in names]);X=np.asarray(Z['x'][sub][:,gold])
g=pd.read_parquet('/home/user/data_raw/all222/XAUUSD_41.parquet',columns=['timestamp','open']);gts=pd.to_datetime(g.timestamp,utc=True).values.astype('datetime64[ns]').astype(np.int64);go=g.open.values
slot=np.asarray(Z['slot'][sub]);T=(pd.to_datetime(dd).tz_localize('UTC').values.astype('datetime64[ns]').astype(np.int64)+(480+5*slot)*60*10**9)
send=pd.to_datetime(dd).tz_localize('UTC').values.astype('datetime64[ns]').astype(np.int64)+18*3600*10**9
def fwd(d):
    t0=T+d*6*10**10;i=np.searchsorted(gts,t0);i=np.minimum(i,len(gts)-1);ok=(gts[i]-t0<=60*10**9*2)&(gts[i]<send)   # entry within 2 min
    t1=np.minimum(t0+15*6*10**10,send);j=np.minimum(np.searchsorted(gts,t1),len(gts)-1);ok&=gts[j]<=send
    return np.where(ok,(go[j]-go[i])/go[i]*1e4,np.nan),go[i]
out={}
for d in [0,1,2]:
    y,ent=fwd(d);ok=np.isfinite(y);res={}
    for Y in range(2018,2027):
        tr=(yr<Y)&(yr>=2017)&ok;te=(yr==Y)&ok;y0=np.asarray(Z['fwd15'][sub]);trm=tr&np.isfinite(y0)
        mu=np.nanmean(X[trm],0);sd=np.nanstd(X[trm],0)+1e-6;f=lambda m:np.clip(np.nan_to_num((X[m]-mu)/sd),-5,5)
        A=f(trm);W=np.linalg.solve(A.T@A+1e4*A.shape[1]/100*np.eye(A.shape[1]),A.T@(y0[trm]-y0[trm].mean()))  # model fit on d=0 target
        thr=np.quantile(np.abs(A@W),.95);p=f(te)@W;s=np.abs(p)>thr;gr=np.sign(p[s])*y[te][s];nd=len(np.unique(dd[te]))
        res[Y]={'n':int(s.sum()),'gross_bp':float(gr.mean()),'net_usd_day_cost0.25':float(((gr*ent[te][s]/1e4)-.25).sum()/nd),'net_usd_day_cost0.10':float(((gr*ent[te][s]/1e4)-.10).sum()/nd)}
    out[d]=res;print('delay',d,' '.join('%d:%+.2fbp/%+.2f$'%(Y%100,v['gross_bp'],v['net_usd_day_cost0.25']) for Y,v in res.items()),flush=True)
json.dump(out,open('/home/user/xauusd/reports/real_exact_meta/latency_check.json','w'),indent=1)
# --- bias check (appended): share of LONG trades and always-long / random-side controls, delay 1
y,ent=fwd(1);ok=np.isfinite(y);y0=np.asarray(Z['fwd15'][sub]);bias={}
for Y in range(2018,2027):
    tr=(yr<Y)&(yr>=2017)&ok&np.isfinite(y0);te=(yr==Y)&ok
    mu=np.nanmean(X[tr],0);sd=np.nanstd(X[tr],0)+1e-6;f=lambda m:np.clip(np.nan_to_num((X[m]-mu)/sd),-5,5)
    A=f(tr);W=np.linalg.solve(A.T@A+1e4*A.shape[1]/100*np.eye(A.shape[1]),A.T@(y0[tr]-y0[tr].mean()));thr=np.quantile(np.abs(A@W),.95);p=f(te)@W;s=np.abs(p)>thr
    side=np.sign(p[s]);r=y[te][s];g=np.random.default_rng(Y);perm=[float((g.permutation(side)*r).mean()) for _ in range(2000)]
    bias[Y]={'long_share':float((side>0).mean()),'gross_bp':float((side*r).mean()),'always_long_same_times_bp':float(r.mean()),
             'perm_p_value':float((np.array(perm)>=(side*r).mean()).mean()),'mean_sides_same_times_bp':float(np.mean(perm))}
    print(Y,json.dumps({k:round(v,3) for k,v in bias[Y].items()}),flush=True)
json.dump(bias,open('/home/user/xauusd/reports/real_exact_meta/latency_bias_check.json','w'),indent=1)
