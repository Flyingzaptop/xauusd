"""Lead-lag between CME GCQ6 (MBO ts_recv, Aurora) and broker XAUUSD depth (recorder timestamps), 7-24 Jul 2026, 01-20 UTC."""
import json,numpy as np,pandas as pd
C=np.load('/home/user/mbo_out/cme_bbo_jul.npz');B=np.load('/home/user/mbo_out/broker_bbo_jul.npz')
cm=(C['bid']+C['ask'])/2;bm=(B['bid']+B['ask'])/2;cts=C['ts'];bts=B['ts']
def mid(ts,m,t):
    i=np.searchsorted(ts,t,side='right')-1;return np.where(i>=0,m[np.maximum(i,0)],np.nan)
def bq(t):
    i=np.searchsorted(bts,t,side='right')-1;return B['bid'][i],B['ask'][i]
MS=1_000_000
start=pd.Timestamp('2026-07-07 01:00',tz='UTC').value;end=pd.Timestamp('2026-07-24 19:00',tz='UTC').value
g=np.arange(start,end,100*MS);h=(g//10**9)%86400/3600;g=g[(h>=1)&(h<20)&(pd.to_datetime(g,utc=True).dayofweek<5)]
# exclude grid points where the broker feed is stale (no update for >2 s)
lastb=bts[np.searchsorted(bts,g,side='right')-1];g=g[(g-lastb)<2000*MS]
out={}
# 1) cross-correlation of 500 ms changes
for D in (100,500):
    rc=mid(cts,cm,g+D*MS)-mid(cts,cm,g);res={}
    for lag in range(-300,1501,20):
        rb=mid(bts,bm,g+(lag+D)*MS)-mid(bts,bm,g+lag*MS);ok=np.isfinite(rc)&np.isfinite(rb)
        res[lag]=float(np.corrcoef(rc[ok],rb[ok])[0,1])
    k=max(res,key=res.get);out[f'xcorr_D{D}']={'peak_lag_ms':k,'peak_corr':res[k],'corr_at_0':res[0],'curve':res}
    print(f'window {D} ms: peak corr {res[k]:.3f} at broker lag {k:+d} ms (corr at lag 0: {res[0]:.3f})',flush=True)
# 2) event study: CME mid jumps >= X within 100 ms, isolated (no jump in previous 2 s)
for X in (0.3,0.5,1.0):
    d=mid(cts,cm,g)-mid(cts,cm,g-100*MS);ev=g[np.abs(d)>=X];sg=np.sign(d[np.abs(d)>=X])
    keep=np.r_[True,np.diff(ev)>2000*MS];ev=ev[keep];sg=sg[keep]
    b0=mid(bts,bm,ev-100*MS);c0=mid(cts,cm,ev-100*MS);c1=mid(cts,cm,ev+2000*MS)
    curve={}
    for tau in (0,25,50,100,150,200,300,500,1000,2000):
        curve[tau]=float(np.nanmedian(sg*(mid(bts,bm,ev+tau*MS)-b0)/np.abs(c1-c0+1e-9)))
    # 3) taker trade at broker after seeing the CME jump with latency L
    tr={}
    for L in (0,25,50,100,200):
        for Hs in (0.5,1,2,5):
            bb,aa=bq(ev+L*MS);bb2,aa2=bq(ev+L*MS+int(Hs*1000)*MS)
            pnl=np.where(sg>0,bb2-aa,bb-aa2);ok=np.isfinite(pnl)
            tr[f'L{L}_H{Hs}']={'n':int(ok.sum()),'mean_usd_per_oz':float(pnl[ok].mean()),'win':float((pnl[ok]>0).mean())}
    out[f'events_{X}']={'n':int(len(ev)),'broker_response_fraction':curve,'taker':tr}
    print(f'CME jump >= ${X} in 100ms: {len(ev)} events | broker reflected (median fraction of CME 2s move):',{k:round(v,2) for k,v in curve.items()},flush=True)
    print('   taker at broker (enter after latency L, exit H later), mean $/oz:',{k:(round(v["mean_usd_per_oz"],3),round(v["win"],2)) for k,v in tr.items() if k.endswith(('_H1','_H2'))},flush=True)
json.dump(out,open('/home/user/xauusd/reports/leadlag_jul2026.json','w'),indent=1)
