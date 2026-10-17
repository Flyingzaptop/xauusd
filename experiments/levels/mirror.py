"""Exploratory (post-hoc, suggested by run.py): (1) trend-following away from VWAP/SMA300 after a k-sigma deviation;
(2) breakout through previous-day high/low. Same conservative execution (same-bar stop+target = loss), cost $0.25."""
import json,numpy as np,pandas as pd
src=open('/home/user/xauusd/experiments/levels/run.py').read().split('RES=[]')[0].replace('cache=True','cache=False');g={};exec(src,g)
race,summarize,C,H,L,V,sess,year,N,TMAX,COST,t=[g[k] for k in ['race','summarize','C','H','L','V','sess','year','N','TMAX','COST','t']]
import numba as nb
@nb.njit(cache=False)
def run_trend(sig_up,sig_dn,line,H,L,C,sess,tmax,tmult,out):
    i=0;n=len(C);k=0
    while i<n-1:
        if sig_up[i] or sig_dn[i]:
            side=1 if sig_up[i] else -1;e=C[i];dd=abs(e-line[i]);tg=e+side*tmult*dd;st=line[i]
            r,j=race(i,side,tg,st,H,L,C,sess,tmax);ex=tg if r==1 else (st if r==-1 else C[j])
            out[k,0]=i;out[k,1]=side;out[k,2]=r;out[k,3]=side*(ex-e);out[k,4]=dd;k+=1;i=j+1
        else:i+=1
    return k
RES=[]
cs=pd.Series(C);vwap=(pd.Series(C*V).groupby(sess).cumsum()/pd.Series(V).groupby(sess).cumsum().replace(0,np.nan)).values
for nm,ln,w in [('VWAP_session',vwap,120),('SMA300',cs.rolling(300).mean().values,300)]:
    dev=C-ln;sd=pd.Series(dev).rolling(w,min_periods=max(10,w//2)).std().values
    for kk in (2,3):
        up=(dev>=kk*sd)&(sd>0);dn=(dev<=-kk*sd)&(sd>0)
        for tm in (1.0,0.5):
            out=np.zeros((N,5));n=run_trend(up,dn,np.nan_to_num(ln),H,L,C,sess,TMAX,tm,out)
            RES.append(summarize(f'TREND away from {nm} dev>={kk}sd target={tm:g}x d, stop=line',out[:n],1/(1+tm)))
# breakout of previous-day H/L: first touch from inside, enter at level, target b beyond, stop b back inside
day=t.dt.floor('D').values;ud,first=np.unique(day,return_index=True);bounds=np.r_[first,N]
rng1=pd.Series(H-L).rolling(60,min_periods=20).mean().values
for bm in (1.0,2.0,4.0):
    rows=[]
    for k in range(1,len(ud)):
        if pd.Timestamp(ud[k]).dayofweek>=5:continue
        a0,b0=bounds[k-1],bounds[k];a,b=bounds[k],bounds[k+1];ph,pl=H[a0:b0].max(),L[a0:b0].min()
        for lev,side in ((ph,1),(pl,-1)):
            for i in range(a+1,b):
                if (side>0 and C[i-1]<lev and H[i]>=lev) or (side<0 and C[i-1]>lev and L[i]<=lev):
                    bb=max(bm*rng1[i],0.30)
                    if not np.isfinite(bb):break
                    tg=lev+side*bb;st=lev-side*bb
                    if (side>0 and L[i]<=st) or (side<0 and H[i]>=st):rows.append([i,side,-1,-bb,bb]);break
                    r,j=race(i,side,tg,st,H,L,C,sess,TMAX);ex=tg if r==1 else (st if r==-1 else C[j]);rows.append([i,side,r,side*(ex-lev),bb]);break
    RES.append(summarize(f'BREAKOUT prev-day H/L b={bm:g}x',np.array(rows),0.5))
json.dump(RES,open('/home/user/xauusd/reports/levels/mirror_results.json','w'),indent=1)
for r in RES:print(f"{r['test']:58} n={r['n']:6d} P(target) {r['p_target']:.3f} vs null {r['null_p']:.3f} | gross ${r['mean_gross']:+.3f} net ${r['mean_net']:+.3f} t {r['t_net']:+.1f} +yrs {r['pos_years']}/{r['years']}")
for r in RES:
    if r['mean_gross']>0:print(r['test'],{k:round(v) for k,v in r['per_year'].items()})
