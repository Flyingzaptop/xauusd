"""Line reversion (VWAP/SMA) and support/resistance bounce tests with random-walk and placebo controls. See PROTOCOL.md."""
import json,numpy as np,pandas as pd,numba as nb
d=pd.read_parquet('/home/user/data_raw/XAUUSD_41.parquet');t=pd.to_datetime(d.timestamp,utc=True)
ts=t.values.astype('datetime64[m]').astype(np.int64);O=d.open.values;H=d.high.values;L=d.low.values;C=d.close.values;V=d.volume.values.astype(float)
N=len(C);gap=np.r_[True,np.diff(ts)>30];sess=np.cumsum(gap)    # session id: new session after a >30 min gap
year=t.dt.year.values;COST=0.25;TMAX=240
@nb.njit(cache=True)
def race(i0,side,target,stop,H,L,C,sess,tmax):
    # entry already done at bar i0 (close or limit); scan from i0+1
    s0=sess[i0];j=i0+1;n=len(C)
    while j<n and j<=i0+tmax and sess[j]==s0:
        if side>0:
            hs=L[j]<=stop;ht=H[j]>=target
        else:
            hs=H[j]>=stop;ht=L[j]<=target
        if hs:return -1,j
        if ht:return 1,j
        j+=1
    return 0,min(j-1,n-1)
@nb.njit(cache=True)
def run_line(sig_long,sig_short,line,a_mult,H,L,C,sess,tmax,out):
    # sequential: one position at a time
    i=0;n=len(C);k=0
    while i<n-1:
        if sig_long[i] or sig_short[i]:
            side=1 if sig_long[i] else -1;e=C[i];tg=line[i];dd=abs(tg-e);st=e-side*a_mult*dd
            r,j=race(i,side,tg,st,H,L,C,sess,tmax)
            ex=tg if r==1 else (st if r==-1 else C[j])
            out[k,0]=i;out[k,1]=side;out[k,2]=r;out[k,3]=side*(ex-e);out[k,4]=dd;k+=1;i=j+1
        else:i+=1
    return k
def summarize(name,rows,null):
    if len(rows)==0:return None
    win=(rows[:,2]==1).mean();pnl=rows[:,3]-COST;yr=year[rows[:,0].astype(int)]
    py={int(y):float(pnl[yr==y].sum()) for y in np.unique(yr)}
    tt=pnl.mean()/(pnl.std()/np.sqrt(len(pnl)))
    return {'test':name,'n':int(len(rows)),'p_target':float(win),'null_p':null,'timeouts':float((rows[:,2]==0).mean()),'mean_gross':float(rows[:,3].mean()),
            'mean_net':float(pnl.mean()),'t_net':float(tt),'pos_years':int(sum(v>0 for v in py.values())),'years':len(py),'per_year':py}
RES=[]
# ---- A. lines
cs=pd.Series(C);pv=pd.Series(C*V);vv=pd.Series(V)
vwap=(pv.groupby(sess).cumsum()/vv.groupby(sess).cumsum().replace(0,np.nan)).values
lines={'VWAP_session':(vwap,120),'SMA20':(cs.rolling(20).mean().values,20),'SMA100':(cs.rolling(100).mean().values,100),'SMA300':(cs.rolling(300).mean().values,300)}
cross_info={}
for nm,(ln,w) in lines.items():
    dev=C-ln;sd=pd.Series(dev).rolling(w,min_periods=max(10,w//2)).std().values
    for kk in (2,3):
        lo=(dev<=-kk*sd)&np.isfinite(sd)&(sd>0);hi=(dev>=kk*sd)&np.isfinite(sd)&(sd>0)
        for am in (1.0,2.0):
            out=np.zeros((N,5));n=run_line(lo,hi,np.nan_to_num(ln),am,H,L,C,sess,TMAX,out)
            r=summarize(f'{nm} dev>={kk}sd, stop={am:g}x target',out[:n],am/(1+am));r and RES.append(r)
        # descriptive: does the MOVING line get crossed within 240 min?
        idx=np.where(lo|hi)[0][::50];crossed=0;tot=0
        for i in idx:
            j=min(i+TMAX,N-1);s=sess[i];seg=slice(i+1,j+1);m=sess[seg]==s
            if m.sum()==0:continue
            tot+=1;sgn=np.sign(dev[i]);crossed+=bool((np.sign(dev[seg][m])!=sgn).any())
        cross_info[f'{nm} dev>={kk}sd']=crossed/max(tot,1)
# ---- B. levels
day=t.dt.floor('D').values;ud,first=np.unique(day,return_index=True);bounds=np.r_[first,N]
rng=np.random.default_rng(7)
rng1=pd.Series(H-L).rolling(60,min_periods=20).mean().values
def day_levels(k):
    a,b=bounds[k-1],bounds[k];ph,pl,pc=H[a:b].max(),L[a:b].min(),C[b-1]
    bins=np.round(C[a:b]*2)/2;vb=pd.Series(V[a:b]).groupby(bins).sum();poc=float(vb.idxmax())
    lv={'prevH':ph,'prevL':pl,'prevC':pc,'POC':poc}
    a2=bounds[max(k-2,0)];seg=C[a2:b];m15=pd.Series(seg).groupby(np.arange(len(seg))//15).agg(['max','min'])
    hh=m15['max'].values;ll=m15['min'].values;sh=[hh[i] for i in range(2,len(hh)-2) if hh[i]==hh[i-2:i+3].max()][-3:];sl=[ll[i] for i in range(2,len(ll)-2) if ll[i]==ll[i-2:i+3].min()][-3:]
    for i,x in enumerate(sh):lv[f'swingH{i}']=x
    for i,x in enumerate(sl):lv[f'swingL{i}']=x
    return lv
@nb.njit(cache=True)
def touch_trades(a,b,level,bm,H,L,C,sess,rng1,tmax,out,k0):
    # first touch of level in [a,b): approach side from previous close; bounce bet with limit at level
    k=k0
    for i in range(a+1,b):
        if C[i-1]<level and H[i]>=level:side=-1
        elif C[i-1]>level and L[i]<=level:side=1
        else:continue
        bb=max(bm*rng1[i],0.30)
        if not np.isfinite(bb):return k
        tg=level+side*bb;st=level-side*bb
        # same-bar stop breach -> loss (conservative)
        if (side<0 and H[i]>=st) or (side>0 and L[i]<=st):
            out[k,0]=i;out[k,1]=side;out[k,2]=-1;out[k,3]=-bb;k+=1;return k
        r,j=race(i,side,tg,st,H,L,C,sess,tmax)
        ex=tg if r==1 else (st if r==-1 else C[j])
        out[k,0]=i;out[k,1]=side;out[k,2]=r;out[k,3]=side*(ex-level);k+=1;return k
    return k
outR={bm:{} for bm in (1.0,2.0)};outP={bm:{} for bm in (1.0,2.0)}
bufR={};bufP={}
for k in range(2,len(ud)):
    if pd.Timestamp(ud[k]).dayofweek>=5:continue
    lv=day_levels(k);a,b=bounds[k],bounds[k+1];p0=C[a]
    rn=np.floor(p0/10)*10;lv['round_lo']=rn;lv['round_hi']=rn+10
    for nm,x in lv.items():
        g=nm.rstrip('0123456789')
        fake=x+rng.choice([-1,1])*rng.uniform(3,8)
        for bm in (1.0,2.0):
            for store,lev in ((bufR,x),(bufP,fake)):
                key=(g,bm);o=store.setdefault(key,[]);tmp=np.zeros((1,4));n=touch_trades(a,b,float(lev),bm,H,L,C,sess,rng1,TMAX,tmp,0)
                if n:o.append(tmp[0])
for (g,bm),rows in bufR.items():
    R_=np.array(rows);P_=np.array(bufP[(g,bm)])
    r=summarize(f'LEVEL {g} bounce b={bm:g}x',R_,0.5);p=summarize(f'PLACEBO {g} bounce b={bm:g}x',P_,0.5)
    wr=(R_[:,2]==1).astype(float);wp=(P_[:,2]==1).astype(float);bs=np.random.default_rng(0)
    diff=[bs.choice(wr,len(wr)).mean()-bs.choice(wp,len(wp)).mean() for _ in range(2000)]
    r['placebo_p_target']=p['p_target'];r['placebo_mean_net']=p['mean_net'];r['diff_vs_placebo_ci95']=[float(np.quantile(diff,.025)),float(np.quantile(diff,.975))];RES.append(r)
json.dump({'results':RES,'moving_line_crossed_within_240m':cross_info},open('/home/user/xauusd/reports/levels/results.json','w'),indent=1)
print('== A. reversion to line (fixed target = line at entry). null = a/(a+d)')
for r in RES:
    if not r['test'].startswith('LEVEL'):print(f"{r['test']:42} n={r['n']:6d} P(target) {r['p_target']:.3f} vs null {r['null_p']:.3f} | net ${r['mean_net']:+.3f} t {r['t_net']:+.1f} +yrs {r['pos_years']}/{r['years']}")
print('== moving line crossed later (descriptive):',{k:round(v,3) for k,v in cross_info.items()})
print('== B. levels: bounce rate real vs placebo')
for r in RES:
    if r['test'].startswith('LEVEL'):print(f"{r['test']:28} n={r['n']:5d} bounce {r['p_target']:.3f} placebo {r['placebo_p_target']:.3f} diff CI [{r['diff_vs_placebo_ci95'][0]:+.3f},{r['diff_vs_placebo_ci95'][1]:+.3f}] | net ${r['mean_net']:+.3f} (placebo {r['placebo_mean_net']:+.3f}) +yrs {r['pos_years']}/{r['years']}")
