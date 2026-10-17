"""Short daily-option strategies on Deribit history (see PROTOCOL.md). usage: vrp.py DERIBIT_DIR OUT.json"""
import sys,glob,json,numpy as np,pandas as pd
D,OUT=sys.argv[1:3];FIT_END='2024-01-01';VAL_END='2025-01-01'
KS=[0.0,0.5,1.0,1.5];MS=[None,0.0,0.1,0.2];WING=1.0
def har_sigma(sym):
    """Forecast sigma of log return over the next 24 h (08:00->08:00 UTC) from Binance 1m; log-HAR fitted on 2021-2023 only."""
    d=pd.read_parquet(f'/home/user/crypto/{sym}USDT_1m.parquet');t=pd.to_datetime(d.open_time,unit='ms');r=np.log(d.close).diff()
    day=(t-pd.Timedelta(hours=8)).dt.floor('D')                      # day k = 08:00 of k .. 08:00 of k+1
    rv=(r**2).groupby(day).sum();rv=rv[rv>0];lr=np.log(rv)
    X=pd.DataFrame({'d':lr,'w':np.log(rv.rolling(7).mean()),'m':np.log(rv.rolling(30).mean())});y=lr.shift(-1)   # predict next day
    ok=X.notna().all(1)&y.notna()&(X.index<FIT_END);A=np.c_[np.ones(ok.sum()),X[ok].values];b=np.linalg.lstsq(A,y[ok].values,rcond=None)[0]
    res=y[ok].values-A@b;pred=np.exp(np.c_[np.ones(len(X)),X.values]@b+res.var()/2)
    # forecast made at 08:00 of day k+1 (after day k closed) for day k+1 -> index by entry date
    s=pd.Series(np.sqrt(pred),index=X.index+pd.Timedelta(days=1));return s,np.sqrt(rv.shift(-0)).rename(lambda x:x)
def load(cur):
    fs=sorted(glob.glob(f'{D}/{cur}_opt_*.parquet'));t=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
    p=t.instrument_name.str.split('-',expand=True);t['exp']=pd.to_datetime(p[1],format='%d%b%y');t['K']=p[2].astype(float);t['cp']=p[3]
    t['ts']=pd.to_datetime(t.timestamp,unit='ms');t['day']=t.ts.dt.floor('D')
    t=t[t.exp==t.day+pd.Timedelta(days=1)].copy()                   # daily options: expire next day 08:00
    dl=pd.read_parquet(f'{D}/{cur}_delivery.parquet');dl=pd.Series(dl.delivery_price.values,index=pd.to_datetime(dl.date))
    return t,dl
def fills(t):
    """per day/strike/type: taker-sell VWAP (our sell price), taker-buy VWAP (our buy price), coin units; mean IV and index."""
    g=t.groupby(['day','K','cp','direction']).apply(lambda x:pd.Series({'px':np.average(x.price,weights=x.amount),'iv':np.average(x.iv,weights=x.amount),'S0':x.index_price.mean()}),include_groups=False)
    return g.unstack('direction')
def leg(px,S0,K,cp,ST,sign):
    """USD P&L in bp of S0 for one option leg, sign -1 = sold, +1 = bought; fees 3 bp + 1.5 bp delivery, each capped 12.5% of price."""
    intr=max(ST-K,0) if cp=='C' else max(K-ST,0);fee=(min(3e-4,.125*px)+min(1.5e-4,.125*px))*S0
    return (sign*(intr-px*S0)-fee)/S0*1e4
def run(cur):
    t,dl=load(cur);sig,_=har_sigma(cur);F=fills(t);rows=[]
    for day,fd in F.groupby(level='day'):
        if day+pd.Timedelta(days=1) not in dl.index or day not in sig.index:continue
        ST=dl[day+pd.Timedelta(days=1)];sh=sig[day];fd=fd.droplevel('day')
        S0=np.nanmean(fd[('S0','sell')].values) if ('S0','sell') in fd else np.nan
        if not np.isfinite(S0):continue
        def pick(cp,target,side):
            c=fd.xs(cp,level='cp') if cp in fd.index.get_level_values('cp') else None
            if c is None or ('px',side) not in c:return None
            c=c[c[('px',side)].notna()]
            if cp=='C':c=c[c.index>=S0*0.999] if target>=S0 else c
            else:c=c[c.index<=S0*1.001] if target<=S0 else c
            if not len(c):return None
            K=c.index[np.argmin(np.abs(c.index-target))];return K,c.loc[K,('px',side)],c.loc[K,('iv',side)]
        for k in KS:
            ce=pick('C',S0*np.exp(k*sh),'sell');pu=pick('P',S0*np.exp(-k*sh),'sell')
            if ce is None or pu is None:continue
            short=leg(ce[1],S0,ce[0],'C',ST,-1)+leg(pu[1],S0,pu[0],'P',ST,-1);iv=(ce[2]+pu[2])/2/100/np.sqrt(365)
            wc=pick('C',S0*np.exp((k+WING)*sh),'buy');wp=pick('P',S0*np.exp(-(k+WING)*sh),'buy')
            wing=leg(wc[1],S0,wc[0],'C',ST,1)+leg(wp[1],S0,wp[0],'P',ST,1) if (wc and wp and wc[0]>ce[0] and wp[0]<pu[0]) else np.nan
            rows.append(dict(day=day,k=k,short=short,wing=wing,iv=iv,sig=sh,ret=np.log(ST/S0)))
    return pd.DataFrame(rows)
def st(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x)
    if n<5:return {'n':n}
    eq=np.cumsum(x);return {'n':n,'mean_bp':float(x.mean()),'t':float(x.mean()/x.std()*np.sqrt(n)),'sharpe':float(x.mean()/x.std()*np.sqrt(365)),
        'worst_bp':float(x.min()),'maxdd_bp':float((np.maximum.accumulate(eq)-eq).max()),'win':float((x>0).mean())}
out={}
for cur in ('BTC','ETH'):
    R=run(cur);R.to_parquet(f'{D}/{cur}_vrp_rows.parquet');res={}
    for per,(a,b) in {'fit':('2000','2024-01-01'),'val':('2024-01-01','2025-01-01'),'test':('2025-01-01','2100')}.items():
        r=R[(R.day>=a)&(R.day<b)]
        for k in KS:
            q=r[r.k==k]
            for m in MS:
                qq=q if m is None else q[q.iv>q.sig*(1+m)]
                for kind in ('short','condor'):
                    x=qq.short if kind=='short' else (qq.short+qq.wing)
                    res[f'{per}|k{k}|m{m}|{kind}']={**st(x),'by_year':{str(y):st(x[qq.day.dt.year==y]) for y in sorted(qq.day.dt.year.unique())}}
    out[cur]=res
    for per in ('fit','val','test'):
        print(cur,per,{k.split('|',1)[1]:(round(v.get('mean_bp',0),1),round(v.get('t',0),1)) for k,v in res.items() if k.startswith(per) and '|mNone|' in k},flush=True)
json.dump(out,open(OUT,'w'),indent=1,default=float)
