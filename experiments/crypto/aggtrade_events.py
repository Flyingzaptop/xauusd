"""Part E: event studies on raw aggTrades for one month. usage: aggtrade_events.py MONTH(YYYY-MM) [OUT.json]"""
import sys,zipfile,json,numpy as np,pandas as pd,time
MO=sys.argv[1];OUT=sys.argv[2] if len(sys.argv)>2 else f'/home/user/xauusd/reports/crypto/aggtrades_{MO}.json'
LAT=200;H={'10s':10_000,'1m':60_000,'5m':300_000,'30m':1_800_000,'60m':3_600_000};res={}
def load(S):
    z=zipfile.ZipFile(f'/home/user/crypto/aggtrades/{S}-aggTrades-{MO}.zip');n=z.namelist()[0]
    d=pd.read_csv(z.open(n),usecols=['price','quantity','transact_time','is_buyer_maker'],dtype={'price':np.float64,'quantity':np.float64,'transact_time':np.int64,'is_buyer_maker':bool})
    return d.transact_time.values,d.price.values,d.quantity.values,np.where(d.is_buyer_maker.values,-1,1).astype(np.int8)
def fwd(ts,px,t0,h):
    i0=np.searchsorted(ts,t0+LAT);i1=np.searchsorted(ts,t0+LAT+h);ok=i1<len(ts)
    r=np.full(len(t0),np.nan);r[ok]=1e4*np.log(px[np.minimum(i1[ok],len(ts)-1)]/px[i0[ok]]);return r
def report(name,t0,sign,ts,px):
    out={'n':int(len(t0))}
    if len(t0)<30:
        print(f'{name:52} n={len(t0)} (too few)',flush=True);res[name]=out;return out
    wk=((t0-t0.min())//(7*86_400_000)).astype(int)
    for hl,h in H.items():
        r=sign*fwd(ts,px,t0,h);ok=np.isfinite(r);r=r[ok];m=(t0[ok]//60000);u,inv=np.unique(m,return_inverse=True);cl=np.bincount(inv,r)
        t=r.sum()/np.sqrt((cl**2).sum()) if len(cl)>5 else 0
        out[hl]={'mean_bp':float(r.mean()),'t':float(t),'by_week':[round(float(r[wk[ok]==w].mean()),2) for w in range(wk.max()+1) if (wk[ok]==w).sum()>5]}
    res[name]=out
    print(f"{name:52} n={len(t0):6d} "+' '.join(f"{hl}:{out[hl]['mean_bp']:+6.2f}(t{out[hl]['t']:+4.1f})" for hl in H),flush=True)
for S,big in (('BTCUSDT',(1e6,5e6)),('ETHUSDT',(5e5,2e6))):
    t0=time.time();ts,px,q,sd=load(S);nt=px*q;print(f'== {S} {MO}: {len(ts)} aggTrades, load {time.time()-t0:.0f}s',flush=True)
    # taker orders: group consecutive rows with same ts and same side
    br=np.r_[True,(np.diff(ts)!=0)|(np.diff(sd)!=0)];gid=np.cumsum(br)-1;st=np.where(br)[0]
    o_ts=ts[st];o_sd=sd[st];o_nt=np.bincount(gid,nt);o_lv=np.bincount(gid)  # aggTrades per order = price levels swept
    o_p0=px[st];o_p1=px[np.r_[st[1:]-1,len(px)-1]]
    print(f'   taker orders {len(o_ts)}; notional p99 ${np.quantile(o_nt,.99):,.0f}',flush=True)
    # E1 large taker orders (direction: +1 = WITH the aggressor)
    for b in big:
        m=o_nt>=b;report(f'{S} E1 taker order >= ${b/1e6:g}M (with aggressor)',o_ts[m],o_sd[m],ts,px)
    # E3 sweeps
    m=(o_lv>=3)&(o_nt>=np.quantile(o_nt,.99));report(f'{S} E3 sweep >=3 levels & notional>=p99 (with)',o_ts[m],o_sd[m],ts,px)
    # E2 absorption on 1-min windows
    mi=ts//60000;u,inv=np.unique(mi,return_inverse=True);sn=np.bincount(inv,nt*sd)
    last=np.r_[np.where(np.diff(inv)!=0)[0],len(inv)-1];first=np.r_[0,last[:-1]+1];dp=1e4*np.log(px[last]/px[first])
    s=pd.Series(np.abs(sn));thr=s.rolling(1440,min_periods=600).quantile(.99).shift(1).values;s2=pd.Series(np.abs(dp));thr2=s2.rolling(1440,min_periods=600).quantile(.3).shift(1).values
    m=(np.abs(sn)>thr)&(np.abs(dp)<thr2);te=(u[m]+1)*60000
    report(f'{S} E2 absorption (with aggressor)',te,np.sign(sn[m]).astype(int),ts,px)
    # E2b (protocol amendment after E2 produced too few events): top-5% |flow| minutes where price did NOT move with the flow
    thr5=s.rolling(1440,min_periods=600).quantile(.95).shift(1).values
    m=(np.abs(sn)>thr5)&(np.sign(dp)!=np.sign(sn));te=(u[m]+1)*60000
    report(f'{S} E2b absorption top5% flow, price not with it (with)',te,np.sign(sn[m]).astype(int),ts,px)
    # E4 bursts per second
    se=ts//1000;us,invs=np.unique(o_ts//1000,return_inverse=True);cnt=np.bincount(invs);sim=np.bincount(invs,o_sd)/cnt
    c=pd.Series(cnt);thr=c.rolling(86400,min_periods=20000).quantile(.999).shift(1).values;print('   E4 seconds with orders',len(cnt),'p99.9 now',np.nanmedian(thr),flush=True)
    m=(cnt>=thr)&(np.abs(sim)>=0.8)&np.isfinite(thr);tb=(us[m]+1)*1000;keep=np.r_[True,np.diff(tb)>60000]
    report(f'{S} E4 burst (with dominant side)',tb[keep],np.sign(sim[m][keep]).astype(int),ts,px)
json.dump(res,open(OUT,'w'),indent=1)
