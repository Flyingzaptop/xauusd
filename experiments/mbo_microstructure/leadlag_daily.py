import numpy as np,pandas as pd,json
C=np.load('/home/user/mbo_out/cme_bbo_jul.npz');B=np.load('/home/user/mbo_out/broker_bbo_jul.npz')
cm=(C['bid']+C['ask'])/2;bm=(B['bid']+B['ask'])/2;cts=C['ts'];bts=B['ts'];MS=1_000_000
mid=lambda ts,m,t:m[np.maximum(np.searchsorted(ts,t,side='right')-1,0)]
def bq(t):
    i=np.searchsorted(bts,t,side='right')-1;return B['bid'][i],B['ask'][i]
rows=[]
for day in pd.bdate_range('2026-07-07','2026-07-24',tz='UTC'):
    g=np.arange((day+pd.Timedelta(hours=1)).value,(day+pd.Timedelta(hours=20)).value,100*MS)
    lastb=bts[np.maximum(np.searchsorted(bts,g,side='right')-1,0)];g=g[(g-lastb)<2000*MS]
    if len(g)<10000:continue
    rc=mid(cts,cm,g+500*MS)-mid(cts,cm,g);best=(None,-1)
    for lag in range(0,401,10):
        rb=mid(bts,bm,g+(lag+500)*MS)-mid(bts,bm,g+lag*MS);c=np.corrcoef(rc,rb)[0,1]
        if c>best[1]:best=(lag,c)
    d=mid(cts,cm,g)-mid(cts,cm,g-100*MS);m=np.abs(d)>=0.5;ev=g[m];sg=np.sign(d[m]);k=np.r_[True,np.diff(ev)>2000*MS];ev,sg=ev[k],sg[k]
    res={'day':str(day.date()),'peak_lag_ms':best[0],'peak_corr':round(best[1],3),'events':int(len(ev))}
    for L in (50,100):
        bb,aa=bq(ev+L*MS);bb2,aa2=bq(ev+L*MS+1000*MS);p=np.where(sg>0,bb2-aa,bb-aa2);res[f'pnl_L{L}']=round(float(p.mean()),3);res[f'sum_L{L}']=round(float(p.sum()),1)
    rows.append(res);print(res,flush=True)
json.dump(rows,open('/home/user/xauusd/reports/leadlag_jul2026_daily.json','w'),indent=1)
