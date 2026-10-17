"""K1 CME-gap fill and K2 hour-of-week seasonality (see PROTOCOL.md). usage: run.py OUT.json"""
import sys,json,numpy as np,pandas as pd
out={}
def st(x):
    x=np.asarray(x,float);n=len(x);return {'n':n,'mean':float(x.mean()) if n else 0.,'t':float(x.mean()/x.std()*np.sqrt(n)) if n>2 and x.std()>0 else 0.}
for s in ('BTCUSDT','ETHUSDT'):
    d=pd.read_parquet(f'/home/user/crypto/{s}_1m.parquet');t=pd.to_datetime(d.open_time,unit='ms');px=pd.Series(d.close.values,index=t);hi=pd.Series(d.high.values,index=t);lo=pd.Series(d.low.values,index=t)
    # ---- K1
    rows=[]
    for fri in pd.date_range('2020-01-03','2026-09-25',freq='W-FRI'):
        summer=4<=fri.month<=10;fc=fri+pd.Timedelta(hours=20 if summer else 21);ro=fri+pd.Timedelta(days=2,hours=22 if summer else 23);end=fri+pd.Timedelta(days=7,hours=20 if summer else 21)
        try:pf=px.asof(fc);p0=px.asof(ro)
        except Exception:continue
        if not(np.isfinite(pf) and np.isfinite(p0)):continue
        gap=np.log(p0/pf);w=slice(ro+pd.Timedelta(minutes=1),end);side=-np.sign(gap)
        hw=hi[w];lw=lo[w]
        if not len(hw):continue
        hit=(lw<=pf).values if side<0 else (hw>=pf).values          # TP: price reaches the Friday close
        j=np.argmax(hit) if hit.any() else -1
        for stop in (False,True):
            ex=px[w].iloc[-1];how='time'
            if stop:
                sl=p0*np.exp(-side*2*abs(gap));sh=(hw>=sl).values if side<0 else (lw<=sl).values;k=np.argmax(sh) if sh.any() else -1
                if k>=0 and (j<0 or k<=j):ex=sl;how='stop'
                elif j>=0:ex=pf;how='tp'
            elif j>=0:ex=pf;how='tp'
            rows.append(dict(week=fri,gap=gap,stop=stop,pnl=side*np.log(ex/p0)*1e4-10,how=how))
    R=pd.DataFrame(rows);k1={}
    for g in (0.005,0.01,0.02):
        for stop in (False,True):
            for per,(a,b) in (('2020-23',('2020','2024')),('2024-26',('2024','2027'))):
                q=R[(R.stop==stop)&(R.gap.abs()>g)&(R.week>=a)&(R.week<b)]
                k1[f'g{g}|stop{stop}|{per}']={**st(q.pnl),'tp_rate':float((q.how=='tp').mean()) if len(q) else 0.}
    out[f'{s}|K1']=k1;print(s,'K1',{k:(v['n'],round(v['mean'],1),round(v['t'],2),round(v['tp_rate'],2)) for k,v in k1.items()},flush=True)
    # ---- K2
    h=px.resample('1h').last();r=np.log(h).diff().shift(-1)*1e4          # return of the hour starting at index
    df=pd.DataFrame({'r':r,'how':h.index.dayofweek*24+h.index.hour,'y':h.index.year}).dropna();k2={}
    for Y in range(2022,2027):
        past=df[df.y<Y];cur=df[df.y==Y]
        g=past.groupby('how').r.agg(['mean','std','count']);g['t']=g['mean']/(g['std']/np.sqrt(g['count']))
        top=g.t.abs().nlargest(24).index;sg=np.sign(g.t)
        q=cur[cur.how.isin(top)];k2[str(Y)]=st(sg[q.how].values*q.r.values-10)
    pooled=pd.concat([df[(df.y==Y)] for Y in range(2024,2027)])
    out[f'{s}|K2']=k2;print(s,'K2',{k:(v['n'],round(v['mean'],1),round(v['t'],2)) for k,v in k2.items()},flush=True)
json.dump(out,open(sys.argv[1],'w'),indent=1)
