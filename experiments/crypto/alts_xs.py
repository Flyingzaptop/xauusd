"""Part G2: clustered inference for the G rule + cross-sectional long/short on alt perps. usage: alts_xs.py DATA_DIR OUT.json"""
import sys,json,numpy as np,pandas as pd
sys.argv+=[];D,OUT=sys.argv[1:3]
SYMS=[s+'USDT' for s in 'SOL XRP DOGE BNB ADA AVAX LINK LTC DOT TRX BCH ETC NEAR APT ARB OP SUI FIL ATOM UNI AAVE INJ TIA SEI WLD 1000PEPE WIF ENA TON HBAR'.split()]
Z={};PX={}
for s in SYMS:
    m=pd.read_parquet(f'{D}/{s}_metrics.parquet');m['t']=pd.to_datetime(m.create_time,utc=True)
    g=m.drop_duplicates('t').set_index('t').sort_index().resample('5min').last()
    x=np.log(g.count_long_short_ratio.astype(float).clip(lower=1e-6))
    z=(x-x.rolling(8640,min_periods=2000).mean().shift(1))/x.rolling(8640,min_periods=2000).std().shift(1)
    z.index=z.index+pd.Timedelta(minutes=5);Z[s]=z                        # usable from create_time + 5 min
    k=pd.read_parquet(f'{D}/{s}_1m.parquet');PX[s]=pd.Series(np.log(k.open.values),index=pd.to_datetime(k.open_time.values.astype(np.int64),unit='ms',utc=True))
Zm=pd.DataFrame(Z).sort_index().ffill(limit=6);P=pd.DataFrame(PX).sort_index()
out={}
# (b) cross-sectional
reb=pd.date_range(pd.Timestamp('2023-01-01',tz='UTC'),P.index[-1]-pd.Timedelta(hours=25),freq='12h')
zr=Zm.reindex(reb,method='ffill');ent=P.reindex(reb+pd.Timedelta(minutes=1),method='bfill');ent.index=reb
for h in (12,24):
    ex=P.reindex(reb+pd.Timedelta(minutes=1)+pd.Timedelta(hours=h),method='bfill');ex.index=reb;r=(ex-ent)*1e4
    pnl=[]
    for t in reb:
        zz=zr.loc[t].dropna();rr=r.loc[t]
        if len(zz)<15:pnl.append(np.nan);continue
        lo=zz.nsmallest(5).index;hi=zz.nlargest(5).index;pnl.append((rr[lo].mean()-rr[hi].mean())/2)   # per unit gross exposure
    pnl=pd.Series(pnl,index=reb).dropna()
    for per,(a,b) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):
        x=pnl[(pnl.index>=a)&(pnl.index<b)];scale=1 if h==12 else 0.5      # 24 h: two overlapping half-capital books
        # t with Newey-West-ish correction for 24 h overlap: use every other observation
        xs=x if h==12 else x.iloc[::2]
        out[f'xs|{h}h|{per}']={'n':int(len(x)),'gross_bp_per_trade':float(x.mean()),'t':float(xs.mean()/xs.std()*np.sqrt(len(xs))),
            **{f'net{c}':float(x.mean()-c) for c in (5,10,15)},'by_year':{str(y):float(x[x.index.year==y].mean()) for y in sorted(set(x.index.year))}}
        print('xs',h,per,{k:(round(v,2) if isinstance(v,float) else v) for k,v in out[f'xs|{h}h|{per}'].items()},flush=True)
# (a) clustered inference for G rule
for h in (12,24):
    rows=[]
    for s in SYMS:
        z=Z[s].dropna();px=PX[s];sig=z[np.abs(z)>2];last=None
        for t,v in sig.items():
            if last is not None and t<last+pd.Timedelta(hours=h):continue
            i=px.index.searchsorted(t+pd.Timedelta(minutes=1));j=px.index.searchsorted(t+pd.Timedelta(minutes=1)+pd.Timedelta(hours=h))
            if j>=len(px):break
            rows.append((t.floor('D'),-np.sign(v)*(px.iloc[j]-px.iloc[i])*1e4));last=t
    R=pd.DataFrame(rows,columns=['day','v']);dm=R.groupby('day').v.mean()
    for per,(a,b) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):
        x=dm[(dm.index>=pd.Timestamp(a,tz='UTC'))&(dm.index<pd.Timestamp(b,tz='UTC'))];rr=R[(R.day>=pd.Timestamp(a,tz='UTC'))&(R.day<pd.Timestamp(b,tz='UTC'))]
        out[f'G_clustered|{h}h|{per}']={'trades':int(len(rr)),'days':int(len(x)),'mean_trade':float(rr.v.mean()),'mean_day':float(x.mean()),'t_days':float(x.mean()/x.std()*np.sqrt(len(x))),
            'by_year':{str(y):float(rr.v[rr.day.dt.year==y].mean()) for y in sorted(set(rr.day.dt.year))}}
        print('G',h,per,{k:(round(v,2) if isinstance(v,float) else v) for k,v in out[f'G_clustered|{h}h|{per}'].items()},flush=True)
json.dump(out,open(OUT,'w'),indent=1)
