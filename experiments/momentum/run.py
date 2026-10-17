"""Daily time-series and cross-sectional momentum (see PROTOCOL.md). usage: run.py OUT.json"""
import sys,json,numpy as np,pandas as pd
def daily(fn):
    k=pd.read_parquet(fn);t=pd.to_datetime(k.open_time.astype(np.int64),unit='ms');return pd.Series(k.close.astype(float).values if 'close' in k else k.open.astype(float).values,index=t).resample('1D').last().dropna()
def stats(r):
    r=r.dropna();a=r.mean()*365;v=r.std()*np.sqrt(365);eq=r.cumsum();return {'ann_ret':float(a),'sharpe':float(a/v) if v>0 else 0.,'maxdd':float((eq.cummax()-eq).max()),'by_year':{str(y):float(r[r.index.year==y].sum()) for y in sorted(set(r.index.year))}}
out={}
# ---- M1 on BTC/ETH (with funding)
for s in ('BTCUSDT','ETHUSDT'):
    p=daily(f'/home/user/crypto/{s}_1m.parquet');r=np.log(p).diff()
    f=pd.read_parquet(f'/home/user/crypto/{s}_funding.parquet');f.index=pd.to_datetime(f.calc_time,unit='ms');fd=f.last_funding_rate.astype(float).resample('1D').sum().reindex(r.index).fillna(0)
    vol=r.rolling(20).std()*np.sqrt(365);w=(0.5/vol).clip(upper=2)
    sig={'tsmom28':np.sign(np.log(p).diff(28)),'sma50':np.sign(p-p.rolling(50).mean()),'sma50_long':(p>p.rolling(50).mean()).astype(float)}
    for nm,sg in sig.items():
        pos=(sg*w).shift(1)                                    # decided at yesterday's close
        ret=pos*r-pos*fd-pos.diff().abs().fillna(0)*10e-4
        for per,(a,b) in (('2020-23',('2020','2024')),('2024-26',('2024','2027'))):
            out[f'{s}|{nm}|{per}']=stats(ret[(ret.index>=a)&(ret.index<b)])
    for per,(a,b) in (('2020-23',('2020','2024')),('2024-26',('2024','2027'))):
        out[f'{s}|buyhold|{per}']=stats(r[(r.index>=a)&(r.index<b)])
# ---- M1 on alts (no funding), equal-weight basket of per-asset strategies; M2 cross-sectional
SY=[x+'USDT' for x in 'SOL XRP DOGE BNB ADA AVAX LINK LTC DOT TRX BCH ETC NEAR APT ARB OP SUI FIL ATOM UNI AAVE INJ TIA SEI WLD 1000PEPE WIF ENA TON HBAR'.split()]
P=pd.DataFrame({s:daily(f'/home/user/crypto/alts/{s}_1m.parquet') for s in SY})
for s in ('BTCUSDT','ETHUSDT'):P[s]=daily(f'/home/user/crypto/{s}_1m.parquet')
P=P[P.index>='2023-01-01'];R=np.log(P).diff()
vol=R.rolling(20).std()*np.sqrt(365);W=(0.5/vol).clip(upper=2)
for nm,S in {'tsmom28':np.sign(np.log(P).diff(28)),'sma50':np.sign(P-P.rolling(50).mean()),'sma50_long':(P>P.rolling(50).mean()).astype(float).where(P.notna())}.items():
    pos=(S*W).shift(1);ret=(pos*R-pos.diff().abs()*10e-4).mean(1)
    for per,(a,b) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):out[f'alts|{nm}|{per}']=stats(ret[(ret.index>=a)&(ret.index<b)])
for per,(a,b) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):out[f'alts|buyhold_ew|{per}']=stats(R.mean(1)[(R.index>=a)&(R.index<b)])
mon=P.index[P.index.dayofweek==0];xs=[]
for i,t in enumerate(mon[:-1]):
    past=np.log(P.loc[:t-pd.Timedelta(days=1)]).iloc[-28:];mom=(past.iloc[-1]-past.iloc[0]).dropna()
    if len(mom)<15:continue
    lo=mom.nsmallest(5).index;hi=mom.nlargest(5).index;nx=mon[i+1]
    fr=np.log(P.loc[nx]/P.loc[t]);xs.append((t,(fr[hi].mean()-fr[lo].mean())/2-2*10e-4))
X=pd.Series(dict(xs))
for per,(a,b) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):
    x=X[(X.index>=a)&(X.index<b)];out[f'alts|xsmom|{per}']={'weekly_mean_pct':float(x.mean()*100),'t':float(x.mean()/x.std()*np.sqrt(len(x))),'sharpe':float(x.mean()/x.std()*np.sqrt(52)),'n':int(len(x))}
for k,v in out.items():print(k,{kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!='by_year'},{y:round(z,2) for y,z in v.get('by_year',{}).items()})
json.dump(out,open(sys.argv[1],'w'),indent=1)
