"""M3: SMA-trend robustness across lookbacks, alts with funding, combined vol-targeted portfolio. usage: robust.py OUT.json"""
import sys,os,io,zipfile,subprocess,json,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
def daily(fn):
    k=pd.read_parquet(fn);t=pd.to_datetime(k.open_time.astype(np.int64),unit='ms');return pd.Series((k.close if 'close' in k else k.open).astype(float).values,index=t).resample('1D').last().dropna()
def stats(r):
    r=r.dropna();a=r.mean()*365;v=r.std()*np.sqrt(365);eq=r.cumsum();return {'ann_ret':float(a),'ann_vol':float(v),'sharpe':float(a/v) if v>0 else 0.,'maxdd':float((eq.cummax()-eq).max()),'by_year':{str(y):float(r[r.index.year==y].sum()) for y in sorted(set(r.index.year))}}
def funding(s):
    fn=f'/home/user/crypto/alts/{s}_fund.parquet'
    if os.path.exists(fn):return pd.read_parquet(fn).r
    def get(mo):
        r=subprocess.run(['curl','-sS','-f','--retry','4',f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{mo}.zip'],capture_output=True)
        if r.returncode:return None
        z=zipfile.ZipFile(io.BytesIO(r.stdout));return pd.read_csv(z.open(z.namelist()[0]))
    mos=[f'{y}-{m:02d}' for y in range(2022,2027) for m in range(1,13) if '2022-12'<=f'{y}-{m:02d}'<='2026-09']
    with ThreadPoolExecutor(8) as ex:ps=[p for p in ex.map(get,mos) if p is not None]
    f=pd.concat(ps);f.index=pd.to_datetime(f.calc_time,unit='ms');d=f.last_funding_rate.astype(float).resample('1D').sum().rename('r').to_frame();d.to_parquet(fn);return d.r
SY=[x+'USDT' for x in 'SOL XRP DOGE BNB ADA AVAX LINK LTC DOT TRX BCH ETC NEAR APT ARB OP SUI FIL ATOM UNI AAVE INJ TIA SEI WLD 1000PEPE WIF ENA TON HBAR'.split()]
P={s:daily(f'/home/user/crypto/alts/{s}_1m.parquet') for s in SY};FU={s:funding(s) for s in SY}
for s in ('BTCUSDT','ETHUSDT'):
    P[s]=daily(f'/home/user/crypto/{s}_1m.parquet');f=pd.read_parquet(f'/home/user/crypto/{s}_funding.parquet');f.index=pd.to_datetime(f.calc_time,unit='ms');FU[s]=f.last_funding_rate.astype(float).resample('1D').sum()
P=pd.DataFrame(P);F=pd.DataFrame(FU).reindex(P.index).fillna(0);R=np.log(P).diff();W=(0.5/(R.rolling(20).std()*np.sqrt(365))).clip(upper=2)
out={}
def sleeve(n,cols):
    S=np.sign(P[cols]-P[cols].rolling(n).mean());pos=(S*W[cols]).shift(1)
    return (pos*R[cols]-pos*F[cols]-pos.diff().abs()*10e-4).mean(1)
for n in (20,50,100,200):
    sl={'BTC':sleeve(n,['BTCUSDT']),'ETH':sleeve(n,['ETHUSDT']),'alts':sleeve(n,SY)}
    for nm,r in sl.items():
        for per,(a,b) in (('2020-23',('2020','2024')),('2024-26',('2024','2027')),('2025-26',('2025','2027'))):
            if nm=='alts' and per=='2020-23':per,a='2023-24','2023';b='2025'
            out[f'sma{n}|{nm}|{per}']=stats(r[(r.index>=a)&(r.index<b)])
    C=pd.DataFrame(sl);C=C[C.index>='2023-03-01']
    port=C.div(C.rolling(60).std().shift(1)).mean(1);port=port*(0.20/np.sqrt(365))/port.rolling(60).std().shift(1)
    for per,(a,b) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):out[f'sma{n}|portfolio20|{per}']=stats(port[(port.index>=a)&(port.index<b)])
for k,v in out.items():print(k,{kk:round(vv,2) for kk,vv in v.items() if kk!='by_year'},{y:round(z,2) for y,z in v['by_year'].items()})
json.dump(out,open(sys.argv[1],'w'),indent=1)
