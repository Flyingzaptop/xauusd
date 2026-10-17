"""Part G: account long/short ratio contrarian rule on alt perps (rule fixed on BTC/ETH in Part D). usage: alts_lsratio.py DATA_DIR OUT.json"""
import sys,os,io,zipfile,subprocess,datetime as dt,json,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
D,OUT=sys.argv[1:3];os.makedirs(D,exist_ok=True)
SYMS=[s+'USDT' for s in 'SOL XRP DOGE BNB ADA AVAX LINK LTC DOT TRX BCH ETC NEAR APT ARB OP SUI FIL ATOM UNI AAVE INJ TIA SEI WLD 1000PEPE WIF ENA TON HBAR'.split()]
B='https://data.binance.vision/data/futures/um'
def fetch(url):
    r=subprocess.run(['curl','-sS','-f','--retry','4',url],capture_output=True)
    if r.returncode:return None
    z=zipfile.ZipFile(io.BytesIO(r.stdout));return pd.read_csv(z.open(z.namelist()[0]))
def metrics(s):
    fn=f'{D}/{s}_metrics.parquet'
    if os.path.exists(fn):return pd.read_parquet(fn)
    days=[dt.date(2022,12,1)+dt.timedelta(i) for i in range((dt.date(2026,9,30)-dt.date(2022,12,1)).days+1)]
    with ThreadPoolExecutor(16) as ex:parts=[p for p in ex.map(lambda d:fetch(f'{B}/daily/metrics/{s}/{s}-metrics-{d}.zip'),days) if p is not None]
    m=pd.concat(parts)[['create_time','count_long_short_ratio']];m.to_parquet(fn);return m
def klines(s):
    fn=f'{D}/{s}_1m.parquet'
    if os.path.exists(fn):return pd.read_parquet(fn)
    mos=[f'{y}-{m:02d}' for y in range(2022,2027) for m in range(1,13) if '2022-12'<=f'{y}-{m:02d}'<='2026-09']
    with ThreadPoolExecutor(8) as ex:parts=[p for p in ex.map(lambda mo:fetch(f'{B}/monthly/klines/{s}/1m/{s}-1m-{mo}.zip'),mos) if p is not None]
    parts=[p if 'open_time' in p.columns else p.set_axis(['open_time','open','high','low','close','volume','close_time','qv','n','tb','tbq','ig'][:p.shape[1]],axis=1) for p in parts]
    k=pd.concat(parts)[['open_time','open']];k=k[pd.to_numeric(k.open_time,errors='coerce').notna()].astype(float);k=k.drop_duplicates('open_time').sort_values('open_time');k.to_parquet(fn);return k
def nonoverlap(e,h):
    keep=[];last=-10**12
    for i,x in enumerate(e):
        if x>=last+h:keep.append(i);last=x
    return np.array(keep,int)
res={};rows=[]
for s in SYMS:
    try:m=metrics(s);k=klines(s)
    except Exception as e:print(s,'fail',repr(e)[:100],flush=True);continue
    m['t']=pd.to_datetime(m.create_time,utc=True);g5=m.drop_duplicates('t').set_index('t').sort_index().resample('5min').last()
    kt=k.open_time.values.astype(np.int64);lo=np.log(k.open.values);N=len(lo)
    avail=((g5.index+pd.Timedelta(minutes=5)).values.astype('datetime64[ms]').astype(np.int64));ei=np.searchsorted(kt,avail)+1
    x=np.log(g5.count_long_short_ratio.astype(float).clip(lower=1e-6).values)
    mu=pd.Series(x).rolling(8640,min_periods=2000).mean().shift(1).values;sd=pd.Series(x).rolling(8640,min_periods=2000).std().shift(1).values;z=(x-mu)/sd
    yr=pd.DatetimeIndex(g5.index).year.values;res[s]={}
    for hh in (720,1440):
        ok=np.isfinite(z)&(np.abs(z)>2)&(ei+hh<N);idx=np.where(ok)[0];ii=idx[nonoverlap(ei[idx],hh)]
        e=ei[ii];v=-np.sign(z[ii])*(lo[e+hh]-lo[e])*1e4;y=yr[ii]
        for Y in np.unique(y):
            w=v[y==Y];res[s][f'{hh//60}h|{Y}']={'n':int(len(w)),'gross':float(w.mean())}
            rows.append(dict(sym=s,h=hh//60,year=int(Y),n=len(w),gross=float(w.mean()),sd=float(w.std())))
    print(s,{k_:(v_['n'],round(v_['gross'],1)) for k_,v_ in res[s].items()},flush=True)
R=pd.DataFrame(rows);summary={}
for h in (12,24):
    for per,yrs in (('2023-24',(2023,2024)),('2025-26',(2025,2026))):
        q=R[(R.h==h)&R.year.isin(yrs)];sm=q.groupby('sym').apply(lambda g:np.average(g.gross,weights=g.n),include_groups=False)
        summary[f'{h}h|{per}']={'symbols':int(len(sm)),'mean_of_symbol_means':float(sm.mean()),'t_symbols':float(sm.mean()/(sm.std()/np.sqrt(len(sm)))),
            'share_positive_gross':float((sm>0).mean()),**{f'net{c}':float(sm.mean()-c) for c in (10,15,20)},'trades':int(q.n.sum())}
        print(h,per,{k_:round(v_,2) for k_,v_ in summary[f'{h}h|{per}'].items()},flush=True)
json.dump({'per_symbol':res,'summary':summary},open(OUT,'w'),indent=1)
