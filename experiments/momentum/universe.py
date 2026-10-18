"""M5: daily klines for every USDT perp ever listed -> survivorship-free top-20 alt trend basket. usage: universe.py DATA_DIR OUT.json"""
import sys,os,io,re,zipfile,subprocess,json,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
D,OUT=sys.argv[1:3];os.makedirs(D,exist_ok=True);S3='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
def ls(prefix):
    out=[];marker=''
    while True:
        x=subprocess.run(['curl','-sS',f'{S3}?prefix={prefix}&delimiter=/&marker={marker}'],capture_output=True,text=True).stdout
        out+=re.findall(r'<Prefix>([^<]+)</Prefix>',x)[1:]+re.findall(r'<Key>([^<]+)</Key>',x);m=re.search(r'<NextMarker>([^<]+)</NextMarker>',x)
        if '<IsTruncated>true' not in x or not m:break
        marker=m.group(1)
    return out
def one(sym):
    fn=f'{D}/{sym}.parquet'
    if os.path.exists(fn):return
    keys=[k for k in ls(f'data/futures/um/monthly/klines/{sym}/1d/') if k.endswith('.zip') and '-20' in k];ps=[]
    for k in keys:
        r=subprocess.run(['curl','-sS','-f','--retry','4','https://data.binance.vision/'+k],capture_output=True)
        if r.returncode:continue
        z=zipfile.ZipFile(io.BytesIO(r.stdout));ps.append(pd.read_csv(z.open(z.namelist()[0]),header=None))
    if not ps:return
    k=pd.concat(ps);k=k[pd.to_numeric(k[0],errors='coerce').notna()];k=pd.DataFrame({'t':k[0].astype(np.int64),'c':k[4].astype(float),'qv':k[7].astype(float)}).drop_duplicates('t').sort_values('t')
    k.to_parquet(fn)
syms=[p.rstrip('/').split('/')[-1] for p in ls('data/futures/um/monthly/klines/')];syms=[s for s in syms if s.endswith('USDT')]
with ThreadPoolExecutor(16) as ex:list(ex.map(one,syms))
P={};V={}
for s in syms:
    fn=f'{D}/{s}.parquet'
    if not os.path.exists(fn) or s in ('BTCUSDT','ETHUSDT'):continue
    k=pd.read_parquet(fn);idx=pd.to_datetime(k.t,unit='ms');P[s]=pd.Series(k.c.values,index=idx);V[s]=pd.Series(k.qv.values,index=idx)
P=pd.DataFrame(P).sort_index();V=pd.DataFrame(V).reindex(P.index);P=P[P.index>='2020-01-01'];V=V.reindex(P.index)
R=np.log(P).diff()                                       # NaN after delisting -> position contributes 0 from then (closed at last price)
W=(0.5/(R.rolling(20,min_periods=15).std()*np.sqrt(365))).clip(upper=2)
months=pd.date_range('2020-06-01','2026-09-01',freq='MS');mem=pd.DataFrame(False,index=P.index,columns=P.columns)
for m in months:
    vol=V.loc[m-pd.Timedelta(days=30):m-pd.Timedelta(days=1)].sum();top=vol[vol>0].nlargest(20).index
    end=m+pd.offsets.MonthBegin(1);mem.loc[(mem.index>=m)&(mem.index<end),top]=True
out={}
for nm,S in {'sma50':np.sign(P-P.rolling(50,min_periods=50).mean()),'ens':sum(np.sign(P-P.rolling(n,min_periods=n).mean()) for n in (20,50,100))/3}.items():
    pos=(S*W).where(mem).shift(1).fillna(0);ret=(pos*R.fillna(0)-pos.diff().abs().fillna(0)*10e-4);n=mem.shift(1).sum(1).replace(0,np.nan);x=(ret.sum(1)/n).dropna()
    for per,(a,b) in (('2021-23',('2021','2024')),('2024-26',('2024','2027'))):
        q=x[(x.index>=a)&(x.index<b)];eq=q.cumsum()
        out[f'{nm}|{per}']={'sharpe':float(q.mean()/q.std()*np.sqrt(365)),'ann':float(q.mean()*365),'maxdd':float((eq.cummax()-eq).max()),'by_year':{str(y):float(q[q.index.year==y].sum()) for y in sorted(set(q.index.year))}}
        print(nm,per,{k:(round(v,2) if isinstance(v,float) else {kk:round(vv,2) for kk,vv in v.items()}) for k,v in out[f'{nm}|{per}'].items()},flush=True)
bh=(R.where(mem.shift(1)).mean(1)).dropna()
for per,(a,b) in (('2021-23',('2021','2024')),('2024-26',('2024','2027'))):
    q=bh[(bh.index>=a)&(bh.index<b)];out[f'buyhold|{per}']={'sharpe':float(q.mean()/q.std()*np.sqrt(365)),'ann':float(q.mean()*365)};print('buyhold',per,out[f'buyhold|{per}'])
json.dump(out,open(OUT,'w'),indent=1)
