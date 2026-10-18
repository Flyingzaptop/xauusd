"""M4: SMA trend rules on untouched 2017-08..2019-12 Binance spot daily klines. usage: oos2017.py OUT.json"""
import sys,io,zipfile,subprocess,json,numpy as np,pandas as pd
def get(s,mo):
    r=subprocess.run(['curl','-sS','-f','--retry','4',f'https://data.binance.vision/data/spot/monthly/klines/{s}/1d/{s}-1d-{mo}.zip'],capture_output=True)
    if r.returncode:return None
    z=zipfile.ZipFile(io.BytesIO(r.stdout));return pd.read_csv(z.open(z.namelist()[0]),header=None)
def stats(r):
    r=r.dropna();a=r.mean()*365;v=r.std()*np.sqrt(365);eq=r.cumsum();return {'ann_ret':float(a),'sharpe':float(a/v) if v>0 else 0.,'maxdd':float((eq.cummax()-eq).max()),'by_year':{str(y):float(r[r.index.year==y].sum()) for y in sorted(set(r.index.year))}}
mos=[f'{y}-{m:02d}' for y in range(2017,2020) for m in range(1,13) if '2017-08'<=f'{y}-{m:02d}'<='2019-12']
out={};sl={}
for s in ('BTCUSDT','ETHUSDT'):
    k=pd.concat([x for x in (get(s,mo) for mo in mos) if x is not None]);p=pd.Series(k[4].astype(float).values,index=pd.to_datetime(k[0],unit='ms')).sort_index()
    r=np.log(p).diff();w=(0.5/(r.rolling(20).std()*np.sqrt(365))).clip(upper=2);out[f'{s}|buyhold']=stats(r[r.index>='2017-11-01'])
    for n in (20,50,100,200):
        pos=(np.sign(p-p.rolling(n).mean())*w).shift(1);ret=pos*r-pos.diff().abs()*10e-4;ret=ret[ret.index>='2017-11-01'];sl[(s,n)]=ret
        out[f'{s}|sma{n}']=stats(ret)
for n in (20,50,100,200):out[f'combo|sma{n}']=stats((sl[('BTCUSDT',n)]+sl[('ETHUSDT',n)])/2)
for k,v in out.items():print(k,{kk:round(vv,2) for kk,vv in v.items() if kk!='by_year'},{y:round(z,2) for y,z in v['by_year'].items()})
json.dump(out,open(sys.argv[1],'w'),indent=1)
