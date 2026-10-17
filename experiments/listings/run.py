"""New-listing short study (see PROTOCOL.md). usage: run.py DATA_DIR OUT.json"""
import sys,os,io,re,zipfile,subprocess,json,numpy as np,pandas as pd
from concurrent.futures import ThreadPoolExecutor
D,OUT=sys.argv[1:3];os.makedirs(D,exist_ok=True);S3='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision';B='https://data.binance.vision/'
def ls(prefix):
    out=[];marker=''
    while True:
        x=subprocess.run(['curl','-sS',f'{S3}?prefix={prefix}&delimiter=/&marker={marker}'],capture_output=True,text=True).stdout
        out+=re.findall(r'<Prefix>([^<]+)</Prefix>',x)[1:]+re.findall(r'<Key>([^<]+)</Key>',x)
        m=re.search(r'<NextMarker>([^<]+)</NextMarker>',x)
        if '<IsTruncated>true' not in x or not m:break
        marker=m.group(1)
    return out
def fetch(key):
    r=subprocess.run(['curl','-sS','-f','--retry','4',B+key],capture_output=True)
    if r.returncode:return None
    z=zipfile.ZipFile(io.BytesIO(r.stdout));return pd.read_csv(z.open(z.namelist()[0]),header=None)
def first_months(sym):
    fn=f'{D}/{sym}.parquet'
    if os.path.exists(fn):return pd.read_parquet(fn),pd.read_parquet(fn.replace('.parquet','_fund.parquet'))
    keys=sorted(k for k in ls(f'data/futures/um/monthly/klines/{sym}/1d/') if k.endswith('.zip'))[:3]
    ks=[fetch(k) for k in keys];ks=[k for k in ks if k is not None]
    if not ks:return None,None
    k=pd.concat(ks);k=k[pd.to_numeric(k[0],errors='coerce').notna()].iloc[:,:6].astype(float);k.columns=['t','o','h','l','c','v']
    fk=sorted(x for x in ls(f'data/futures/um/monthly/fundingRate/{sym}/') if x.endswith('.zip'))[:3]
    fs=[fetch(x) for x in fk];fs=[f for f in fs if f is not None]
    f=pd.concat(fs) if fs else pd.DataFrame({0:[],2:[]});f=f[pd.to_numeric(f[0],errors='coerce').notna()];f=pd.DataFrame({'t':f[0].astype(float),'r':f[2].astype(float)})
    k.to_parquet(fn);f.to_parquet(fn.replace('.parquet','_fund.parquet'));return k,f
syms=[p.rstrip('/').split('/')[-1] for p in ls('data/futures/um/monthly/klines/')];syms=[s for s in syms if s.endswith('USDT')]
print('symbols',len(syms),flush=True)
with ThreadPoolExecutor(12) as ex:data=dict(zip(syms,ex.map(first_months,syms)))
btc=data['BTCUSDT'][0] if data.get('BTCUSDT',(None,))[0] is not None else None
bk=pd.read_parquet('/home/user/crypto/BTCUSDT_1m.parquet');bd=bk.groupby(bk.open_time//86400000).open.first()
rows=[]
for s,(k,f) in data.items():
    if k is None or len(k)<35 or s=='BTCUSDT':continue
    k=k.sort_values('t').reset_index(drop=True);L=pd.to_datetime(k.t.iloc[0],unit='ms')
    if not (pd.Timestamp('2021-01-01')<=L<=pd.Timestamp('2026-08-31')):continue
    for h in (7,14,30):
        if len(k)<2+h:continue
        o0=k.o.iloc[1];o1=k.o.iloc[1+h];t0=k.t.iloc[1];t1=k.t.iloc[1+h]
        fund=f.r[(f.t>=t0)&(f.t<t1)].sum()*1e4 if len(f) else 0.0
        d0=int(t0//86400000);d1=int(t1//86400000);bret=(np.log(bd[d1])-np.log(bd[d0]))*1e4 if d0 in bd.index and d1 in bd.index else np.nan
        r=np.log(o1/o0)*1e4;rows.append(dict(sym=s,L=L,h=h,ret=r,fund=fund,btc=bret,short=-r+fund,hedged=-(r-bret)+fund,mx=np.log(k.h.iloc[1:2+h].max()/o0)*1e4))
R=pd.DataFrame(rows);R.to_parquet(f'{D}/rows.parquet');out={}
for h in (7,14,30):
    for per,(a,b) in (('2021-23',('2021','2024')),('2024-26',('2024','2027'))):
        q=R[(R.h==h)&(R.L>=a)&(R.L<b)];res={'n':int(len(q))}
        for col in ('short','hedged'):
            x=q[col].dropna();mm=x.groupby(q.L.dt.to_period('M')).mean()
            res[col]={'mean':float(x.mean()),'median':float(x.median()),'share_pos':float((x>0).mean()),'worst':float(x.min()),
                'months':int(len(mm)),'t_months':float(mm.mean()/mm.std()*np.sqrt(len(mm))),**{f'net{c}':float(x.mean()-c) for c in (15,30)},
                'by_year':{str(y):float(x[q.L.dt.year==y].mean()) for y in sorted(set(q.L.dt.year))}}
        res['mean_funding']=float(q.fund.mean());res['median_max_up_bp']=float(q.mx.median());out[f'{h}d|{per}']=res
        print(h,per,res['n'],{c:(round(res[c]['mean'],1),round(res[c]['median'],1),round(res[c]['t_months'],2),round(res[c]['share_pos'],2)) for c in ('short','hedged')},'fund',round(res['mean_funding'],1),flush=True)
json.dump(out,open(OUT,'w'),indent=1)
