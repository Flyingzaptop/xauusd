"""Part F: hourly signed flow by taker-order size bucket -> forward returns. Builds/caches hourly tables per month."""
import zipfile,os,json,sys,numpy as np,pandas as pd
from scipy.stats import spearmanr
B=[0,1e3,1e4,1e5,1e6,np.inf];NB=['retail<1k','small1-10k','mid10-100k','large100k-1M','whale>=1M']
CACHE='/home/user/crypto/hourly';os.makedirs(CACHE,exist_ok=True)
def hourly(S,mo):
    f=f'{CACHE}/{S}-{mo}.parquet'
    if os.path.exists(f):return pd.read_parquet(f)
    z=zipfile.ZipFile(f'/home/user/crypto/aggtrades/{S}-aggTrades-{mo}.zip')
    d=pd.read_csv(z.open(z.namelist()[0]),usecols=['price','quantity','transact_time','is_buyer_maker'])
    ts=d.transact_time.values;px=d.price.values;nt=px*d.quantity.values;sd=np.where(d.is_buyer_maker.values,-1,1)
    br=np.r_[True,(np.diff(ts)!=0)|(np.diff(sd)!=0)];gid=np.cumsum(br)-1;st=np.where(br)[0]
    o_ts=ts[st];o_sd=sd[st];o_nt=np.bincount(gid,nt);hr=o_ts//3_600_000;b=np.digitize(o_nt,B)-1
    df=pd.DataFrame({'hr':hr,'b':b,'s':o_sd*o_nt,'a':o_nt});g=df.groupby(['hr','b']).agg(s=('s','sum'),a=('a','sum')).unstack('b').fillna(0)
    g.columns=[f'{x}_{NB[int(y)]}' for x,y in g.columns]
    # hourly close price = last trade price of the hour; open of next hour start + 1 min proxy via first trade after hr end + 60 s
    th=ts//3_600_000;last=pd.Series(px).groupby(th).last();g['close']=last.reindex(g.index).values
    g=g.reset_index();g.to_parquet(f,index=False);return g
months=[f'2026-{m:02d}' for m in range(1,10)];out={}
for S in ('BTCUSDT','ETHUSDT'):
    H=pd.concat([hourly(S,m) for m in months]).sort_values('hr').drop_duplicates('hr').set_index('hr')
    H=H.reindex(range(H.index.min(),H.index.max()+1));lp=np.log(H.close.ffill().values)
    mo=pd.to_datetime(H.index.values*3_600_000,unit='ms').strftime('%Y-%m').values
    print(f'== {S}: {len(H)} hours',flush=True)
    for W in (1,4,24):
        for nb in NB:
            s=H[f's_{nb}'].fillna(0).rolling(W).sum();a=H[f'a_{nb}'].fillna(0).rolling(W).sum();imb=(s/a.replace(0,np.nan)).values
            for h in (1,4,24):
                y=np.full(len(lp),np.nan);y[:-h]=(lp[h:]-lp[:-h])*1e4   # close of hour t -> close of hour t+h (signal known at hour end)
                idx=np.arange(0,len(lp),h);x=imb[idx];yy=y[idx];mm=mo[idx];ok=np.isfinite(x)&np.isfinite(yy)
                if ok.sum()<50:continue
                rho=spearmanr(x[ok],yy[ok]).correlation;q=pd.qcut(x[ok],5,labels=False,duplicates='drop');spread=yy[ok][q==q.max()].mean()-yy[ok][q==0].mean()
                per={m:float(spearmanr(x[ok][mm[ok]==m],yy[ok][mm[ok]==m]).correlation) for m in months if (mm[ok]==m).sum()>15}
                sgn=np.sign(rho);same=sum(1 for v in per.values() if np.sign(v)==sgn)
                out[f'{S}|W{W}|{nb}|h{h}']={'rho':float(rho),'q5_minus_q1_bp':float(spread),'months_same_sign':f'{same}/{len(per)}','n':int(ok.sum()),'per_month':per}
for k,v in sorted(out.items(),key=lambda kv:-abs(kv[1]['rho'])):pass
json.dump(out,open('/home/user/xauusd/reports/crypto/size_flow.json','w'),indent=1)
keys=sorted({k.split('|',1)[1] for k in out})
print(f"{'window|bucket|horizon':34} {'BTC rho':>8} {'spread':>7} {'mo':>5} | {'ETH rho':>8} {'spread':>7} {'mo':>5}")
rows=[]
for k in keys:
    b=out.get('BTCUSDT|'+k);e=out.get('ETHUSDT|'+k)
    if b and e:rows.append((abs(b['rho'])+abs(e['rho']),k,b,e))
for _,k,b,e in sorted(rows,reverse=True)[:20]:
    flag=' <==' if np.sign(b['rho'])==np.sign(e['rho']) and int(b['months_same_sign'].split('/')[0])>=7 and int(e['months_same_sign'].split('/')[0])>=7 and min(abs(b['q5_minus_q1_bp']),abs(e['q5_minus_q1_bp']))>10 else ''
    print(f"{k:34} {b['rho']:+8.3f} {b['q5_minus_q1_bp']:+7.1f} {b['months_same_sign']:>5} | {e['rho']:+8.3f} {e['q5_minus_q1_bp']:+7.1f} {e['months_same_sign']:>5}{flag}")
