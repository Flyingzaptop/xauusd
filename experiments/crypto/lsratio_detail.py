"""Detail of the account long/short ratio signal: raw sign relation per year (no walk-forward fitting), thresholds, horizons."""
import glob,json,sys,numpy as np,pandas as pd
sys.path.insert(0,'/home/user/xauusd/experiments/scan222');import kernels as K
out={}
for S in ('BTCUSDT','ETHUSDT'):
    m=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'/home/user/crypto/um/{S}/metrics/*.csv'))]);m['t']=pd.to_datetime(m.create_time,utc=True)
    g5=m.drop_duplicates('t').set_index('t').sort_index().resample('5min').last()
    k=pd.read_parquet(f'/home/user/crypto/{S}_1m.parquet');kt=k.open_time.values;lo=np.log(k.open.values);N=len(lo)
    avail=(g5.index+pd.Timedelta(minutes=5)).values.astype('datetime64[ms]').astype(np.int64);ei=np.searchsorted(kt,avail)+1;ok=ei+1500<N
    year=pd.DatetimeIndex(g5.index).year.values
    x=np.log(g5.count_long_short_ratio.astype(float).values);mu=pd.Series(x).rolling(8640,min_periods=2000).mean().shift(1).values;sd=pd.Series(x).rolling(8640,min_periods=2000).std().shift(1).values;z=(x-mu)/sd
    print(f'== {S}: account L/S ratio; raw mean of sign(z)*forward return (bp) — positive = WITH the crowd, negative = AGAINST the crowd')
    for thr in (1.5,2,3):
        for hh in (60,240,720,1440):
            idx=np.where(ok&(np.abs(z)>thr)&np.isfinite(z))[0];ii=idx[K.nonoverlap(ei[idx],hh)];e=ei[ii];y=(lo[e+hh]-lo[e])*1e4;v=np.sign(z[ii])*y;yy=year[ii]
            per={int(Y):round(float(v[yy==Y].mean()),1) for Y in np.unique(yy)};nper={int(Y):int((yy==Y).sum()) for Y in np.unique(yy)}
            t=v.mean()/(v.std()/np.sqrt(len(v)))
            print(f'  |z|>{thr} h={hh//60:>2}h n={len(v):5d} mean {v.mean():+7.2f} t {t:+5.1f} | per year {per} | n/yr {nper}')
            out[f'{S}|{thr}|{hh}']={'n':int(len(v)),'mean':float(v.mean()),'t':float(t),'per_year':per}
    # what does z do: correlation of z with past 4h return (is the ratio just lagging price?)
    past=(lo[np.minimum(ei-1,N-1)]-lo[np.maximum(ei-241,0)])*1e4;msk=np.isfinite(z)&ok
    print('  corr(z, past 4h return) = %.3f'%np.corrcoef(z[msk],past[msk])[0,1])
json.dump(out,open('/home/user/xauusd/reports/crypto/lsratio_detail.json','w'),indent=1)
