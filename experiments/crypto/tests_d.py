"""Part D tests: OI/price quadrants and positioning ratios (see PROTOCOL.md Part D)."""
import glob,json,sys,numpy as np,pandas as pd
sys.path.insert(0,'/home/user/xauusd/experiments/crypto');sys.path.insert(0,'/home/user/xauusd/experiments/scan222')
import kernels as K
src=open('/home/user/xauusd/experiments/crypto/tests.py').read();g={};exec(src.split('\nRES=[]\n')[0],g);summ,wf_dir=g['summ'],g['wf_dir']
RES=[]
for S in ('BTCUSDT','ETHUSDT'):
    fs=sorted(glob.glob(f'/home/user/crypto/um/{S}/metrics/*.csv'))
    m=pd.concat([pd.read_csv(f) for f in fs]);m['t']=pd.to_datetime(m.create_time,utc=True);m=m.drop_duplicates('t').sort_values('t').set_index('t')
    yrs=m.index.year.value_counts().sort_index();print(S,'metrics rows',len(m),'by year',yrs.to_dict(),flush=True)
    k=pd.read_parquet(f'/home/user/crypto/{S}_1m.parquet');kt=k.open_time.values;O=k.open.values;lo=np.log(O);N=len(O)
    # metrics on a 5-min grid; value usable from create_time+5min
    g5=m.resample('5min').last()
    avail=(g5.index+pd.Timedelta(minutes=5)).values.astype('datetime64[ms]').astype(np.int64)
    ei=np.searchsorted(kt,avail)+1;ok=ei+250<N
    px=lo[np.minimum(ei-1,N-1)]                 # last open before availability (approx price at signal time)
    year=pd.DatetimeIndex(g5.index).year.values
    oi=np.log(g5.sum_open_interest.astype(float).values)
    for w,lab in ((3,'15m'),(12,'60m')):
        dp=px-np.roll(px,w);doi=oi-np.roll(oi,w);dp[:w]=np.nan;doi[:w]=np.nan
        qp=pd.Series(np.abs(dp)).rolling(8640,min_periods=2000).quantile(.9).shift(1).values   # trailing 30 days of 5-min rows
        qo=pd.Series(np.abs(doi)).rolling(8640,min_periods=2000).quantile(.9).shift(1).values
        big=ok&(np.abs(dp)>qp)&(np.abs(doi)>qo)&np.isfinite(qp)&np.isfinite(qo)
        for cls,cond in {'price_up_OI_up':(dp>0)&(doi>0),'price_up_OI_down(short squeeze)':(dp>0)&(doi<0),'price_dn_OI_up':(dp<0)&(doi>0),'price_dn_OI_down(long liq)':(dp<0)&(doi<0)}.items():
            idx=np.where(big&cond)[0]
            for hh,hl in ((60,'1h'),(240,'4h')):
                ii=idx[K.nonoverlap(ei[idx],hh)];e=ei[ii];y=(lo[e+hh]-lo[e])*1e4
                gg=wf_dir(np.sign(dp[ii]),y,year[ii]);kp=np.isfinite(gg)
                r=summ(f'{S} D1 {lab} {cls} -> {hl}',e[kp],gg[kp],pd.to_datetime(kt,unit='ms').year.values);r and RES.append(r)
    for col,lab in (('count_long_short_ratio','acct L/S'),('sum_toptrader_long_short_ratio','top-trader pos L/S'),('sum_taker_long_short_vol_ratio','taker vol L/S')):
        x=np.log(g5[col].astype(float).values);z=(x-pd.Series(x).rolling(8640,min_periods=2000).mean().shift(1).values)/pd.Series(x).rolling(8640,min_periods=2000).std().shift(1).values
        idx=np.where(ok&(np.abs(z)>2)&np.isfinite(z))[0]
        for hh,hl in ((60,'1h'),(240,'4h')):
            ii=idx[K.nonoverlap(ei[idx],hh)];e=ei[ii];y=(lo[e+hh]-lo[e])*1e4;gg=wf_dir(np.sign(z[ii]),y,year[ii]);kp=np.isfinite(gg)
            r=summ(f'{S} D2 {lab} |z|>2 -> {hl}',e[kp],gg[kp],pd.to_datetime(kt,unit='ms').year.values);r and RES.append(r)
json.dump(RES,open('/home/user/xauusd/reports/crypto/tests_d.json','w'),indent=1)
print(f"{'test':58} {'n':>5} {'gross':>7} {'t':>5} | net@4 t pos | net@10 t pos")
for r in RES:print(f"{r['test']:58} {r['n']:5d} {r['gross_bp']:+7.2f} {r['t_gross']:+5.1f} | {r['net4']:+7.2f} {r['t4']:+5.1f} {r['pos4']} | {r['net10']:+7.2f} {r['t10']:+5.1f} {r['pos10']}{'  <== CANDIDATE' if r['candidate'] else ''}")
