"""Raw Binance aggTrades -> taker-order events + per-second and per-minute tables (no hand aggregation of the event stream itself).
usage: prep.py SYMBOL ZIP_OR_CSV [ZIP_OR_CSV ...] OUT.npz
Event = consecutive aggTrades with the same transact_time and aggressor side (one taker order, possibly sweeping levels).
Event features (float16): 0 log1p(dt_ms) 1 side(+1 buy/-1 sell) 2 log10(notional$)-3 3 log1p(levels-1)
                          4 bp move of first fill vs previous order's last price (clip +-50) 5 bp sweep move first->last (signed, clip +-50)
Per-second table (contiguous seconds): offset into events, n events, last price, and summary
  [log1p n, signed log-notional, log1p notional/1e3, ret bp vs prev second, range bp, max log-notional]; per-minute table likewise."""
import sys,zipfile,numpy as np,pandas as pd
S=sys.argv[1];srcs=sys.argv[2:-1];out=sys.argv[-1]
def read(p):
    if p.endswith('.zip'):
        z=zipfile.ZipFile(p);f=z.open(z.namelist()[0])
    else:f=open(p)
    d=pd.read_csv(f,usecols=['price','quantity','transact_time','is_buyer_maker'],dtype={'price':np.float64,'quantity':np.float64,'transact_time':np.int64,'is_buyer_maker':bool})
    return d
d=pd.concat([read(p) for p in srcs]).sort_values('transact_time',kind='stable')
ts=d.transact_time.values;px=d.price.values;nt=px*d.quantity.values;sd=np.where(d.is_buyer_maker.values,-1,1).astype(np.int8)
if ts.max()>1e14:ts=ts//1000
br=np.r_[True,(np.diff(ts)!=0)|(np.diff(sd)!=0)];gid=np.cumsum(br)-1;st=np.where(br)[0];en=np.r_[st[1:]-1,len(px)-1]
o_ts=ts[st];o_sd=sd[st];o_nt=np.bincount(gid,nt);o_lv=np.bincount(gid);p0=px[st];p1=px[en]
N=len(o_ts);dt=np.r_[0,np.diff(o_ts)].astype(np.float64)
prev_last=np.r_[p0[0],p1[:-1]]
F=np.stack([np.log1p(dt),o_sd.astype(np.float64),np.log10(np.maximum(o_nt,1))-3,np.log1p(o_lv-1),
            np.clip(1e4*np.log(p0/prev_last),-50,50),np.clip(1e4*np.log(p1/p0)*1,-50,50)],1).astype(np.float16)
def table(unit_ms):
    u=o_ts//unit_ms;u0,u1=u[0],u[-1];T=u1-u0+1;k=(u-u0).astype(np.int64)
    cnt=np.bincount(k,minlength=T);off=np.r_[0,np.cumsum(cnt)[:-1]]
    sn=np.bincount(k,o_sd*o_nt,minlength=T);tn=np.bincount(k,o_nt,minlength=T)
    last=np.full(T,np.nan);last[k]=p1     # last write wins = last order's price in the unit (events are time-sorted)
    last=pd.Series(last).ffill().bfill().values
    hi=pd.Series(np.maximum(p0,p1)).groupby(k).max().reindex(range(T)).values;lo=pd.Series(np.minimum(p0,p1)).groupby(k).min().reindex(range(T)).values
    mx=pd.Series(o_nt).groupby(k).max().reindex(range(T)).fillna(0).values
    ret=np.r_[0,1e4*np.diff(np.log(last))];rng=np.nan_to_num(1e4*np.log(hi/lo))
    summ=np.stack([np.log1p(cnt),np.sign(sn)*np.log1p(np.abs(sn)/1e3),np.log1p(tn/1e3),np.clip(ret,-100,100),np.clip(rng,0,200),np.log10(np.maximum(mx,1))-3],1).astype(np.float32)
    return int(u0),off.astype(np.int64),cnt.astype(np.int32),last,summ
s0,s_off,s_cnt,s_last,s_sum=table(1000);m0,_,_,m_last,m_sum=table(60000)
np.savez_compressed(out,sym=S,ev=F,s0=s0,s_cnt=s_cnt,s_last=s_last,s_sum=s_sum,m0=m0,m_last=m_last,m_sum=m_sum)
print(f'{S}: aggTrades {len(ts)} -> events {N}; seconds {len(s_cnt)} (max events/s {s_cnt.max()}, p99 {int(np.quantile(s_cnt,.99))}); minutes {len(m_last)}; '
      f'event feature ranges {[ (float(F[:,j].min()),float(F[:,j].max())) for j in range(6)]}')
