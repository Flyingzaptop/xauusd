"""Cross-venue funding carry Binance vs Hyperliquid (see PROTOCOL.md). usage: run.py DATA_DIR OUT.json"""
import sys,glob,json,numpy as np,pandas as pd
D,OUT=sys.argv[1:3];ENTER,EXIT,COST=3.0,1.0,20.0
out={};pooled=[]
for fn in sorted(glob.glob(f'{D}/hl_*.parquet')):
    c=fn.split('hl_')[1][:-8]
    try:h=pd.read_parquet(fn);b=pd.read_parquet(f'{D}/bn_{c}.parquet')
    except Exception:continue
    if not len(h) or not len(b):continue
    h['t']=pd.to_datetime(h.time,unit='ms',utc=True);h['r']=h.fundingRate.astype(float)
    b['t']=pd.to_datetime(b.calc_time,unit='ms',utc=True);b['r']=b.last_funding_rate.astype(float)
    b=b.drop_duplicates('t').set_index('t').r.sort_index()
    b=b.resample('8h',closed='right',label='right').sum(min_count=1).dropna()   # sum all settlements in each 8 h window (some coins pay every 4 h)
    # HL hourly rates summed over the 8 h window ending at each Binance settlement time
    hs=h.set_index('t').r.sort_index().resample('1h').sum(min_count=1)
    h8=hs.rolling(8,min_periods=7).sum().reindex(b.index)
    s=((h8-b)*1e4).dropna()                              # spread in bp per 8 h: positive -> HL longs pay more -> short HL, long Binance
    sig=s.rolling(21,min_periods=15).mean().shift(1)     # trailing 7 days, known before the period
    pos=np.zeros(len(s));p=0
    for i,(v) in enumerate(sig.values):
        if np.isnan(v):pos[i]=0;continue
        if p==0 and abs(v)>ENTER:p=np.sign(v)
        elif p!=0 and (abs(v)<EXIT or np.sign(v)!=p):p=0
        pos[i]=p
    pos=pd.Series(pos,index=s.index);gain=pos*s;trades=(pos.diff().abs()>0).sum()/2+(pos.iloc[0]!=0)/2
    turn=(pos.diff().fillna(pos).abs());cost=turn*COST/2      # half the round-trip cost at entry, half at exit
    net=gain-cost;df=pd.DataFrame({'gain':gain,'cost':cost,'net':net,'inv':pos!=0,'coin':c});pooled.append(df)
    res={}
    for per,(a,z) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):
        x=df[(df.index>=pd.Timestamp(a,tz='UTC'))&(df.index<pd.Timestamp(z,tz='UTC'))]
        if not len(x):continue
        yrs=len(x)/(3*365);res[per]={'net_ann_pct':float(x.net.sum()/1e2/yrs),'gross_ann_pct':float(x.gain.sum()/1e2/yrs),'invested':float(x.inv.mean()),
            'net_when_invested_ann_pct':float(x.net[x.inv].sum()/1e2/(x.inv.sum()/(3*365))) if x.inv.any() else 0.,'mean_spread_bp8h':float(s.reindex(x.index).mean())}
    out[c]=res;print(c,{k:{kk:round(vv,2) for kk,vv in v.items()} for k,v in res.items()},flush=True)
P=pd.concat(pooled)
for per,(a,z) in (('2023-24',('2023','2025')),('2025-26',('2025','2027'))):
    x=P[(P.index>=pd.Timestamp(a,tz='UTC'))&(P.index<pd.Timestamp(z,tz='UTC'))]
    m=x.groupby([x.index.to_period('M'),'coin']).net.sum().groupby(level=0).mean()   # average coin's monthly net bp
    out[f'pooled|{per}']={'mean_coin_month_net_bp':float(m.mean()),'t_months':float(m.mean()/m.std()*np.sqrt(len(m))),'ann_pct_per_coin':float(m.mean()*12/100),'months':int(len(m))}
    print('POOLED',per,out[f'pooled|{per}'],flush=True)
json.dump(out,open(OUT,'w'),indent=1)
