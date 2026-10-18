"""Candidate spec (written after M1-M4, so NOT new evidence): BTC+ETH, signal = mean of sign(P - SMA_n) for n in 20/50/100,
per-asset vol target, portfolio scaled to 20% annual vol (trailing 60 d), daily at 00:00 UTC, 10 bp per unit traded, funding on
perps (2020+), spot before. Prints stats per year 2017-2026 and saves an equity curve. usage: candidate.py OUT_DIR"""
import sys,io,zipfile,subprocess,json,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
O=sys.argv[1]
def spot(s):
    mos=[f'{y}-{m:02d}' for y in range(2017,2020) for m in range(1,13) if '2017-08'<=f'{y}-{m:02d}'<='2019-12'];ps=[]
    for mo in mos:
        r=subprocess.run(['curl','-sS','-f','--retry','4',f'https://data.binance.vision/data/spot/monthly/klines/{s}/1d/{s}-1d-{mo}.zip'],capture_output=True)
        if not r.returncode:z=zipfile.ZipFile(io.BytesIO(r.stdout));ps.append(pd.read_csv(z.open(z.namelist()[0]),header=None))
    k=pd.concat(ps);return pd.Series(k[4].astype(float).values,index=pd.to_datetime(k[0],unit='ms')).sort_index()
def perp(s):
    k=pd.read_parquet(f'/home/user/crypto/{s}_1m.parquet');return pd.Series(k.close.values,index=pd.to_datetime(k.open_time,unit='ms')).resample('1D').last().dropna()
sl={};bh={}
for s in ('BTCUSDT','ETHUSDT'):
    p=pd.concat([spot(s),perp(s)]);p=p[~p.index.duplicated(keep='last')].sort_index()
    f=pd.read_parquet(f'/home/user/crypto/{s}_funding.parquet');f.index=pd.to_datetime(f.calc_time,unit='ms');fd=f.last_funding_rate.astype(float).resample('1D').sum().reindex(p.index).fillna(0)
    r=np.log(p).diff();w=(0.5/(r.rolling(20).std()*np.sqrt(365))).clip(upper=2)
    sig=sum(np.sign(p-p.rolling(n).mean()) for n in (20,50,100))/3
    sl[s]=(sig*w);bh[s]=r;FD=locals().setdefault('FD',{});FD[s]=fd
# notional positions: sleeve vol-normalized, portfolio scaled to 20% vol, GROSS LEVERAGE CAPPED AT 2x; decided on yesterday's data
S=pd.DataFrame(sl);R=pd.DataFrame(bh);Fd=pd.DataFrame(FD)
raw=(S.shift(1)*R).dropna();sc=1/raw.rolling(60,min_periods=30).std().shift(1)          # equalize sleeve risk
pre=(S.shift(1)*sc/2);pr=(pre*R).sum(1,min_count=1);ps=(0.20/np.sqrt(365))/pr.rolling(60,min_periods=30).std().shift(1)
CAP=float(__import__("os").environ.get("CAP","1"));N=pre.mul(ps,axis=0);g=N.abs().sum(1);N=N.div((g/CAP).clip(lower=1),axis=0)   # cap gross leverage (default 1x)
port=((N*R).sum(1,min_count=1)-(N*Fd).sum(1)-N.diff().abs().sum(1)*10e-4);port=port[port.index>='2017-12-15'].dropna()
print('gross leverage: median %.2f, p95 %.2f, share capped %.3f'%(N.abs().sum(1).median(),N.abs().sum(1).quantile(.95),(g>CAP).mean()))
B=pd.DataFrame(bh).reindex(port.index).mean(1)
def st(r):
    a=r.mean()*365;v=r.std()*np.sqrt(365);eq=r.cumsum();return {'ann_ret_log':round(float(a),3),'ann_vol':round(float(v),3),'sharpe':round(float(a/v),2),'maxdd_log':round(float((eq.cummax()-eq).max()),3)}
res={'all':st(port),'by_year':{str(y):{**st(port[port.index.year==y]),'buyhold_ret':round(float(B[B.index.year==y].sum()),2)} for y in sorted(set(port.index.year))},'buyhold_all':st(B)}
print(json.dumps(res,indent=1));json.dump(res,open(f'{O}/candidate.json','w'),indent=1)
fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(np.exp(port.cumsum()),label='trend BTC+ETH, 20% vol target',color='#2a6fdb')
ax.plot(np.exp(B.cumsum()),label='buy & hold BTC+ETH 50/50',color='#999',alpha=.8);ax.set_yscale('log');ax.grid(alpha=.3);ax.legend();ax.set_title('Daily trend following, 2017-11 .. 2026-09 (costs, funding included)')
for x in ('2020-01-01','2024-01-01'):ax.axvline(pd.Timestamp(x),color='k',lw=.6,ls='--')
fig.tight_layout();fig.savefig(f'{O}/candidate_equity.png',dpi=110)
