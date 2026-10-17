"""Build per-day XAUUSD sessions (08:00-18:00 UTC, decisions every 5 min) with causal features,
entry/exit prices, money in bp and the teacher's future view. See PROTOCOL.md."""
import sys,json,hashlib,numpy as np,pandas as pd
from pathlib import Path
SRC=Path(sys.argv[1] if len(sys.argv)>1 else '/home/user/data_raw/XAUUSD_41.parquet')
OUT=Path(sys.argv[2] if len(sys.argv)>2 else '/home/user/data_prep/xau_sessions.npz')
STEP=5;N=120;H=24;COST_USD=.25;CTX=8;S0=8*60;S1=18*60
d=pd.read_parquet(SRC);d['timestamp']=pd.to_datetime(d['timestamp'],utc=True)
d=d.set_index('timestamp').sort_index();d=d[~d.index.duplicated()]
grid=pd.date_range(d.index[0].floor('D'),d.index[-1].ceil('D'),freq='1min',tz='UTC')
m=d.reindex(grid);obs=m['close'].notna().values
c=m['close'].ffill();hi=m['high'].ffill();lo=m['low'].ffill()
lc=np.log(c);r1=lc.diff().where(obs)          # only observed minute returns
F={}
for k in [1,5,15,30,60,240,1440]:F[f'ret{k}']=(lc-lc.shift(k))*1e4
for k in [15,60,240]:F[f'vol{k}']=r1.rolling(k,min_periods=max(5,k//4)).std()*1e4
for k in [60,240]:
    H_=hi.rolling(k,min_periods=1).max();L_=lo.rolling(k,min_periods=1).min();F[f'pos{k}']=(c-L_)/(H_-L_).replace(0,np.nan)-.5
last=pd.Series(np.where(obs,np.arange(len(grid)),np.nan),index=grid).ffill()
F['age']=np.log1p(np.arange(len(grid))-last)
F=pd.DataFrame(F,index=grid)
# feature value usable at decision minute t = value computed at minute t-1 (bar t-1 closed at t)
F=F.shift(1)
tod=(grid.hour*60+grid.minute)/1440;dow=grid.dayofweek/7
F['tod_s']=np.sin(2*np.pi*tod);F['tod_c']=np.cos(2*np.pi*tod);F['dow_s']=np.sin(2*np.pi*dow);F['dow_c']=np.cos(2*np.pi*dow)
names=list(F.columns);Fv=F.values.astype(np.float32)
op=m['open'].values;cl=m['close'].values;idx0=pd.Series(np.arange(len(grid)),index=grid)
days=pd.date_range(grid[0].normalize(),grid[-1].normalize(),freq='B',tz='UTC')
X=[];E=[];MON=[];FUT=[];MASK=[];DAY=[]
for day in days:
    s=idx0.get(day+pd.Timedelta(minutes=S0));e=idx0.get(day+pd.Timedelta(minutes=S1))
    if s is None or e is None or s<1500:continue
    o=op[s:e+1].copy();cc=cl[s:e+1]
    dec=np.arange(N)*STEP;entry=o[dec];mask=np.isfinite(entry)
    if mask.sum()<N//2:continue  # holiday / no session: nothing tradeable
    # exit price: first available open at/after planned exit within session, else last close <= 18:00
    nxt=pd.Series(o).bfill().values;lastc=pd.Series(cc).ffill().values[-1]
    ex=np.minimum(dec+H*STEP,S1-S0);ep=nxt[ex];ep=np.where(np.isfinite(ep),ep,lastc)
    ent=np.where(mask,entry,np.nan);delta=(ep-ent)/ent*1e4;cost=COST_USD/ent*1e4
    money=np.stack([np.zeros(N),delta-cost,-delta-cost],-1);money[~mask]=0
    fill=pd.Series(o).ffill().bfill().values
    fut=np.zeros((N,H,2),np.float32)
    for i in range(N):
        if not mask[i]:continue
        pts=dec[i]+STEP*np.arange(1,H+1);ok=pts<=S1-S0
        fut[i,ok,0]=(fill[pts[ok]]-entry[i])/entry[i]*1e4;fut[i,ok,1]=1
    rows=s+dec;ctx=np.stack([Fv[rows-STEP*(CTX-1-j)] for j in range(CTX)],1)  # [N,CTX,F]
    X.append(ctx);E.append(np.c_[ent,ep]);MON.append(money);FUT.append(fut);MASK.append(mask);DAY.append(str(day.date()))
X=np.stack(X);DAY=np.array(DAY)
tr=(DAY>='2017-01-01')&(DAY<='2022-12-31')
flat=X[tr].reshape(-1,X.shape[-1]);mu=np.nanmean(flat,0);sd=np.nanstd(flat,0)+1e-6
Xn=np.clip(np.nan_to_num((X-mu)/sd),-5,5).astype(np.float32)
OUT.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(OUT,x=Xn,money=np.stack(MON).astype(np.float32),future=np.stack(FUT),mask=np.stack(MASK),day=DAY,prices=np.stack(E).astype(np.float32),mu=mu,sd=sd,names=np.array(names))
meta={'src':str(SRC),'src_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'days':len(DAY),'first':DAY[0],'last':DAY[-1],'features':names,'N':N,'H':H,'step':STEP,'ctx':CTX,'cost_usd':COST_USD}
OUT.with_suffix('.json').write_text(json.dumps(meta,indent=1));print(json.dumps(meta)[:600])
