"""Causal intermarket features on the session decision grid (08:00-17:55 UTC every 5 min, days from
xau_sessions.npz). For every instrument, values use only bars with open timestamp < t (bar available at +1min):
ret{5,15,60,240,1440} bp from last known close, vol60 (std of last 60 observed 1-min log returns, bp),
log1p(quote age, minutes). Missing history -> NaN (LightGBM handles it; no fabricated zeros).
Also gold forward returns (bp, entry Open at t -> Open at first bar >= t+h within session) for h=15/30/60/120."""
import sys,json,numpy as np,pandas as pd
from pathlib import Path
RAW=Path(sys.argv[1] if len(sys.argv)>1 else '/home/user/data_raw/all222');S=np.load('/home/user/data_prep/xau_sessions.npz')
OUT=Path(sys.argv[2] if len(sys.argv)>2 else '/home/user/data_prep/intermarket.npz')
man=json.load(open(Path(__file__).resolve().parents[1]/'meta_reward_transformer_v3/drive_manifest.json'))
names=sorted(f['name'] for f in man)
days=pd.to_datetime(S['day']).tz_localize('UTC');dec=np.arange(120)*5+480
T=(days.values[:,None]+(dec*60*10**9).astype('timedelta64[ns]')[None,:]).reshape(-1)  # decision times
Tn=T.astype('datetime64[ns]').astype(np.int64)
feat=[];fn=[]
for k,nm in enumerate(names):
    d=pd.read_parquet(RAW/nm,columns=['timestamp','close']);ts=pd.to_datetime(d['timestamp'],utc=True).values.astype('datetime64[ns]').astype(np.int64)
    o=np.argsort(ts,kind='stable');ts=ts[o];c=np.log(d['close'].values[o].astype(np.float64))
    r=np.r_[np.nan,np.diff(c)];cs=np.nancumsum(np.nan_to_num(r));cs2=np.cumsum(np.nan_to_num(r)**2)
    def asof(t):  # index of last bar with open < t - 1min + 1ns  <=> bar closed by t
        return np.searchsorted(ts,t-60*10**9,side='right')-1
    i0=asof(Tn);ok=i0>=0;cur=np.where(ok,c[np.maximum(i0,0)],np.nan)
    cols=[]
    for h in [5,15,60,240,1440]:
        j=asof(Tn-h*60*10**9);cols.append(np.where(ok&(j>=0),(cur-c[np.maximum(j,0)])*1e4,np.nan))
    lo=np.maximum(i0-60,0);n=(i0-lo).clip(1);m1=(cs[np.maximum(i0,0)]-cs[lo])/n;m2=(cs2[np.maximum(i0,0)]-cs2[lo])/n
    cols.append(np.where(ok&(i0>=60),np.sqrt(np.maximum(m2-m1**2,0))*1e4,np.nan))
    cols.append(np.where(ok,np.log1p((Tn-ts[np.maximum(i0,0)])/6e10),np.nan))
    feat.append(np.stack(cols,1).astype(np.float32));base=nm.replace('.parquet','')
    fn+=[f'{base}:{x}' for x in ['ret5','ret15','ret60','ret240','ret1440','vol60','age']]
    if k%20==0:print(k,nm,flush=True)
F=np.concatenate(feat,1)
# gold forward returns from the session open arrays
g=pd.read_parquet(RAW/'XAUUSD_41.parquet',columns=['timestamp','open']);gts=pd.to_datetime(g['timestamp'],utc=True).values.astype('datetime64[ns]').astype(np.int64);go=g['open'].values
ent_i=np.searchsorted(gts,Tn);ent_ok=(ent_i<len(gts))&(gts[np.minimum(ent_i,len(gts)-1)]==Tn);ent=go[np.minimum(ent_i,len(gts)-1)]
send=(days.values[:,None]+np.timedelta64(18*60,'m')).repeat(120,1).reshape(-1).astype('datetime64[ns]').astype(np.int64)
fwd={}
for h in [15,30,60,120]:
    te=np.minimum(Tn+h*6*10**10,send);j=np.searchsorted(gts,te);j=np.minimum(j,len(gts)-1);okx=gts[j]<=send
    fwd[f'fwd{h}']=np.where(ent_ok&okx,(go[j]-ent)/ent*1e4,np.nan).astype(np.float32)
np.savez(OUT,x=F,names=np.array(fn),day=np.repeat(S['day'],120),slot=np.tile(np.arange(120),len(S['day'])),entry=ent.astype(np.float32),entry_ok=ent_ok,**fwd)
print('done',F.shape,{k:float(np.nanstd(v)) for k,v in fwd.items()})
