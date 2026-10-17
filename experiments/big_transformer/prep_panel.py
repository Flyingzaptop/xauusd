"""Build the M1 panel for all 222 instruments on a weekday-minute UTC grid, plus gold targets.
panel_kgc[k, t, c] (float16) describes the bar that OPENED at minute t (available from t+1):
  c0 = log-return of close vs previous known close (0 if bar missing), scaled by instrument train std
  c1 = log(high/low) scaled, c2 = 1 if the bar exists.
Targets (float32, NaN if unavailable): gold log-return in bp from Open[t+1] (decision at t+1, after bar t closed,
plus 1 minute of latency is NOT assumed here) ... see TARGET note below.
TARGET: decision made at minute d = t+1 (bar t just closed). Entry at Open of minute d+1 (1-minute latency),
exit at Open of first bar >= d+1+h. y_h = 1e4*log(exit/entry), h in 15,30,60,120.
Also vol60 = trailing std of gold 1-min returns (bp) up to bar t, used to standardize targets."""
import sys,json,numpy as np,pandas as pd
from pathlib import Path
RAW=Path(sys.argv[1]);OUT=Path(sys.argv[2]);MAN=json.load(open(sys.argv[3]))
names=sorted(f['name'] for f in MAN);OUT.mkdir(parents=True,exist_ok=True)
t0=pd.Timestamp('2016-09-05',tz='UTC');t1=pd.Timestamp('2026-09-05',tz='UTC')
grid=pd.date_range(t0,t1,freq='1min',tz='UTC',inclusive='left');grid=grid[grid.dayofweek<5];G=len(grid)
gi=grid.values.astype('datetime64[ns]').astype(np.int64)
TRAIN_END=np.datetime64('2020-12-31T23:59','ns').astype(np.int64)  # scaling stats from earliest fold's train only
panel=np.lib.format.open_memmap(OUT/'panel_kgc.npy',mode='w+',dtype=np.float16,shape=(len(names),G,3))  # instrument-major; transpose on load
scale={}
for k,nm in enumerate(names):
    d=pd.read_parquet(RAW/nm,columns=['timestamp','high','low','close']);ts=pd.to_datetime(d['timestamp'],utc=True).values.astype('datetime64[ns]').astype(np.int64)
    o=np.argsort(ts,kind='stable');ts=ts[o];c=np.log(d['close'].values[o]);hl=np.log(d['high'].values[o]/d['low'].values[o])
    keep=np.r_[True,np.diff(ts)>0];ts,c,hl=ts[keep],c[keep],hl[keep]
    r=np.r_[0.,np.diff(c)]  # vs previous known close (gaps included)
    pos=np.searchsorted(gi,ts);ok=(pos<G)&(gi[np.minimum(pos,G-1)]==ts)
    R=np.zeros(G,np.float32);HL=np.zeros(G,np.float32);M=np.zeros(G,np.float16)
    R[pos[ok]]=r[ok];HL[pos[ok]]=hl[ok];M[pos[ok]]=1
    bad=~(np.isfinite(R)&np.isfinite(HL));R[bad]=0;HL[bad]=0;M[bad]=0  # zero/invalid prices -> treated as missing bar
    tr=(gi<=TRAIN_END)&(M>0);sr=float(R[tr].std()+1e-9) if tr.sum()>100 else float(R[M>0].std()+1e-9);sh=float(HL[tr].std()+1e-9) if tr.sum()>100 else float(HL[M>0].std()+1e-9)
    panel[k]=np.stack([np.clip(R/sr,-20,20),np.clip(HL/sh,0,20),M.astype(np.float32)],1).astype(np.float16);scale[nm]=[sr,sh]
    if k%25==0:print(k,nm,int(M.astype(np.int64).sum()),flush=True)
panel.flush()
g=pd.read_parquet(RAW/'XAUUSD_41.parquet',columns=['timestamp','open','close']);gts=pd.to_datetime(g['timestamp'],utc=True).values.astype('datetime64[ns]').astype(np.int64)
o=np.argsort(gts,kind='stable');gts=gts[o];go=g['open'].values[o];gc=np.log(g['close'].values[o])
Y={}
d_ns=gi+60*10**9  # decision time = after bar t closed
ent_t=d_ns+60*10**9;ei=np.searchsorted(gts,ent_t);ei=np.minimum(ei,len(gts)-1);eok=(gts[ei]-ent_t)<=2*60*10**9  # entry within 2 min
for h in [15,30,60,120]:
    xt=ent_t+h*60*10**9;xi=np.minimum(np.searchsorted(gts,xt),len(gts)-1);xok=(gts[xi]-xt)<=5*60*10**9
    Y[f'y{h}']=np.where(eok&xok,1e4*np.log(go[xi]/go[ei]),np.nan).astype(np.float32)
gr=np.r_[0,np.diff(gc)]*1e4;gpos=np.searchsorted(gi,gts);gok=(gpos<len(gi))&(gi[np.minimum(gpos,len(gi)-1)]==gts)
GR=np.full(G,np.nan,np.float32);GR[gpos[gok]]=gr[gok]
vol60=pd.Series(GR).rolling(60,min_periods=20).std().values.astype(np.float32)
np.savez(OUT/'targets.npz',grid_ns=gi,vol60=vol60,entry=go[ei].astype(np.float32),entry_ok=eok,**Y)
(OUT/'meta.json').write_text(json.dumps({'names':names,'G':G,'scale':scale,'gold_index':names.index('XAUUSD_41.parquet')}))
print('done',G,len(names))
