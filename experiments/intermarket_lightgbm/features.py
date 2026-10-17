"""Causal calendar-minute features; closed bars only; no calendar compression."""
import numpy as np,pandas as pd
BASE=[f'return_{h}' for h in [1,5,15,30,60,240,1440]]+['logvol_15','logvol_60','logvol_240','vol_ratio','position_15','position_60','position_240','range_15','range_60','range_240','body','upper_wick','lower_wick','relative_tick_volume','acceleration_5','acceleration_15']
LAGS=[(f'return_{h}',lag) for h in [5,15,60] for lag in [5,15,60]]+[(f,lag) for f in ['logvol_15','position_60'] for lag in [15,60]]
NAMES=BASE+[f'{f}_lag{lag}' for f,lag in LAGS]+['log_quote_age','availability_60','log_previous_gap']
assert len(NAMES)==39

def encode(table,grid_start,grid_n,anchors):
 df=table.to_pandas(use_threads=False);df=df.sort_values('timestamp').reset_index(drop=True)
 if df['timestamp'].duplicated().any():raise ValueError('Duplicate candle timestamps')
 t=pd.DatetimeIndex(df.timestamp).as_unit('s').asi8
 if np.any(t%60):raise ValueError('Non-minute candle timestamps')
 x=df[['open','high','low','close']].to_numpy()
 valid=np.all(np.isfinite(x)&(x>0),axis=1)&(x[:,1]>=np.maximum(x[:,0],x[:,3])-1e-8)&(x[:,2]<=np.minimum(x[:,0],x[:,3])+1e-8)&(x[:,1]>=x[:,2])
 audit={'rows':len(t),'invalid_ohlc':int((~valid).sum()),'first':str(df.timestamp.iloc[0]),'last':str(df.timestamp.iloc[-1]),'metadata':{k.decode():v.decode() for k,v in (table.schema.metadata or {}).items()}}
 if not valid.all():df=df.loc[valid].reset_index(drop=True);t=t[valid]
 ix=(t-grid_start)//60;keep=(ix>=0)&(ix<grid_n);df=df.loc[keep];ix=ix[keep];t=t[keep]
 frame=pd.DataFrame(np.nan,index=np.arange(grid_n),columns=['open','high','low','close','volume'],dtype=np.float64)
 frame.iloc[ix,:]=df[['open','high','low','close','volume']].to_numpy()
 c=frame.close;lc=np.log(c);r1=lc.diff();sd={h:r1.rolling(h,min_periods=max(8,h//2)).std(ddof=0) for h in [15,60,240]};scale=sd[60].clip(lower=1e-7)
 f={}
 for h in [1,5,15,30,60,240,1440]:f[f'return_{h}']=(lc-lc.shift(h))/(scale*np.sqrt(h))
 for h in [15,60,240]:f[f'logvol_{h}']=np.log1p(sd[h]*10000)
 f['vol_ratio']=sd[15]/sd[240].clip(lower=1e-7)
 for h in [15,60,240]:
  hi=frame.high.rolling(h,min_periods=max(3,h//4)).max();lo=frame.low.rolling(h,min_periods=max(3,h//4)).min();span=(hi-lo).clip(lower=1e-12)
  f[f'position_{h}']=(c-lo)/span
  f[f'range_{h}']=np.log1p(span/(c*scale*np.sqrt(h)))
 f['body']=np.log(c/frame.open)/scale
 f['upper_wick']=np.log(frame.high/frame[['open','close']].max(axis=1))/scale
 f['lower_wick']=np.log(frame[['open','close']].min(axis=1)/frame.low)/scale
 f['relative_tick_volume']=np.log1p(frame.volume/frame.volume.rolling(60,min_periods=20).mean().clip(lower=1))
 for h in [5,15]:f[f'acceleration_{h}']=((lc-lc.shift(h))-(lc.shift(h)-lc.shift(2*h)))/(scale*np.sqrt(h))
 # f is defined only at truly observed bars; stale states retain their explicit age.
 a=(anchors-grid_start)//60;lastpos=np.searchsorted(ix,a,side='right')-1;known=lastpos>=0;lastix=ix[np.maximum(lastpos,0)]
 cols=[]
 def take(series,lag=0):
  j=lastix-lag;good=known&(j>=0);out=np.full(len(a),np.nan,dtype=np.float32);v=np.asarray(series)
  out[good]=np.clip(v[j[good]],-50,50).astype(np.float32);return out
 for name in BASE:cols.append(take(f[name]))
 for name,lag in LAGS:cols.append(take(f[name],lag))
 age=np.full(len(a),np.nan,dtype=np.float32);age[known]=np.log1p(np.minimum(a[known]-lastix[known],43200));cols.append(age)
 availability=frame.close.notna().astype(float).rolling(60,min_periods=1).mean().to_numpy();cols.append(availability[a].astype(np.float32))
 gaps=np.r_[np.nan,np.diff(t)/60-1];gap=np.full(len(a),np.nan,dtype=np.float32);gap[known]=np.log1p(np.minimum(gaps[np.maximum(lastpos[known],0)],43200));cols.append(gap)
 audit.update(anchor_known_fraction=float(known.mean()),anchor_fresh_fraction=float((known&(a==lastix)).mean()),feature_count=len(cols))
 return np.stack(cols,axis=0),audit
