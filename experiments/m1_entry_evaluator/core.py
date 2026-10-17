"""Exact retrospective entry consequences; no learned score or reward weights."""
import numpy as np
import pandas as pd
H=120
HORIZONS=(5,15,30,60,120)
LEVELS=(.25,.5,1.,2.)

def metadata(ts,close):
    n=len(ts);i=np.arange(n)
    if np.any(np.diff(ts)<=0) or np.any(ts%60):raise ValueError('Expected unique ascending UTC minute opens')
    ends=np.r_[np.flatnonzero((np.diff(ts)!=60)|(ts[1:]//86400!=ts[:-1]//86400)),n-1]
    end=ends[np.searchsorted(ends,i)]
    length=np.minimum(H,end-i+1).astype('int16')
    prev=np.maximum(0,i-H)
    past=(i>=H)&(ts-ts[prev]==H*60)
    delta=np.r_[np.nan,np.diff(close)]
    # Exactly 120 prior bars -> 119 price differences, all before entry.
    rms=pd.Series(delta**2).rolling(H-1,min_periods=H-1).mean().shift(1).to_numpy()
    scale=np.sqrt(H*rms);scale[~past|(scale<=1e-9)]=np.nan
    return length,past,scale

def block(ts,ohlc,length,past,scale,start,stop):
    ids=np.arange(start,stop);steps=np.arange(H);n=length[ids]
    j=np.minimum(ids[:,None]+steps,len(ts)-1);valid=steps[None,:]<n[:,None]
    entry=ohlc[ids,0];cl=np.where(valid,ohlc[j,3]-entry[:,None],np.nan)
    hi=np.where(valid,ohlc[j,1]-entry[:,None],np.nan);lo=np.where(valid,ohlc[j,2]-entry[:,None],np.nan)
    out={'timestamp':pd.to_datetime(ts[ids],unit='s',utc=True),'entry_open':entry,
         'available_future_minutes':n,'full_120_intraday':n==H,'past_120_complete':past[ids],
         'past_scale_dollars':scale[ids]}
    for prefix,sign in [('long',1),('short',-1)]:
        q=sign*cl;favourable=hi if sign==1 else -lo;adverse=lo if sign==1 else -hi
        mean=np.nanmean(q,axis=1);quant=np.nanquantile(q,[.1,.5,.9],axis=1)
        out[prefix+'_mean_pnl']=mean
        for k,x in zip(['q10_pnl','median_pnl','q90_pnl'],quant):out[prefix+'_'+k]=x
        out[prefix+'_max_favourable']=np.maximum(0,np.nanmax(favourable,axis=1))
        out[prefix+'_max_adverse']=np.maximum(0,-np.nanmin(adverse,axis=1))
        out[prefix+'_profitable_close_fraction']=np.sum(q>0,axis=1)/n
        out[prefix+'_entire_candle_profitable_fraction']=np.sum(adverse>0,axis=1)/n
        out[prefix+'_mean_downside']=np.nanmean(np.maximum(-q,0),axis=1)
        running=np.maximum.accumulate(np.maximum(np.nan_to_num(q,nan=0),0),axis=1)
        out[prefix+'_max_close_drawdown']=np.nanmax(running-q,axis=1)
        for h in HORIZONS:out[f'{prefix}_pnl_{h}m']=np.where(n>=h,q[:,h-1],np.nan)
        # First hit minute-bin only; exact intra-bar order is not known.
        for level in LEVELS:
            for typ,path in [('profit',favourable),('loss',-adverse)]:
                hit=valid&(path>=scale[ids,None]*level)
                first=np.argmax(hit,axis=1)+1
                out[f'{prefix}_first_{typ}_{level:g}s_min']=np.where(hit.any(axis=1),first,np.nan)
    return pd.DataFrame({k:(v.astype('float32') if isinstance(v,np.ndarray) and v.dtype.kind=='f' else v) for k,v in out.items()})
