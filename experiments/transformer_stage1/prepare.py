"""Only stage one: describe eight rolling 15-minute candles of visible past."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from representation import LAGS,encode,aggregate,encode_path,decode_path
ROOT=Path(__file__).resolve().parents[2]
def minute(s):return int(np.datetime64(s,'m').astype(int))
def build(source,out):
    z=np.load(source);ts=z['timestamp']//60;assert np.all(np.diff(ts)>0)
    start,end=minute('2023-12-10'),minute('2024-03-01');selected=(ts>=start)&(ts<end);ix=ts[selected]-start;logp=z['log_prices'][selected];n=end-start
    raw=np.full((n,4),np.nan);raw[ix]=logp;observed=np.zeros(n,bool);observed[ix]=True
    if not observed[0]:raw[0]=z['log_prices'][np.searchsorted(ts,start)-1,2]
    last=np.maximum.accumulate(np.where(observed,np.arange(n),0));raw[~observed]=raw[last[~observed],2,None]
    grids={};availability={}
    for step in [1,5,60]:grids[step]=aggregate(raw,step).reshape(-1,4);availability[step]=observed.reshape(-1,step).mean(1)
    c=np.r_[0,np.cumsum(observed)];origins=np.arange((336+55)*60,n+1);origins=origins[c[origins]-c[origins-415]==415]
    periods={'train':('2024-01-01','2024-02-01',1),'val':('2024-02-01','2024-02-08',15),'test':('2024-02-15','2024-03-01',15)}
    ids={s:origins[(start+origins>=minute(a))&(start+origins<minute(b))&((start+origins)%stride==0)] for s,(a,b,stride) in periods.items()}
    # Fit the same family of lag-angle scales on TRAIN observations only.
    t=np.arange(minute('2024-01-01')-start,minute('2024-02-01')-start);scales=[]
    for lag in LAGS:
        good=(c[t+1]-c[t-lag]==lag+1);a=t[good];scales.append(float(np.median(np.abs(raw[a,2]-raw[a-lag,2]))))
    s0=max(scales[0],1e-8);h=float(np.clip(np.polyfit(np.log(LAGS),np.log(np.maximum(scales,1e-8)),1)[0],.1,.9));s15=s0*15**h
    arrays={'start_minute':np.array(start),'s0':np.array(s0),'hurst':np.array(h),'s15':np.array(s15)};meta={}
    for step,g in grids.items():
        features=np.zeros((len(g),37),np.float32)
        for j,lag in enumerate(LAGS):features[lag:,j*4:j*4+4]=encode(g[lag:],g[:-lag],s0*(step*lag)**h)
        features[:,-1]=availability[step];arrays[f'features_{step}']=features
    for split,i in ids.items():
        visible=raw[i[:,None]+np.arange(-120,0)];target=aggregate(visible,15)
        anchor=aggregate(raw[i[:,None]+np.arange(-135,-120)],15).reshape(-1,4)
        before=aggregate(raw[i[:,None]+np.arange(-150,-135)],15).reshape(-1,4)
        angle=encode_path(target,anchor,s15);assert np.allclose(decode_path(angle,anchor,s15),target,atol=1e-12)
        arrays.update({f'{split}_origins':i,f'{split}_angles':angle.astype('float32'),f'{split}_target_log':target,f'{split}_anchor_log':anchor,f'{split}_previous_vector':encode(anchor,before,s15).astype('float32')})
        meta[split]=dict(examples=len(i),stride_minutes=periods[split][2],first_origin=str(np.datetime64(int(start+i[0]),'m')),last_origin=str(np.datetime64(int(start+i[-1]),'m')),max_abs_angle=float(np.abs(angle).max()))
    meta.update(stage=1,task='Reconstruct eight rolling 15-minute candles from the last 120 visible minutes; last four candles describe the last 60 minutes.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),contexts={'M1':360,'M5':576,'H1':336},channels=['H','O','C','L'],lags=LAGS.tolist(),s0=s0,hurst=h,s15=s15,scale_fit='Median absolute TRAIN close displacement; log-lag regression exponent, clipped to [0.1,0.9].',input='Only angular log-price displacement vectors and observation fractions; learned relative-position/scale embeddings in model. No prices, anchors, volatility levels, future candles, or target values in forward input.',target='Same angular per-channel displacements between consecutive reconstructed 15-minute OHLC candles.',aggregate='Sum raw log-channel displacements before angle encoding; reconstruct path, max High/first Open/last Close/min Low.',availability='Last 415 M1 minutes must be real consecutive candles; longer calendar histories forward-filled with observation fraction.',splits=periods,exposure='Research-exposed raw data; stage one deterministic past reconstruction, not forecasting validation.')
    np.savez_compressed(out,**arrays);out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz');p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.source,a.out)
