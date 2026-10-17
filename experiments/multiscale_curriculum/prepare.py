"""Build regular-time, past-only multiscale inputs and 120-minute targets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def minute(s):return int(np.datetime64(s,'m').astype(int))
def build(source,out):
    z=np.load(source);ts=z['timestamp'];assert np.all(ts%60==0) and np.all(np.diff(ts)>0)
    start,end=minute('2023-12-17'),minute('2024-03-01');mask=(ts//60>=start)&(ts//60<end)
    ix=ts[mask]//60-start;price=np.rint(np.exp(z['log_prices'][mask])*100)/100
    # Original columns H,O,C,L. All internal arrays O,H,L,C.
    price=price[:,[1,0,3,2]];n=end-start;raw=np.full((n,4),np.nan);raw[ix]=price;observed=np.zeros(n,bool);observed[ix]=True
    if not observed[0]:raw[0]=round(float(np.exp(z['log_prices'][np.searchsorted(ts,start*60)-1,2]))*100)/100
    last=np.maximum.accumulate(np.where(observed,np.arange(n),0));filled=raw.copy();filled[~observed]=raw[last[~observed],3,None]
    grids={};clocks={}
    for step in [1,5,60]:
        a=filled.reshape(-1,step,4);o=observed.reshape(-1,step)
        grids[step]=np.column_stack([a[:,0,0],a[:,:,1].max(1),a[:,:,2].min(1),a[:,-1,3],o.mean(1)]).astype('float32')
        clocks[step]=(start+np.arange(len(a))*step).astype('int64')
    close=filled[:,3];returns=np.diff(close,prepend=close[0]);cs=np.r_[0.,np.cumsum(returns)];css=np.r_[0.,np.cumsum(returns**2)]
    candidates=np.arange(360,n-120+1);full=np.r_[0,np.cumsum(observed)]
    candidates=candidates[(full[candidates+120]-full[candidates]==120)&observed[candidates-1]&(candidates>=14*1440)]
    # Scale uses only 359 close changes inside 360 visible minutes. Weekend zeros are explicit.
    def scales(i):
        count=359;mu=(cs[i]-cs[i-359])/count
        return np.sqrt(np.maximum((css[i]-css[i-359])/count-mu**2,0))*np.sqrt(60)
    splits={'train':('2024-01-01','2024-02-01',1),'val':('2024-02-01','2024-02-08',15),'test':('2024-02-15','2024-03-01',15)}
    ids={s:candidates[(candidates+start>=minute(a))&(candidates+start+120<=minute(b))&((candidates+start)%stride==0)] for s,(a,b,stride) in splits.items()}
    floor=max(float(np.quantile(scales(ids['train']),.05)),.1);arrays={f'grid_{s}':g for s,g in grids.items()};arrays.update({f'clock_{s}':v for s,v in clocks.items()});arrays['start_minute']=np.array(start);meta={}
    for split,i in ids.items():
        scale=np.maximum(scales(i),floor);anchor=close[i-1];future=filled[i[:,None]+np.arange(120)]
        def stats(a,size):
            a=a.reshape(len(a),-1,size,4)
            return np.stack([a[:,:,:,3].mean(-1),a[:,:,:,2].min(-1),a[:,:,:,1].max(-1)],-1)
        final=(stats(future,15)-anchor[:,None,None])/scale[:,None,None]
        coarse=(stats(future,30)[:,:,0]-anchor[:,None])/scale[:,None]
        past=(stats(filled[i[:,None]+np.arange(-120,0)],30)-anchor[:,None,None])/scale[:,None,None]
        activity=[]
        for h in [60,120]:
            a=future[:,:h];variation=np.abs(np.diff(np.column_stack([anchor,a[:,:,3]]),axis=1)).sum(1)
            activity.extend([(a[:,:,1].max(1)-a[:,:,2].min(1))/scale,variation/scale])
        for k,v in dict(origins=i,scale=scale,anchor=anchor,final=final,coarse=coarse,past=past,activity=np.array(activity).T).items():arrays[f'{split}_{k}']=v.astype('int64' if k=='origins' else 'float32')
        meta[split]=dict(examples=len(i),first_origin=str(np.datetime64(int(start+i[0]),'m')),last_origin=str(np.datetime64(int(start+i[-1]),'m')),stride_minutes=splits[split][2],period=splits[split][:2])
        assert np.isfinite(final).all() and np.all(final[:,:,1]<=final[:,:,0]) and np.all(final[:,:,0]<=final[:,:,2])
    # Prove aggregation availability for every selected origin, including partial hours.
    for i in ids.values():
        for step,length in [(1,360),(5,576),(60,336)]:
            stop=i//step;assert np.all(stop>=length);assert np.all(stop*step<=i)
    meta.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scale_floor_dollars=floor,contexts={'M1':360,'M5':576,'H1':336},feature_columns=['O','H','L','C','observed_fraction','time_sin','time_cos','log_scale'],targets=['mean_Close','min_Low','max_High'],quantiles=[.1,.5,.9],timestamp_convention='Source timestamp treated as candle start; forecast origin is end of the last visible M1 bar.',missing='Forward-filled OHLC, observed fraction marks missing minutes; future horizon must have all 120 observed minutes.')
    np.savez_compressed(out,**arrays);out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();build(a.source,a.out)
