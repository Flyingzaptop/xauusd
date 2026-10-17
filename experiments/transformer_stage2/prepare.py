"""Future-only activity targets, with unchanged stage-one vector inputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def minute(s):return int(np.datetime64(s,'m').astype(int))
def build(first,source,out):
    s=np.load(first);z=np.load(source);ts=z['timestamp']//60;logp=z['log_prices'];start=int(s['start_minute']);arrays={k:s[k] for k in s.files if k.startswith('features_') or k in ['s0','s15','hurst','start_minute']};meta={}
    periods={'train':('2024-01-01','2024-02-01'),'val':('2024-02-01','2024-02-08'),'test':('2024-02-15','2024-03-01')}
    for split,(a,b) in periods.items():
        ids=s[split+'_origins'];ends=np.searchsorted(ts,start+ids);good=(start+ids+120<=minute(b))&(ends+119<len(ts));pos=np.minimum(ends+119,len(ts)-1)
        good&=(ts[ends]==start+ids)&(ts[pos]-ts[ends]==119)
        original=np.flatnonzero(good);ids=ids[original];ends=ends[original]
        assert np.array_equal(ts[ends[:,None]+np.arange(120)],start+ids[:,None]+np.arange(120))
        current=np.exp(logp[ends-1,2]);future=np.exp(logp[ends[:,None]+np.arange(120)]);targets=[];proxy=[]
        for h in [60,120]:
            f=future[:,:h];variation=np.abs(np.diff(np.column_stack([current,f[:,:,2]]),axis=1)).sum(1)
            targets.append(np.column_stack([(f[:,:,0].max(1)-f[:,:,3].min(1))/current,variation/current]))
            # Pure vector proxy: integrate past Close changes in an arbitrary
            # coordinate, then normalize last Close to one. No price anchor input.
            v=s['features_1'][ids[:,None]+np.arange(-h,0),2];d=float(s['s0'])*np.tan(v.astype('float64')*(np.pi/2))
            cumulative=np.column_stack([np.zeros(len(ids)),np.cumsum(d,axis=1)]);relative=np.exp(cumulative-cumulative[:,-1,None])
            proxy.append(np.column_stack([relative.max(1)-relative.min(1),np.abs(np.diff(relative,axis=1)).sum(1)]))
        y=np.stack(targets,1);past=np.stack(proxy,1)
        assert np.all(y>0) and np.all(y[:,1]>=y[:,0]-1e-12)
        for k,v in dict(origins=ids,relative=y,past_proxy=past,anchor_price=current,angles=s[split+'_angles'][original],anchor_log=s[split+'_anchor_log'][original],target_log=s[split+'_target_log'][original]).items():arrays[split+'_'+k]=v
        meta[split]=dict(examples=len(ids),removed_without_full_future=int((~good).sum()),first_origin=str(np.datetime64(int(start+ids[0]),'m')),last_origin=str(np.datetime64(int(start+ids[-1]),'m')),period=[a,b])
    scale=np.maximum(np.median(arrays['train_relative'],axis=0),1e-8);arrays['target_scale']=scale
    for split in periods:arrays[split+'_y']=(arrays[split+'_relative']/scale).astype('float32')
    arrays['proxy_floor']=np.maximum(np.quantile(arrays['train_past_proxy'],.01,axis=0),1e-8)
    meta.update(stage=2,task='Predict future 60/120-minute High-Low range and total absolute Close variation; no direction.',horizons_minutes=[60,120],metrics=['range','total_abs_close_changes'],target='Future range/current Close and future variation/current Close, normalized by fixed TRAIN medians. Relative magnitudes, not absolute price levels.',target_scale=scale.tolist(),vectors='Exact unchanged arrays from stage1. No refitting of s0/H, no future tokens.',s0=float(s['s0']),hurst=float(s['hurst']),stage1_dataset_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),exposure='Same research-exposed raw periods as stage1; stage2 targets are strictly future, stage1 targets were visible past.')
    np.savez_compressed(out,**arrays);out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage1-data',type=Path,default=ROOT/'reports/transformer_stage1_20261012/dataset.npz');p.add_argument('--source',type=Path,default=ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz');p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.stage1_data,a.source,a.out)
