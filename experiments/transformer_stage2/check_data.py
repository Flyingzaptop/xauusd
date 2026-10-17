"""Independent future targets and actual future-mutation leakage test."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/transformer_stage1'))
from representation import aggregate,encode,LAGS

def rebuild(raw,observed,s0,h):
    result={}
    for step in [1,5,60]:
        g=aggregate(raw,step);f=np.zeros((len(g),37),np.float32)
        for j,lag in enumerate(LAGS):f[lag:,4*j:4*j+4]=encode(g[lag:],g[:-lag],s0*(step*lag)**h)
        f[:,-1]=observed.reshape(-1,step).mean(1);result[step]=f
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();z=np.load(a.data);s=np.load(ROOT/'reports/transformer_stage1_20261012/dataset.npz');rawsource=np.load(ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz');ts=rawsource['timestamp']//60;lp=rawsource['log_prices'];start=int(z['start_minute']);count=0
    for step in [1,5,60]:assert np.array_equal(z[f'features_{step}'],s[f'features_{step}'])
    for split in ['train','val','test']:
        ids=z[split+'_origins']
        for j in np.linspace(0,len(ids)-1,30,dtype=int):
            origin=start+int(ids[j]);end=np.searchsorted(ts,origin);assert np.array_equal(ts[end:end+120],origin+np.arange(120))
            anchor=float(np.exp(lp[end-1,2]));future=np.exp(lp[end:end+120]);targets=[]
            for h in [60,120]:targets.append([(future[:h,0].max()-future[:h,3].min())/anchor,np.abs(np.diff(np.r_[anchor,future[:h,2]])).sum()/anchor])
            assert np.allclose(z[split+'_relative'][j],targets,rtol=1e-10,atol=1e-12)
            assert np.allclose(z[split+'_y'][j]*z['target_scale'],targets,rtol=1e-6)
            assert np.isclose(z[split+'_anchor_price'][j],anchor)
            for h_index,h in enumerate([60,120]):
                historical=np.exp(lp[end-1-h:end,2])/anchor
                proxy=[historical.max()-historical.min(),np.abs(np.diff(historical)).sum()]
                assert np.allclose(z[split+'_past_proxy'][j,h_index],proxy,rtol=1e-5,atol=1e-9)
            for step,length in [(1,360),(5,576),(60,336)]:assert (int(ids[j])//step)*step<=int(ids[j]) and int(ids[j])//step-length>=55
            count+=1
    # Rebuild vector features after replacing ALL unobserved-at-origin prices.
    # Includes an origin inside an unfinished H1 bucket, the classic leakage trap.
    n=len(z['features_1']);observed=np.zeros(n,bool);raw=np.full((n,4),np.nan);mask=(ts>=start)&(ts<start+n);ix=ts[mask]-start;raw[ix]=lp[mask];observed[ix]=True
    if not observed[0]:raw[0]=lp[np.searchsorted(ts,start)-1,2]
    last=np.maximum.accumulate(np.where(observed,np.arange(n),0));raw[~observed]=raw[last[~observed],2,None]
    s0=float(z['s0']);h=float(z['hurst']);chosen=[int(z['train_origins'][0]),int(z['train_origins'][len(z['train_origins'])//2]),int(z['test_origins'][0])];changed_targets=0
    for i in chosen:
        mutated=raw.copy();mutated[i:]+=np.linspace(.5,2.,n-i)[:,None];f=rebuild(mutated,observed,s0,h)
        for step,length in [(1,360),(5,576),(60,336)]:
            indexes=i//step+np.arange(-length,0);assert np.array_equal(f[step][indexes],z[f'features_{step}'][indexes])
        anchor=np.exp(raw[i-1,2]);before=np.exp(raw[i:i+120]);after=np.exp(mutated[i:i+120]);old=(before[:,0].max()-before[:,3].min())/anchor;new=(after[:,0].max()-after[:,3].min())/anchor
        assert not np.isclose(old,new);changed_targets+=1
    assert np.allclose(z['target_scale'],np.maximum(np.median(z['train_relative'],axis=0),1e-8))
    result=dict(status='passed',stage=2,future_targets_recomputed=count,stage1_vectors_unchanged=True,future_mutation_cases=changed_targets,future_changes_inputs=False,future_changes_targets=True,unfinished_H1_excluded=True,past_proxy_from_vectors_only=True,normalization_train_only=True,no_stage3=True)
    a.data.with_name('data_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
