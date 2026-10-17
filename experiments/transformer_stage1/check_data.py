"""Verify raw aggregation, vector summation, reversible angles and input limits."""
import argparse,json
from pathlib import Path
import numpy as np
from representation import encode,decode,aggregate,encode_path,decode_path,LAGS
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();z=np.load(a.data);source=np.load(ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz');ts=source['timestamp']//60;raw=source['log_prices'];start=int(z['start_minute']);s0=float(z['s0']);h=float(z['hurst']);s15=float(z['s15']);count=0;oracle_error=0.
    for split in ['train','val','test']:
        ids=z[split+'_origins']
        for j in np.linspace(0,len(ids)-1,30,dtype=int):
            i=int(ids[j]);origin=start+i;end=np.searchsorted(ts,origin);assert np.array_equal(ts[end-415:end],origin+np.arange(-415,0))
            path=raw[end-120:end];target=aggregate(path,15);anchor=aggregate(raw[end-135:end-120],15).reshape(4)
            assert np.allclose(target,z[split+'_target_log'][j],atol=1e-12);assert np.allclose(anchor,z[split+'_anchor_log'][j],atol=1e-12)
            assert np.allclose(encode_path(target,anchor,s15),z[split+'_angles'][j],atol=1e-7)
            # Exact vector oracle: angle -> displacement, cumsum -> full M1 path,
            # max/min -> aggregated candles. Anchor is for reconstruction only.
            vectors=z['features_1'][i-120:i,:4];restored=decode_path(vectors,raw[end-121],s0);oracle=aggregate(restored,15)
            oracle_error=max(oracle_error,float(np.max(np.abs(np.exp(oracle)-np.exp(target)))))
            # Target vectors also recover from the input alone: reconstruct 135
            # minutes in an arbitrary zero coordinate, then take coarse deltas.
            relative=decode_path(z['features_1'][i-135:i,:4],np.zeros(4),s0)
            coarse=aggregate(relative,15)
            oracle_angles=encode_path(coarse[1:],coarse[0],s15)
            assert np.allclose(oracle_angles,z[split+'_angles'][j],atol=2e-6)
            for lagindex,lag in enumerate(LAGS):
                actual=encode(raw[end-1],raw[end-1-lag],s0*lag**h);assert np.allclose(actual,z['features_1'][i-1,lagindex*4:lagindex*4+4],atol=1e-7)
                summed=np.diff(raw[end-1-lag:end],axis=0).sum(0);assert np.allclose(summed,raw[end-1]-raw[end-1-lag],atol=1e-12)
            for step,length in [(1,360),(5,576),(60,336)]:
                stop=i//step;assert stop-length>=55 and (start+stop*step)<=origin
                k=stop-1;t=start+k*step;times=t+np.arange(step);pos=np.searchsorted(ts,times,side='right')-1;obs=ts[pos]==times
                current=raw[pos].copy();current[~obs]=raw[pos[~obs],2,None]
                times2=t-step+np.arange(step);pos2=np.searchsorted(ts,times2,side='right')-1;obs2=ts[pos2]==times2
                previous=raw[pos2].copy();previous[~obs2]=raw[pos2[~obs2],2,None]
                expected=encode(aggregate(current,step).reshape(4),aggregate(previous,step).reshape(4),s0*step**h)
                assert np.allclose(z[f'features_{step}'][k,:4],expected,atol=1e-7)
                assert np.isclose(z[f'features_{step}'][k,-1],obs.mean())
            # Translation of log-price coordinate (multiplicative USD rescale)
            # leaves vectors unchanged, and correctly rescales decoded paths.
            shift=np.log(2.3);assert np.allclose(encode_path(target+shift,anchor+shift,s15),encode_path(target,anchor,s15),atol=1e-11)
            count+=1
    assert oracle_error<.005,oracle_error
    e=np.array([[0.,0.,0.,0.],[2.,-1.,1.,-2.]])*s0;angles=encode(e[1],e[0],s0)
    assert np.allclose(decode(angles,e[0],s0),e[1])
    assert not np.allclose(2*encode(e[1],e[0],s0),encode(2*e[1],e[0],s0))
    result=dict(status='passed',raw_target_checks=count,vector_input_checks=count*12,completed_past_only_context=True,no_absolute_prices_in_model_inputs=True,log_coordinate_translation_invariance=True,angles_not_added_directly=True,anchor_free_target_vector_oracle=True,float32_vector_oracle_max_dollar_error=oracle_error,stage=1)
    a.data.with_name('data_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
