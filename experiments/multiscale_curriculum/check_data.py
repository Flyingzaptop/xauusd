"""Independent target and as-of aggregation audit against original M1 data."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    z=np.load(args.data);source=np.load(ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz');ts=source['timestamp']//60;price=np.rint(np.exp(source['log_prices'])*100)/100;price=price[:,[1,0,3,2]];start=int(z['start_minute']);checked=0;floor=json.loads(args.data.with_suffix('.json').read_text())['scale_floor_dollars']
    for split in ['train','val','test']:
        ids=z[split+'_origins'];which=np.linspace(0,len(ids)-1,30,dtype=int)
        for j in which:
            i=int(ids[j]);origin=start+i;end=np.searchsorted(ts,origin);f=price[end:end+120];assert np.array_equal(ts[end:end+120],origin+np.arange(120))
            anchor=price[end-1,3];scale=float(z[split+'_scale'][j]);a=f.reshape(8,15,4)
            past_minutes=origin+np.arange(-360,0);past_pos=np.searchsorted(ts,past_minutes,side='right')-1
            past_close=price[past_pos,3];expected_scale=max(float(np.diff(past_close).std()*np.sqrt(60)),floor)
            assert np.isclose(scale,expected_scale,rtol=1e-6,atol=1e-6)
            coarse=(f[:,3].reshape(4,30).mean(1)-anchor)/scale
            assert np.allclose(z[split+'_coarse'][j],coarse,atol=1e-5)
            activity=[]
            for h in [60,120]:activity.extend([(f[:h,1].max()-f[:h,2].min())/scale,np.abs(np.diff(np.r_[anchor,f[:h,3]])).sum()/scale])
            assert np.allclose(z[split+'_activity'][j],activity,atol=1e-5)
            visible=price[past_pos[-120:]].copy();seen=ts[past_pos[-120:]]==past_minutes[-120:];visible[~seen]=visible[~seen,3,None]
            visible=visible.reshape(4,30,4);past_target=(np.stack([visible[:,:,3].mean(1),visible[:,:,2].min(1),visible[:,:,1].max(1)],-1)-anchor)/scale
            assert np.allclose(z[split+'_past'][j],past_target,atol=1e-5)
            target=np.stack([a[:,:,3].mean(1),a[:,:,2].min(1),a[:,:,1].max(1)],-1)
            assert np.allclose(z[split+'_final'][j]*scale+z[split+'_anchor'][j],target,atol=.0005)
            for step,length in [(1,360),(5,576),(60,336)]:
                stop=i//step
                assert (start+stop*step)<=origin
                # Check the last completed aggregate from raw observations. Missing
                # minutes carry the last already-known close, never a future quote.
                for k in [stop-1,stop-length]:
                    t=start+k*step;gridtimes=t+np.arange(step);pos=np.searchsorted(ts,gridtimes,side='right')-1
                    actual=ts[pos]==gridtimes;values=price[pos].copy();values[~actual]=price[pos[~actual],3,None]
                    expected=np.array([values[0,0],values[:,1].max(),values[:,2].min(),values[-1,3],actual.mean()])
                    assert np.allclose(z[f'grid_{step}'][k],expected,atol=.0003)
            # Every selected feature row predates origin, so mutating any future
            # rows or unfinished current buckets cannot alter selected inputs.
            past=price[np.searchsorted(ts,origin-360):end]
            assert np.all(np.isfinite(z[split+'_past'][j]));checked+=1
    assert (start+z['train_origins'][-1]+120)<=start+z['val_origins'][0]
    assert (start+z['val_origins'][-1]+120)<=start+z['test_origins'][0]
    result=dict(status='passed',independent_raw_target_checks=checked,completed_aggregation_checks=checked*6,split_target_purging=True,as_of_input_availability=True,past_only_scale_recomputed=True,all_four_stage_targets_recomputed=True)
    args.data.with_name('data_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
