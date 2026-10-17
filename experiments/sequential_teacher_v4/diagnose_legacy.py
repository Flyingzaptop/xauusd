"""Actual v2/v3 tensors required: never substitute synthetic or missing instruments."""
from pathlib import Path
import argparse,json,importlib.util
import numpy as np,torch

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--inputs',required=True,help='npz past[N,222,120,96],clock[N,8],future[N,120,8], optional net[N,2]');a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True);args=a.parse_args();torch.set_num_threads(2)
    src=Path(__file__).parent.parent/'meta_reward_transformer_v3/model.py';spec=importlib.util.spec_from_file_location('legacy_model',src);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    z=np.load(args.inputs);x=z['past'];c=z['clock'];f=z['future'];assert x.shape[1:]==(222,120,96) and c.shape==(len(x),8) and f.shape==(len(x),120,8);assert len(x)>=32,'At least 32 representative points required'
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=ck['config'];model=m.MarketTransformer(teacher=True,d=cfg['hidden'],depth=cfg['depth_temporal'],patch=cfg['patch']).eval();model.load_state_dict(ck['model'])
    order=np.random.default_rng(87).permutation(len(x));scores={}
    for mode in ['actual','future_shuffle','past_shuffle','zero_future']:
        out=[];variance=[]
        with torch.no_grad():
            for j in range(0,len(x),4):
                ix=np.arange(j,min(j+4,len(x)));pi=order[ix] if mode=='past_shuffle' else ix;fi=order[ix] if mode=='future_shuffle' else ix
                future=np.zeros_like(f[fi]) if mode=='zero_future' else f[fi]
                d=model(torch.tensor(x[pi]),torch.tensor(c[pi]),torch.tensor(future));out.append(d.mean.numpy());variance.append(d.variance.numpy())
        scores[mode]=np.concatenate(out);scores[mode+'_sampling_variance']=np.concatenate(variance)
    actual=scores['actual'];report={'rows':len(x),'cycle':ck.get('cycle'),'mean':actual.mean(0).tolist(),'std':actual.std(0).tolist(),'sampling_std':np.sqrt(scores['actual_sampling_variance']).mean(0).tolist(),'ablation_changes':{k:np.abs(scores[k]-actual).mean(0).tolist() for k in ['future_shuffle','past_shuffle','zero_future']}}
    if 'net' in z:report['correlation_to_net']=[float(np.corrcoef(actual[:,j],z['net'][:,j])[0,1]) for j in range(2)]
    p=Path(args.out);p.mkdir(parents=True,exist_ok=True);np.savez_compressed(p/'all_scores.npz',**scores);(p/'diagnostic.json').write_text(json.dumps(report,indent=2))
