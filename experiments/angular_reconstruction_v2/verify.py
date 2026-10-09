"""Recompute saved metrics, temporal aggregation and coverage independently."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import ROOT,OUT,old,COLS

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));assert manifest['status'] in ['evaluation_complete','complete']
    assert old.ref.sha(Path(__file__).with_name('train.py'))==manifest['code_sha256']
    assert old.ref.sha(Path(__file__).with_name('core.py'))==manifest['core_sha256']
    assert old.ref.sha(Path(__file__).with_name('protocol.md'))==manifest['protocol_sha256']
    assert old.ref.sha(OUT/'metrics.csv')==manifest['metrics_sha256']
    lp,ts,pieces,_,_,info=old.read_data();z=np.load(OUT/'test_origins.npz');a=z['anchors'];n=z['lengths']
    expected_a,expected_n=old.make_spans(pieces['test'],40,991)
    assert np.array_equal(a,expected_a) and np.array_equal(n,expected_n)
    for kind in ['A','B','C']:
        curve=pd.read_csv(OUT/'models'/kind/'curve.csv');assert curve.epoch.tolist()==[1,2,3,4]
        assert (curve.targets==old.target_count(pieces['train'])).all()
        cp=torch.load(OUT/'models'/kind/'best.pt',map_location='cpu',weights_only=False)
        zero=json.loads((OUT/'models'/kind/'validation_epoch_0.json').read_text())
        scores=[np.mean([r['ratio_to_linear_bridge'] for r in zero])]+curve.validation_ratio_to_bridge.tolist()
        assert abs(cp['validation_score']-min(scores))<1e-8
        assert cp['epoch']==int(np.argmin(scores))
    truth=lp[a[:,None]+np.arange(1,41)]-lp[a,2][:,None,None]
    previous=lp[a[:,None]+np.arange(40)]-lp[a,2][:,None,None]
    scale=json.loads((OUT/'scale.json').read_text());y=old.ref.encode_next(truth,previous,scale['S0'])[:,:,COLS]
    mask=np.arange(40)[None,:]<n[:,None];frame=pd.read_csv(OUT/'metrics.csv');agg=pd.read_csv(OUT/'multiscale_metrics.csv');checks=0;maximum=0.
    cases=np.broadcast_to(np.arange(len(a))[:,None],mask.shape)[mask]
    minute=((ts[a[:,None]+np.arange(1,41)])//60)[mask]
    for path in sorted((OUT/'predictions').glob('*.npz')):
        name=path.stem;s=np.load(path);p=s['relative_log_prices'];angles=s['angles']
        assert p.shape==truth.shape and np.isfinite(p).all() and np.isfinite(angles).all()
        for j,ch in enumerate(['H','C','L']):
            row=frame[(frame.model==name)&(frame.group=='ALL')&(frame.channel==ch)].iloc[0]
            values={'angle_mse':np.mean((angles[:,:,j]-y[:,:,j])[mask]**2),
                'price_mse_bps2':np.mean((p[:,:,COLS[j]]-truth[:,:,COLS[j]])[mask]**2)*1e8}
            for key,v in values.items():
                err=abs(v-row[key])/max(abs(row[key]),1e-12);assert err<1e-7;maximum=max(maximum,err);checks+=1
        # Pandas grouping independent of Torch aggregation used in training/evaluation.
        for k in [3,5,10]:
            d=pd.DataFrame(dict(case=cases,block=minute//k,pH=p[:,:,0][mask],pC=p[:,:,2][mask],pL=p[:,:,3][mask],
                tH=truth[:,:,0][mask],tC=truth[:,:,2][mask],tL=truth[:,:,3][mask]))
            g=d.groupby(['case','block'],sort=False).agg(count=('pH','size'),pH=('pH','max'),pC=('pC','last'),pL=('pL','min'),tH=('tH','max'),tC=('tC','last'),tL=('tL','min'))
            g=g[g['count']==k]
            for ch in ['H','C','L']:
                row=agg[(agg.model==name)&(agg.timeframe==f'M{k}')&(agg.channel==ch)].iloc[0]
                value=np.mean((g['p'+ch]-g['t'+ch])**2)*1e8
                assert len(g)==int(row.complete_bars)
                err=abs(value-row.price_mse_bps2)/max(abs(row.price_mse_bps2),1e-12);assert err<1e-7;maximum=max(maximum,err);checks+=1
    result=dict(status='passed',independent_metric_checks=checks,max_relative_difference=maximum,
        all_training_epochs_full=True,all_test_origins_verified=True,validation_only_selection_verified=True,
        utc_multiscale_aggregation_independently_verified=True,code_sha256=old.ref.sha(Path(__file__)))
    old.dump(OUT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
