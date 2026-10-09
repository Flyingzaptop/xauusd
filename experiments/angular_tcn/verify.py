"""Independent reconstruction of reported scores, split checks and inference parity."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import ROOT,OUT,SEEDS,CHANNELS,sha,AngularHistory,model
from predict import predict_frame,read_recent

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    assert sha(Path(__file__).with_name('run.py'))==manifest['script_sha256']
    assert sha(Path(__file__).with_name('protocol.md'))==manifest['protocol_sha256']
    source=ROOT/'data/M1/xauusd_m1.parquet';assert sha(source)==manifest['source_sha256']
    assert sha(OUT/'metrics.csv')==manifest['metrics_sha256']
    df=pd.read_parquet(source);ts=df.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
    lp=np.log(df[['high','open','close','low']].to_numpy(float))
    seg=np.cumsum(np.r_[True,np.diff(ts)!=60]);splits=np.load(OUT/'split_indices.npz')
    bounds=[int(pd.Timestamp(x,tz='UTC').timestamp()) for x in ['2025-01-01','2026-01-01','2026-09-01']]
    for split in splits.files:
        ix=splits[split];assert np.all(seg[ix-182]==seg[ix+1])
        if split=='train':assert np.all(ts[ix+1]<bounds[0])
        if split=='val':assert np.all(ts[ix-182]>=bounds[0]) and np.all(ts[ix+1]<bounds[1])
        if split=='test':assert np.all(ts[ix-182]>=bounds[1]) and np.all(ts[ix+1]<bounds[2])
        if split=='tail':assert np.all(ts[ix-182]>=bounds[2])
    scale=json.loads((OUT/'scale.json').read_text(encoding='utf-8'));s0=scale['S0']
    metrics=pd.read_csv(OUT/'metrics.csv');checks=0;max_error=0
    for split in ['test','tail']:
        z=np.load(OUT/'predictions'/f'{split}.npz');ix=z['origins']
        true=(2/np.pi)*np.arctan((lp[ix+1]-lp[ix])/s0)
        for name in z.files:
            if name in ['origins','target','timestamp']:continue
            a=z[name].astype(float);decoded=lp[ix]+s0*np.tan(np.pi*np.clip(a,-1+1e-6,1-1e-6)/2)
            assert np.isfinite(decoded).all()
            for j,ch in enumerate(CHANNELS):
                row=metrics[(metrics.split==split)&(metrics.model==name)&(metrics.channel==ch)&(metrics.period=='ALL')].iloc[0]
                for key,value in [('angle_mse',np.mean((a[:,j]-true[:,j])**2)),
                    ('angle_mae_degrees',np.mean(abs(a[:,j]-true[:,j]))*90),
                    ('price_mse_bps2',np.mean((decoded[:,j]-lp[ix+1,j])**2)*1e8)]:
                    err=abs(value-row[key])/max(abs(row[key]),1e-9);max_error=max(max_error,err)
                    assert err<1e-4,(split,name,ch,key,err);checks+=1
    # Validate checkpoints against the logged validation minima, independently of test.
    for seed in SEEDS:
        curve=pd.read_csv(OUT/'models'/str(seed)/'curve.csv')
        ck=torch.load(OUT/'models'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
        assert abs(curve.val_mse.min()-ck['val_mse'])<1e-8
        assert int(curve.loc[curve.val_mse.idxmin(),'epoch'])==ck['best_epoch']
    # Run production inference on an evaluation point using ONLY its 183 past bars.
    z=np.load(OUT/'predictions/test.npz');n=len(z['origins'])//2;origin=int(z['origins'][n])
    forecast=predict_frame(df.iloc[origin-182:origin+1])
    new=np.array([forecast['angle_normalized'][ch] for ch in CHANNELS])
    single_parity=float(np.max(abs(new-z['ensemble'][n])))
    # TF32 kernels differ with batch shape. Compare saved batch512 evaluation with
    # the same 183-bar history repeated512 times: no additional market information.
    history=AngularHistory(lp[origin-182:origin+1],scale['S0'],scale['H'])
    x=history.x([182]).repeat(512,1,1);matched=[]
    for seed in SEEDS:
        ck=torch.load(OUT/'models'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
        net=model(seed,ck['state']).eval()
        with torch.no_grad():matched.append(net(x)[0].cpu().numpy())
    parity=float(np.max(abs(np.mean(matched,axis=0)-z['ensemble'][n])))
    assert parity<5e-5,parity
    diagnostic=json.loads((OUT/'precision_diagnosis.json').read_text(encoding='utf-8'))
    assert diagnostic['input_exactly_equal']
    assert diagnostic['by_precision']['ieee_fp32']['single_vs_batch512_max_difference']<1e-6
    result=dict(status='passed',source_code_protocol_and_metric_hashes_match=True,full_window_split_and_continuity_checks=True,
        independently_recomputed_metrics=checks,max_relative_metric_error=max_error,
        best_checkpoints_match_validation=True,matched_batch_past_only_inference_max_angle_difference=parity,
        standalone_tf32_single_batch_max_difference_degrees=single_parity*90,
        precision_cause='Identical input; TF32 batch-size rounding. Discrepancy disappears in IEEE FP32; see precision_diagnosis.json.',
        verifier_sha256=sha(Path(__file__)))
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
