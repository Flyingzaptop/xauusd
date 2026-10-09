"""Independent full-coverage, optimizer-step, holdout and score verification."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import ROOT,OUT,ref,TCN,DEVICE,TARGET,chunks_for

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));assert manifest['status']=='complete'
    assert ref.sha(Path(__file__).with_name('run.py'))==manifest['script_sha256']
    assert ref.sha(Path(__file__).with_name('protocol.md'))==manifest['protocol_sha256']
    source=ROOT/'data/M1/xauusd_m1.parquet';assert ref.sha(source)==manifest['source_sha256']
    checkpoint=OUT/'checkpoints/epoch_30.pt';assert ref.sha(checkpoint)==manifest['final_checkpoint_sha256']
    assert ref.sha(OUT/'metrics.csv')==manifest['metrics_sha256']
    df=pd.read_parquet(source);ts=df.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
    price=df[['high','open','close','low']].to_numpy(float);lp=np.log(price)
    assert np.isfinite(price).all() and (price>0).all()
    seg=np.cumsum(np.r_[True,np.diff(ts)!=60]);ix=np.arange(182,len(ts)-1);ix=ix[seg[ix-182]==seg[ix+1]]
    boundary=pd.Timestamp(df.timestamp.max())-pd.DateOffset(years=1);cut=int(boundary.timestamp())
    z=np.load(OUT/'split_indices.npz');train=z['train'];holdout=z['holdout']
    assert np.array_equal(train,ix[ts[ix+1]<cut]) and np.array_equal(holdout,ix[ts[ix+1]>=cut])
    assert len(np.intersect1d(train,holdout))==0
    curve=pd.read_csv(OUT/'training_curve.csv')
    assert curve.epoch.tolist()==list(range(1,31)) and (curve.targets==len(train)).all()
    assert (curve.updates==len(chunks_for(train))).all()
    assert (curve.learning_rate<=1e-4).all() and np.all(np.diff(curve.learning_rate)<=0)
    expected_years=pd.Series(pd.to_datetime(ts[train+1],unit='s',utc=True).year).value_counts().sort_index()
    for epoch in range(1,31):
        years=pd.read_csv(OUT/f'train_years_epoch_{epoch:02d}.csv')
        assert years.targets.sum()==len(train)
        assert years.set_index('year').targets.to_dict()==expected_years.to_dict()
        row=curve[curve.epoch==epoch].iloc[0]
        for channel in ['H','C','L']:
            reconstructed=float((years[channel]*years.targets).sum()/years.targets.sum())
            assert abs(reconstructed-row[f'loss_{channel}'])<1e-6
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['epoch']==30 and ck['dropout']==.25 and ck['targets']==['H','C','L']
    steps=[int(v['step'].item()) for v in ck['optimizer']['state'].values()]
    assert all(v==int(curve.updates.sum()) for v in steps)
    net=TCN().to(DEVICE);net.load_state_dict(ck['model']);net.eval()
    scale=ck['scale'];s0=scale['S0'];preds=np.load(OUT/'holdout_predictions.npz');assert np.array_equal(preds['origins'],holdout)
    target=(2/np.pi)*np.arctan((lp[holdout+1]-lp[holdout])/s0)
    assert np.max(abs(target-preds['target']))<1e-12
    metrics=pd.read_csv(OUT/'metrics.csv');nchecks=0;max_error=0
    for name in preds.files:
        if name in ['origins','timestamp','target']:continue
        p=preds[name];decoded=lp[holdout]+s0*np.tan(np.pi*np.clip(p,-1+1e-6,1-1e-6)/2)
        assert np.isfinite(decoded).all()
        assert np.max(abs(decoded[:,1]-lp[holdout,2]))<1e-10
        for col,ch in enumerate(['H','O','C','L']):
            row=metrics[(metrics.period=='ALL')&(metrics.model==name)&(metrics.channel==ch)].iloc[0]
            for key,value in [('angle_mse',np.mean((p[:,col]-target[:,col])**2)),
                ('price_mse_bps2',np.mean((decoded[:,col]-lp[holdout+1,col])**2)*1e8),
                ('angle_mae_degrees',np.mean(abs(p[:,col]-target[:,col]))*90)]:
                rel=abs(value-row[key])/max(abs(row[key]),1e-12);assert rel<1e-6;max_error=max(max_error,rel);nchecks+=1
    data=ref.AngularHistory(lp,scale['S0'],scale['H']);chunks=chunks_for(holdout)
    chosen=np.quantile(np.arange(len(chunks)),[.1,.3,.5,.7,.9]).astype(int);max_dense=0;max_window=0
    with torch.no_grad():
        for index in chosen:
            begin,n=chunks[index];x=data.a[begin-127:begin+n].T.unsqueeze(0).contiguous()
            p=net(x)[0,:,-n:].T.cpu().numpy();start=np.searchsorted(holdout,begin)
            err=float(np.max(abs(p-preds['TCN'][start:start+n][:,TARGET])));assert err<1e-6;max_dense=max(max_dense,err)
            point=n//2;origin=begin+point;single=net(data.x([origin]))[0,:,-1].cpu().numpy()
            difference=float(np.max(abs(single-p[point])));assert difference<1e-3;max_window=max(max_window,difference)
    result=dict(status='passed',full_source_hash_and_checkpoint_verified=True,full_train_and_holdout_origin_sets_verified=True,
        completed_epochs=30,targets_per_epoch=len(train),total_target_presentations=int(curve.targets.sum()),
        actual_optimizer_steps=steps[0],lr_never_above_1e_4=True,dropout=.25,independent_metric_checks=nchecks,
        max_relative_metric_difference=max_error,dense_prediction_reproduction_max_difference=max_dense,
        independent_window_vs_dense_max_difference_degrees=max_window*90,
        rounding_note='TF32 may round differently for different sequence lengths; dense equivalence also tested in CPU unit tests.',
        code_sha256=ref.sha(Path(__file__)))
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
