"""User-requested interim comparison; never updates the training process or weights."""
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
from run import ROOT,OUT,TCN,ref,DEVICE,TARGET

def main(epoch):
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    olddir=ROOT/'reports/angular_tcn_20261008'
    cached=np.load(olddir/'predictions/test.npz');ix=cached['origins']
    oldscale=json.loads((olddir/'scale.json').read_text(encoding='utf-8'))
    ck=torch.load(OUT/'checkpoints'/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False)
    frame=pd.read_parquet(ROOT/'data/M1/xauusd_m1.parquet')
    lp=np.log(frame[['high','open','close','low']].to_numpy(float));prior=lp[ix];truth=lp[ix+1]
    times=frame.timestamp.iloc[ix+1]
    assert (times>=pd.Timestamp(ck['scale']['fit_end_exclusive'])).all()
    data=ref.AngularHistory(lp,ck['scale']['S0'],ck['scale']['H'])
    net=TCN().to(DEVICE);net.load_state_dict(ck['model']);net.eval();outputs=[]
    with torch.no_grad():
        for pos in range(0,len(ix),256):outputs.append(net(data.x(ix[pos:pos+256]))[:,:,-1].cpu().numpy())
    new=np.concatenate(outputs).astype(float)
    predicted_logs={'previous_ensemble':ref.decode_log(cached['ensemble'],prior,oldscale['S0'])[:,TARGET],
        'previous_seed17':ref.decode_log(cached['seed17'],prior,oldscale['S0'])[:,TARGET],
        f'new_epoch{epoch}':ref.decode_log(new,prior[:,TARGET],ck['scale']['S0'])}
    actual=truth[:,TARGET];s0=ck['scale']['S0'];target=ref.encode_next(actual,prior[:,TARGET],s0)
    rows=[];losses={}
    for name,p in predicted_logs.items():
        angles=ref.encode_next(p,prior[:,TARGET],s0)
        error=angles-target;price_error=(p-actual)*10000
        losses[name]=dict(angle_HCL=np.mean(error**2,axis=1),price_HCL=np.mean(price_error**2,axis=1),
            price_Close=price_error[:,1]**2,angle_Close=error[:,1]**2)
        for j,label in enumerate(['H','C','L']):
            nz=actual[:,j]!=prior[:,TARGET[j]]
            rows.append(dict(model=name,channel=label,angle_mse=float(np.mean(error[:,j]**2)),
                angle_mae_degrees=float(np.mean(abs(error[:,j]))*90),price_mse_bps2=float(np.mean(price_error[:,j]**2)),
                price_mae_bps=float(np.mean(abs(price_error[:,j]))),
                direction_accuracy=float(np.mean(np.sign(p[nz,j]-prior[nz,TARGET[j]])==np.sign(actual[nz,j]-prior[nz,TARGET[j]])))))
    days,inverse=np.unique(times.to_numpy(dtype='datetime64[D]'),return_inverse=True);counts=np.bincount(inverse)
    rng=np.random.default_rng(2727);draw=rng.integers(0,len(days),(2000,len(days)))
    comparisons=[];newloss=losses[f'new_epoch{epoch}']
    for name in ['previous_ensemble','previous_seed17']:
        for metric in newloss:
            difference=newloss[metric]-losses[name][metric];sums=np.bincount(inverse,weights=difference)
            boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
            comparisons.append(dict(baseline=name,metric=metric,new=float(newloss[metric].mean()),old=float(losses[name][metric].mean()),
                improvement_pct=float(-difference.mean()/losses[name][metric].mean()*100),delta=float(difference.mean()),ci_low=float(lo),ci_high=float(hi)))
    zero_mse=float(np.mean(((truth[:,2]-prior[:,2])*10000)**2))
    result=dict(checkpoint_epoch=epoch,examples=len(ix),first_target=str(times.min()),last_target=str(times.max()),
        evaluation='User-requested interim diagnostic on exactly the previous TCN research-evaluation origins. No training changes.',
        comparability='H/C/L only; angles converted to the new common S0. Price errors use identical targets and units. Previous ensemble has3 models; new checkpoint is one model.',
        zero_close_mse_bps2=zero_mse,metrics=rows,comparisons=comparisons)
    path=OUT/f'interim_epoch_{epoch:02d}.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT/f'interim_epoch_{epoch:02d}.csv',index=False)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,required=True);args=p.parse_args();main(args.epoch)
