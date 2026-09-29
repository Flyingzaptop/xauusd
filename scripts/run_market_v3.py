"""Frozen, continuous-state M1 research run. Fit and exposed evaluation are separate."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.market_data_v3 import load_split
from src.stream_batch_v3 import PackedStream
from src.stream_models_v3 import StreamForecaster, matched_baseline_widths, parameter_count

SOURCE_FILES=['src/crossing_memory_v2.py','src/market_data_v3.py','src/stream_batch_v3.py',
              'src/stream_models_v3.py','scripts/run_market_v3.py']


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def provenance(cfg,path):
    cache=Path(cfg['output'])
    audit=json.loads((cache/'audit.json').read_text())
    if audit['provenance']['config_sha256']!=digest(path):
        raise RuntimeError('Prepared data configuration differs; regenerate before freezing')
    return dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_sha256={f:digest(f) for f in SOURCE_FILES},config_sha256=digest(path),
                data_sha256=audit['provenance']['data_sha256'],
                cache_sha256={name:digest(cache/f'{name}.npz') for name in ['train','validation','evaluation']},
                python=sys.version,torch=torch.__version__,numpy=np.__version__)


def tensors(batch):
    return {k:{n:torch.from_numpy(a) for n,a in value.items()} if isinstance(value,dict)
            else torch.from_numpy(value) for k,value in batch.items()}


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64)
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    n=len(y)
    if not n:
        raise ValueError('Empty scored population')
    loss=-y*np.log(p)-(1-y)*np.log1p(-p)
    bins=np.minimum((p*10).astype(int),9)
    calibration=[]
    ece=0.
    for b in range(10):
        keep=bins==b
        if keep.any():
            predicted=float(p[keep].mean())
            observed=float(y[keep].mean())
            ece+=float(keep.mean())*abs(predicted-observed)
            calibration.append(dict(bin=b,count=int(keep.sum()),predicted=predicted,observed=observed))
    return dict(n=n,log_loss=float(loss.mean()),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                brier=float(np.square(p-y).mean()),ece=ece,calibration=calibration)


@torch.no_grad()
def predict(model,split,cfg):
    model.eval()
    result=np.full(len(split['y']),np.nan,dtype=np.float32)
    state=None
    for raw in PackedStream(split,cfg['batch_size'],cfg['chunk_length']):
        batch=tensors(raw)
        logits,state=model(batch['x'],batch['tree'],batch['reset'],state)
        real=raw['indices']>=0
        result[raw['indices'][real]]=logits.numpy()[real]
    if not np.isfinite(result).all():
        raise FloatingPointError('Missing or nonfinite stream predictions')
    return expit(result).astype(np.float64)


def make_model(kind,cfg,widths):
    return StreamForecaster(kind,widths.get(kind,cfg['c_gru_hidden']),cfg['memory_hidden'],4)


def fit_seed(seed,cfg,prov,model_filter=None):
    cache=Path(cfg['output'])
    train=load_split(cache,'train')
    validation=load_split(cache,'validation')
    packer=PackedStream(train,cfg['batch_size'],cfg['chunk_length'])
    widths=matched_baseline_widths(cfg['c_gru_hidden'],cfg['memory_hidden'],4)
    frozen_budget=json.loads(Path('reports/v3/frozen-budget.json').read_text())
    expected=frozen_budget['splits']['train']['optimizer_updates_per_epoch']
    for kind in ([model_filter] if model_filter is not None else cfg['models']):
        torch.manual_seed(seed)
        model=make_model(kind,cfg,widths)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        best=float('inf')
        curve=[]
        batch_curve=[]
        elapsed_start=time.perf_counter()
        updates_total=0
        state_max=dict(gru=0.,tree=0.)
        max_grad=0.
        for epoch in range(1,cfg['epochs']+1):
            model.train()
            state=None
            loss_sum=0.
            seen=updates=0
            epoch_start=time.perf_counter()
            for step,raw in enumerate(packer,1):
                batch=tensors(raw)
                if state is not None:
                    state=model.detach_state(state)
                logits,state=model(batch['x'],batch['tree'],batch['reset'],state)
                for name,value in state.items():
                    if value is not None:
                        if not torch.isfinite(value).all():
                            raise FloatingPointError('Nonfinite carried '+name+' state')
                        state_max[name]=max(state_max[name],float(value.detach().abs().max()))
                keep=batch['valid']
                count=int(keep.sum())
                if count:
                    loss=nn.functional.binary_cross_entropy_with_logits(logits[keep],batch['y'][keep])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite train loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                    max_grad=max(max_grad,float(norm))
                    optimizer.step()
                    updates+=1
                    updates_total+=1
                    value=float(loss.detach())
                    loss_sum+=value*count
                    seen+=count
                    batch_curve.append(dict(epoch=epoch,chunk=step,update=updates_total,n=count,loss=value,
                                            gradient_norm_before_clip=float(norm)))
                if step%100==0:
                    print(json.dumps(dict(model=kind,seed=seed,epoch=epoch,chunk=step,chunks=len(packer),
                                          supervised=seen,train_log_loss_so_far=loss_sum/max(seen,1))),flush=True)
            if updates!=expected or seen!=int(train['valid'].sum()):
                raise RuntimeError('Actual data/update count differs from frozen budget')
            vp=predict(model,validation,cfg)
            vm=metrics(validation['y'][validation['valid']],vp[validation['valid']])
            row=dict(epoch=epoch,train_log_loss=loss_sum/seen,supervised=seen,optimizer_updates=updates,
                     validation=vm,elapsed_seconds=time.perf_counter()-epoch_start)
            curve.append(row)
            if vm['log_loss']<best:
                best=vm['log_loss']
                best_epoch=epoch
                torch.save(model.state_dict(),cache/f'{kind}-{seed}-best.pt')
                np.savez_compressed(cache/f'{kind}-{seed}-validation.npz',probability=vp[validation['valid']])
            print(json.dumps(dict(model=kind,seed=seed,epoch=epoch,train_log_loss=row['train_log_loss'],
                                  validation_log_loss=vm['log_loss'],updates=updates)),flush=True)
        record=dict(model=kind,seed=seed,parameters=parameter_count(model),gru_hidden=model.gru_hidden,
                    checkpoint_sha256=digest(cache/f'{kind}-{seed}-best.pt'),
                    validation_predictions_sha256=digest(cache/f'{kind}-{seed}-validation.npz'),
                    training_seconds=time.perf_counter()-elapsed_start,epoch_curve=curve,batch_curve=batch_curve,
                    total_optimizer_updates=updates_total,supervised_examples_per_epoch=int(train['valid'].sum()),
                    best_epoch=best_epoch,maximum_state_abs=state_max,maximum_gradient_norm_before_clip=max_grad,
                    all_gradients_finite=True)
        Path(f'reports/v3/fit-{kind}-{seed}.json').write_text(json.dumps(dict(provenance=prov,config=cfg,
                        runs=[record],complete=True),indent=2))
        print(json.dumps(dict(finished=kind,seed=seed,best_epoch=best_epoch,seconds=record['training_seconds'])),flush=True)


def evaluate_all(cfg,prov):
    cache=Path(cfg['output'])
    fits=[]
    for seed in cfg['seeds']:
        for kind in cfg['models']:
            path=Path(f'reports/v3/fit-{kind}-{seed}.json')
            if not path.exists():
                raise RuntimeError('Incomplete or mismatched fit provenance; no evaluation')
            report=json.loads(path.read_text())
            if not report['complete'] or report['provenance']!=prov or report['config']!=cfg:
                raise RuntimeError('Incomplete or mismatched fit provenance; no evaluation')
            fits.extend(report['runs'])
    expected={(model,seed) for model in cfg['models'] for seed in cfg['seeds']}
    actual=[(run['model'],run['seed']) for run in fits]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise RuntimeError('Missing or duplicate model/seed fits; no evaluation')
    for run in fits:
        stem=f"{run['model']}-{run['seed']}"
        if digest(cache/f'{stem}-best.pt')!=run['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint hash differs: '+stem)
    split=load_split(cache,'evaluation')
    keep=split['valid']
    y=split['y'][keep]
    ts=split['timestamp'][keep]
    months=pd.to_datetime(ts,unit='s',utc=True).strftime('%Y-%m')
    widths=matched_baseline_widths(cfg['c_gru_hidden'],cfg['memory_hidden'],4)
    rows=[]
    predictions={}
    for run in fits:
        kind,seed=run['model'],run['seed']
        model=make_model(kind,cfg,widths)
        model.load_state_dict(torch.load(cache/f'{kind}-{seed}-best.pt',weights_only=True))
        p=predict(model,split,cfg)[keep]
        predictions[(kind,seed)]=p
        rows.append(dict(model=kind,seed=seed,period='exposed_evaluation',**metrics(y,p)))
        for month in sorted(set(months)):
            mask=months==month
            rows.append(dict(model=kind,seed=seed,period=month,**metrics(y[mask],p[mask])))
        print(json.dumps(dict(evaluated=kind,seed=seed,log_loss=rows[-len(set(months))-1]['log_loss'])),flush=True)
    train=load_split(cache,'train')
    p_constant=float(train['y'][train['valid']].mean())
    constant=np.full(len(y),p_constant)
    rows.append(dict(model='constant',seed=None,period='exposed_evaluation',**metrics(y,constant)))
    for month in sorted(set(months)):
        mask=months==month
        rows.append(dict(model='constant',seed=None,period=month,**metrics(y[mask],constant[mask])))
    def logloss(p):
        p=np.clip(p,1e-7,1-1e-7)
        return -y*np.log(p)-(1-y)*np.log1p(-p)
    losses={model:np.stack([logloss(predictions[(model,seed)]) for seed in cfg['seeds']]) for model in cfg['models']}
    losses['constant']=np.repeat(logloss(constant)[None,:],len(cfg['seeds']),axis=0)
    weeks=(ts+3*86400)//(7*86400)
    unique=np.unique(weeks)
    counts=np.array([(weeks==week).sum() for week in unique])
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    sampled=rng.integers(0,len(unique),size=(cfg['bootstrap_replicates'],len(unique)))
    contrasts=[('c','b'),('b','a'),('c','a'),('c','no_transfer'),('c','no_open')]+[(m,'constant') for m in cfg['models']]
    comparisons=[]
    for left,right in contrasts:
        per_seed=losses[left]-losses[right]
        difference=per_seed.mean(0)
        sums=np.array([difference[weeks==week].sum() for week in unique])
        draws=sums[sampled].sum(1)/counts[sampled].sum(1)
        comparisons.append(dict(comparison=f'{left} - {right}',difference=float(difference.mean()),
                                ci95=np.quantile(draws,[.025,.975]).tolist(),weeks=len(unique),
                                per_seed={str(seed):float(per_seed[i].mean()) for i,seed in enumerate(cfg['seeds'])}))
    output=dict(provenance=prov,config=cfg,evaluation_status='exposed exploratory research, not independent confirmation',
                constant_train_probability=p_constant,metrics=rows,paired_weekly_intervals=comparisons)
    Path('reports/v3/evaluation.json').write_text(json.dumps(output,indent=2))
    pd.DataFrame([{k:v for k,v in row.items() if k!='calibration'} for row in rows]).to_csv('reports/v3/metrics.csv',index=False)
    np.savez_compressed(cache/'evaluation-predictions.npz',timestamp=ts,y=y,
                        **{f'{model}-{seed}':p for (model,seed),p in predictions.items()})
    print(json.dumps(comparisons,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/market-v3.json')
    p.add_argument('--phase',choices=['fit','evaluate'],required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--model',choices=['a','b','c','no_transfer','no_open'])
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    Path('reports/v3').mkdir(parents=True,exist_ok=True)
    prov=provenance(cfg,args.config)
    if args.phase=='fit':
        if args.seed not in cfg['seeds']:
            raise ValueError('Use a preregistered --seed')
        fit_seed(args.seed,cfg,prov,args.model)
    else:
        evaluate_all(cfg,prov)


if __name__=='__main__':
    main()
