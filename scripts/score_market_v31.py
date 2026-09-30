"""Separate calibration barrier and final exposed evaluation for v3.1."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.market_data_v31 import load_split
from src.provenance_v31 import get_provenance,digest
from src.inference_v31 import predict_logits,binary_metrics
from src.calibration_v31 import (fit_affine,apply_affine,SLOPE_LOWER_BOUND,
    MAX_ITERATIONS,FUNCTION_TOLERANCE,GRADIENT_TOLERANCE)
from src.stream_models_v3 import matched_baseline_widths
from scripts.run_market_v3 import make_model

CHOICES=('short_best','full_best')


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def read_fits(cfg,prov):
    records=[]
    for model in cfg['models']:
        for seed in cfg['seeds']:
            path=Path(f'reports/v31/fit-{model}-{seed}.json')
            if not path.exists():raise RuntimeError('All15 fits must finish first')
            record=json.loads(path.read_text())
            if not record['complete'] or record['provenance']!=prov or record['config']!=cfg:
                raise RuntimeError('Incomplete or mismatched fit: '+str(path))
            if (record['model'],record['seed'])!=(model,seed):raise RuntimeError('Wrong fit identity')
            for choice in CHOICES:
                selected=record['checkpoints'][choice]
                if digest(selected['path'])!=selected['sha256']:raise RuntimeError('Checkpoint hash mismatch')
            records.append(record)
    return records


def load_model(record,choice,cfg):
    cp=record['checkpoints'][choice]
    # Own, hash-verified recovery payload includes optimizer/RNG objects.
    saved=torch.load(cp['path'],map_location='cpu',weights_only=False)
    if saved['epoch']!=cp['epoch']:raise RuntimeError('Checkpoint epoch mismatch')
    if (saved['model'],saved['seed'])!=(record['model'],record['seed']):
        raise RuntimeError('Checkpoint model/seed mismatch')
    if abs(saved['selection_log_loss']-cp['selection_log_loss'])>1e-12:
        raise RuntimeError('Checkpoint selection score mismatch')
    widths=matched_baseline_widths(cfg['c_gru_hidden'],cfg['memory_hidden'],4)
    model=make_model(record['model'],cfg,widths)
    model.load_state_dict(saved['model_state_dict'])
    h=hashlib.sha256()
    for name,value in sorted(saved['model_state_dict'].items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return model,h.hexdigest()


def calibrate(cfg,prov):
    fits=read_fits(cfg,prov)
    expected_config=dict(method='positive_affine_logit',a_min=SLOPE_LOWER_BOUND,
        maxiter=MAX_ITERATIONS,ftol=FUNCTION_TOLERANCE,gtol=GRADIENT_TOLERANCE)
    if cfg['calibration']!=expected_config:raise RuntimeError('Calibrator settings differ from fixed method')
    split=load_split(cfg['output'],'calibration')
    keep=split['valid']; y=split['y'][keep]
    entries=[]; started=time.perf_counter()
    for record in fits:
        reused={}
        for choice in CHOICES:
            model,weight_hash=load_model(record,choice,cfg)
            if weight_hash in reused:
                logits,parameters=reused[weight_hash]
                reused_prediction=True
            else:
                logits=predict_logits(model,split,cfg)[keep]
                parameters=fit_affine(logits,y)
                reused[weight_hash]=(logits,parameters)
                reused_prediction=False
            calibrated,_=apply_affine(logits,parameters)
            cp=record['checkpoints'][choice]
            stem=f"{record['model']}-{record['seed']}-{choice}"
            path=Path(cfg['output'])/f'{stem}-calibration-logits.npz'
            np.savez_compressed(path,logits=logits,timestamp=split['timestamp'][keep])
            entries.append(dict(model=record['model'],seed=record['seed'],choice=choice,
                checkpoint=cp,model_weights_sha256=weight_hash,parameters=parameters,
                raw=binary_metrics(y,logits),calibrated=binary_metrics(y,calibrated,logits),
                logits_path=path.as_posix(),logits_sha256=digest(path),reused_same_weights=reused_prediction))
            print(json.dumps(dict(calibrated=stem,a=parameters['a'],b=parameters['b'],status=parameters['status'])),flush=True)
    result=dict(provenance=prov,config=cfg,complete=True,entries=entries,
                phase='calibration only; evaluation not loaded',wall_seconds=time.perf_counter()-started)
    path=Path('reports/v31/calibrators.json'); atomic_json(path,result)
    atomic_json('reports/v31/calibration-freeze.json',dict(calibrators_sha256=digest(path),
                entry_count=len(entries),provenance=prov,complete=True))
    print('All30 calibrator identities frozen; evaluation has not been loaded.',flush=True)


def evaluate(cfg,prov):
    fits=read_fits(cfg,prov)
    freeze=json.loads(Path('reports/v31/calibration-freeze.json').read_text())
    path=Path('reports/v31/calibrators.json')
    if not freeze['complete'] or freeze['provenance']!=prov or digest(path)!=freeze['calibrators_sha256']:
        raise RuntimeError('Calibrator freeze mismatch')
    calibration=json.loads(path.read_text())
    entries={(e['model'],e['seed'],e['choice']):e for e in calibration['entries']}
    expected={(m,s,c) for m in cfg['models'] for s in cfg['seeds'] for c in CHOICES}
    if (len(calibration['entries'])!=30 or set(entries)!=expected or
            calibration['provenance']!=prov or calibration['config']!=cfg or not calibration['complete']):
        raise RuntimeError('Missing/mismatched calibrators')
    for e in entries.values():
        if digest(e['logits_path'])!=e['logits_sha256']:raise RuntimeError('Calibration logit hash differs')
        if digest(e['checkpoint']['path'])!=e['checkpoint']['sha256']:raise RuntimeError('Calibrated checkpoint changed')
    split=load_split(cfg['output'],'evaluation')
    keep=split['valid']; y=split['y'][keep]; stamp=split['timestamp'][keep]
    months=pd.to_datetime(stamp,unit='s',utc=True).strftime('%Y-%m')
    rows=[]; scores={}; stored={}; started=time.perf_counter()
    for record in fits:
        reused={}
        for choice in CHOICES:
            entry=entries[(record['model'],record['seed'],choice)]
            model,weight_hash=load_model(record,choice,cfg)
            if weight_hash!=entry['model_weights_sha256']:raise RuntimeError('Weight identity differs')
            if weight_hash not in reused:reused[weight_hash]=predict_logits(model,split,cfg)[keep]
            logits=reused[weight_hash]
            calibrated,_=apply_affine(logits,entry['parameters'])
            order=np.argsort(logits,kind='stable')
            if np.any(np.diff(calibrated[order])<0):raise RuntimeError('Calibration changed ranking')
            stem=f"{record['model']}-{record['seed']}-{choice}"
            stored[stem]=logits
            for mode,z in [('raw',logits),('calibrated',calibrated)]:
                key=(record['model'],record['seed'],choice,mode); scores[key]=z
                overall=binary_metrics(y,z,logits)
                rows.append(dict(model=record['model'],seed=record['seed'],choice=choice,mode=mode,period='exposed_evaluation',**overall))
                for month in sorted(set(months)):
                    m=months==month
                    rows.append(dict(model=record['model'],seed=record['seed'],choice=choice,mode=mode,period=month,
                                     **binary_metrics(y[m],z[m],logits[m])))
            print(json.dumps(dict(evaluated=stem,selected_epoch=entry['checkpoint']['epoch'])),flush=True)
    train=load_split(cfg['output'],'train')
    frequency=float(train['y'][train['valid']].mean())
    constant=np.full(len(y),np.log(frequency/(1-frequency)))
    for period in ['exposed_evaluation',*sorted(set(months))]:
        m=np.ones(len(y),dtype=bool) if period=='exposed_evaluation' else months==period
        rows.append(dict(model='constant',seed=None,choice='train_frequency',mode='raw',period=period,
                         **binary_metrics(y[m],constant[m])))
    def losses(model,choice,mode):
        if model=='constant':return np.tile(np.logaddexp(0,constant)-y*constant,(len(cfg['seeds']),1))
        return np.stack([np.logaddexp(0,scores[(model,s,choice,mode)])-y*scores[(model,s,choice,mode)] for s in cfg['seeds']])
    specs=[]
    for choice in CHOICES:
        for mode in ['raw','calibrated']:
            for left,right in [('c','a'),('c','b'),('b','a'),('c','no_transfer'),('c','no_open')]:
                specs.append(((left,choice,mode),(right,choice,mode),'between_models'))
            for model in cfg['models']:
                specs.append(((model,choice,mode),('constant','train_frequency','raw'),'constant_control'))
    for model in cfg['models']:
        for mode in ['raw','calibrated']:
            specs.append(((model,'full_best',mode),(model,'short_best',mode),'training_extension'))
        for choice in CHOICES:
            specs.append(((model,choice,'calibrated'),(model,choice,'raw'),'calibration_effect'))
    week=(stamp+3*86400)//(7*86400); groups=np.unique(week)
    counts=np.array([(week==g).sum() for g in groups])
    draws=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(groups),size=(cfg['bootstrap_replicates'],len(groups)))
    comparisons=[]
    for left,right,family in specs:
        difference=losses(*left)-losses(*right); mean=difference.mean(0)
        sums=np.array([mean[week==g].sum() for g in groups]); boot=sums[draws].sum(1)/counts[draws].sum(1)
        comparisons.append(dict(left=list(left),right=list(right),family=family,
            primary=left==('c','full_best','calibrated') and right==('a','full_best','calibrated'),
            difference=float(mean.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),weeks=len(groups),
            per_seed={str(s):float(difference[i].mean()) for i,s in enumerate(cfg['seeds'])}))
    output=dict(provenance=prov,config=cfg,evaluation_status='exposed exploratory research, not independent confirmation',
        calibration_freeze_sha256=digest('reports/v31/calibration-freeze.json'),
        constant_train_probability=frequency,metrics=rows,comparisons=comparisons,
        evaluated_selected_checkpoints=30,evaluated_variant_identities=60,wall_seconds=time.perf_counter()-started)
    atomic_json('reports/v31/evaluation.json',output)
    pd.DataFrame([{k:v for k,v in r.items() if k!='calibration'} for r in rows]).to_csv('reports/v31/metrics.csv',index=False)
    np.savez_compressed(Path(cfg['output'])/'evaluation-logits.npz',y=y,timestamp=stamp,**stored)
    print(json.dumps(next(c for c in comparisons if c['primary']),indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--phase',choices=['calibrate','evaluate'],required=True)
    args=p.parse_args(); cfg=json.loads(Path('configs/market-v31.json').read_text())
    torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    prov=get_provenance(cfg)
    (calibrate if args.phase=='calibrate' else evaluate)(cfg,prov)


if __name__=='__main__':main()
