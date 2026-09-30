"""Stable binary metrics and causal raw-logit inference for v3.1."""
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from src.stream_batch_v3 import PackedStream
from scripts.run_market_v3 import tensors


@torch.no_grad()
def predict_logits(model, split, cfg):
    model.eval()
    result=np.full(len(split['y']),np.nan,dtype=np.float64)
    state=None
    for raw in PackedStream(split,cfg['batch_size'],cfg['chunk_length']):
        batch=tensors(raw)
        logits,state=model(batch['x'],batch['tree'],batch['reset'],state)
        real=raw['indices']>=0
        result[raw['indices'][real]]=logits.numpy()[real]
    if not np.isfinite(result).all():
        raise FloatingPointError('Missing or nonfinite stream logits')
    return result


def binary_metrics(y, logits, ranking_logits=None):
    y=np.asarray(y,dtype=np.float64)
    z=np.asarray(logits,dtype=np.float64)
    if len(y)==0 or y.shape!=z.shape or not np.isfinite(z).all():
        raise ValueError('Invalid metric inputs')
    p=expit(z)
    ranking=z if ranking_logits is None else np.asarray(ranking_logits)
    # The positive affine calibrator preserves the original ranking; evaluate
    # ranking on original logits to avoid new probability saturation/tie errors.
    auc=float(roc_auc_score(y,ranking)) if len(np.unique(y))==2 else None
    bins=np.minimum((p*10).astype(int),9)
    table=[]
    ece=0.
    for b in range(10):
        mask=bins==b
        if mask.any():
            predicted=float(p[mask].mean()); observed=float(y[mask].mean())
            table.append(dict(bin=b,count=int(mask.sum()),predicted=predicted,observed=observed))
            ece+=mask.mean()*abs(predicted-observed)
    return dict(n=len(y),log_loss=float(np.mean(np.logaddexp(0,z)-y*z)),auc=auc,
                brier=float(np.square(p-y).mean()),ece=float(ece),calibration=table)
