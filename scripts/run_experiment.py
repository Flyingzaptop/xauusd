"""Frozen bounded M1 comparison. Run from repository root."""
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
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.crossing_memory import CrossingMemory, route_windows


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(cfg):
    for key in ['horizon', 'window', 'stride', 'max_train', 'max_validation', 'max_test']:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(key+' must be a positive integer')
    frame = pd.read_parquet(cfg['data'])
    ts = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit('ns').astype('int64').to_numpy() // 10**9
    ohlc = frame[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError('Nonunique or unordered source: do not silently repair')
    good = np.isfinite(ohlc).all(1) & (ohlc > 0).all(1)
    good &= (ohlc[:, 1] >= ohlc.max(1)) & (ohlc[:, 2] <= ohlc.min(1))
    bad = np.r_[0, np.cumsum((np.diff(ts) != 60) | ~good[1:] | ~good[:-1])]
    logp = np.log(ohlc)
    bounds = [int(pd.Timestamp(cfg[k], tz='UTC').timestamp()) for k in
              ['train_start', 'validation_start', 'test_start', 'end']]
    train_rows = (ts >= bounds[0]) & (ts < bounds[1])
    moves = np.abs(np.diff(logp[:, 3]))
    train_moves = moves[train_rows[1:] & train_rows[:-1] & (np.diff(ts) == 60) & good[1:] & good[:-1]]
    base = float(np.median(train_moves))
    if base <= 0:
        raise ValueError('Degenerate training threshold')
    thresholds = (base * np.array(cfg['threshold_multipliers'])).tolist()
    result = {}
    w, horizon = cfg['window'], cfg['horizon']
    for k, name in enumerate(['train', 'validation', 'test']):
        ends = np.arange(w-1, len(ts)-horizon, cfg['stride'])
        starts, targets = ends-w+1, ends+horizon
        valid = (ts[starts] >= bounds[k]) & (ts[targets] < bounds[k+1])
        valid &= bad[targets] == bad[starts]
        ends = ends[valid]
        candidate_count = len(ends)
        cap = cfg['max_'+name]
        if len(ends) > cap:
            ends = ends[np.linspace(0, len(ends)-1, cap).astype(int)]
        if not len(ends):
            raise ValueError('Empty split: '+name)
        ix = ends[:, None] - np.arange(w-1, -1, -1)[None, :]
        prices = ohlc[ix, 3]
        p = logp[ix]
        x = np.stack([np.concatenate([np.zeros((len(ix), 1)), np.diff(p[:, :, 3], axis=1)], axis=1),
                      p[:, :, 3]-p[:, :, 0], p[:, :, 1]-p[:, :, 3], p[:, :, 2]-p[:, :, 3]], axis=-1)
        stats, closures = route_windows(prices, thresholds)
        r = logp[ends+horizon, 3]-logp[ends, 3]
        result[name] = dict(x=x.astype('float32'), stats=stats, closures=closures,
                            y=(r > 0).astype('float32'), r=r.astype('float32'),
                            timestamp=ts[ends], candidate_count=candidate_count)
    mean = result['train']['x'].mean((0, 1))
    std = result['train']['x'].std((0, 1)).clip(1e-8)
    for split in result.values():
        split['x'] = np.clip((split['x']-mean)/std, -10, 10)
    scale = float(result['train']['r'].std().clip(1e-8))
    metadata = dict(thresholds=thresholds, feature_mean=mean.tolist(), feature_std=std.tolist(),
                    return_scale=scale, splits={n: dict(samples=len(d['y']), candidates=d['candidate_count'],
                    first=str(pd.to_datetime(d['timestamp'][0], unit='s', utc=True)),
                    last=str(pd.to_datetime(d['timestamp'][-1], unit='s', utc=True))) for n,d in result.items()})
    return result, metadata


class Model(nn.Module):
    def __init__(self, kind, cfg):
        super().__init__()
        self.kind = kind
        size = 4 if kind == 'A' else 22
        self.memory = None
        if kind.startswith('C'):
            self.memory = CrossingMemory(4, hidden_dim=cfg['memory_hidden'], scales=3,
                                         transfer=kind != 'C_no_transfer', read_open=kind != 'C_no_open')
            size += self.memory.output_dim
        self.gru = nn.GRU(size, cfg['hidden'], batch_first=True)
        self.head = nn.Linear(cfg['hidden'], 2)

    def forward(self, x, stats, closures):
        inputs = [x]
        if self.kind != 'A':
            inputs.append(stats.flatten(2))
        if self.memory is not None:
            inputs.append(self.memory(x, stats, closures))
        h, _ = self.gru(torch.cat(inputs, dim=-1))
        return self.head(h[:, -1])


def tensors(split):
    return {k: torch.from_numpy(np.asarray(split[k])) for k in ['x', 'stats', 'closures', 'y', 'r']}


@torch.no_grad()
def predict(model, split, batch):
    model.eval()
    return np.concatenate([model(split['x'][i:i+batch], split['stats'][i:i+batch],
                                 split['closures'][i:i+batch]).numpy()
                           for i in range(0, len(split['y']), batch)])


def metrics(y, r, probability, predicted_return):
    probability = np.clip(probability, 1e-7, 1-1e-7)
    bins = np.minimum((probability*10).astype(int), 9)
    calibration = []
    ece = 0.
    for b in range(10):
        keep = bins == b
        if keep.any():
            item = dict(bin=b, count=int(keep.sum()), predicted=float(probability[keep].mean()), observed=float(y[keep].mean()))
            ece += keep.mean()*abs(item['predicted']-item['observed'])
            calibration.append(item)
    return dict(n=len(y), log_loss=float(log_loss(y, probability, labels=[0, 1])),
                roc_auc=float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
                brier=float(brier_score_loss(y, probability)), ece=float(ece),
                mae=float(np.abs(r-predicted_return).mean()), calibration=calibration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.json')
    parser.add_argument('--phase', choices=['fit', 'evaluate'], required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    artifact = Path('artifacts/crossing-memory-v1')
    artifact.mkdir(parents=True, exist_ok=True)
    data, prep = prepare(cfg)
    provenance = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      config_sha256=digest(args.config), data_sha256=digest(cfg['data']), config=cfg,
                      preprocessing=prep, torch_version=torch.__version__)
    train, val = tensors(data['train']), tensors(data['validation'])
    scale = prep['return_scale']
    if args.phase == 'fit':
        records = []
        for kind in cfg['models']:
            for seed in cfg['seeds']:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = Model(kind, cfg)
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
                best = float('inf')
                history = []
                started = time.perf_counter()
                for epoch in range(cfg['epochs']):
                    model.train()
                    order = rng.permutation(len(train['y']))
                    losses = []
                    for start in range(0, len(order), cfg['batch_size']):
                        ix = order[start:start+cfg['batch_size']]
                        out = model(train['x'][ix], train['stats'][ix], train['closures'][ix])
                        loss = nn.functional.binary_cross_entropy_with_logits(out[:, 0], train['y'][ix])
                        loss += cfg['regression_weight']*nn.functional.mse_loss(out[:, 1], train['r'][ix]/scale)
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    out = predict(model, val, cfg['batch_size'])
                    probability = 1/(1+np.exp(-out[:, 0]))
                    score = float(log_loss(data['validation']['y'], probability, labels=[0,1]))
                    history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_log_loss=score))
                    if score < best:
                        best = score
                        best_epoch = epoch+1
                        torch.save(model.state_dict(), artifact/f'{kind}-{seed}.pt')
                    print(json.dumps(dict(model=kind, seed=seed, **history[-1])), flush=True)
                records.append(dict(model=kind, seed=seed, parameters=sum(p.numel() for p in model.parameters()),
                                    training_seconds=time.perf_counter()-started, best_epoch=best_epoch, history=history))
        (artifact/'fit.json').write_text(json.dumps(dict(provenance=provenance, runs=records), indent=2))
        print('All fits complete. Test predictions have not been evaluated.', flush=True)
        return
    fit = json.loads((artifact/'fit.json').read_text())
    for key in ['code_commit', 'config_sha256', 'data_sha256']:
        if fit['provenance'][key] != provenance[key]:
            raise RuntimeError('Fit/evaluation provenance differs: '+key)
    test = tensors(data['test'])
    rows, predictions = [], {}
    for record in fit['runs']:
        kind, seed = record['model'], record['seed']
        model = Model(kind, cfg)
        model.load_state_dict(torch.load(artifact/f'{kind}-{seed}.pt', weights_only=True))
        for split_name, split_tensor in [('validation', val), ('test', test)]:
            split = data[split_name]
            out = predict(model, split_tensor, cfg['batch_size'])
            p = 1/(1+np.exp(-out[:, 0]))
            rhat = out[:, 1]*scale
            rows.append(dict(model=kind, seed=seed, period=split_name, **metrics(split['y'], split['r'], p, rhat)))
            if split_name == 'test':
                predictions[(kind, seed)] = p
                months = pd.to_datetime(split['timestamp'], unit='s', utc=True).strftime('%Y-%m')
                for month in sorted(set(months)):
                    keep = months == month
                    rows.append(dict(model=kind, seed=seed, period=month, **metrics(split['y'][keep], split['r'][keep], p[keep], rhat[keep])))
    for name in ['validation', 'test']:
        split = data[name]
        p = np.full(len(split['y']), data['train']['y'].mean())
        rhat = np.full(len(split['y']), data['train']['r'].mean())
        rows.append(dict(model='constant', seed=None, period=name, **metrics(split['y'], split['r'], p, rhat)))
        if name == 'test':
            months = pd.to_datetime(split['timestamp'], unit='s', utc=True).strftime('%Y-%m')
            for month in sorted(set(months)):
                keep = months == month
                rows.append(dict(model='constant', seed=None, period=month, **metrics(split['y'][keep], split['r'][keep], p[keep], rhat[keep])))
    split = data['test']
    y = split['y']
    losses = {kind: np.mean([-y*np.log(np.clip(predictions[(kind,s)],1e-7,1-1e-7))-(1-y)*np.log(np.clip(1-predictions[(kind,s)],1e-7,1-1e-7)) for s in cfg['seeds']], axis=0) for kind in cfg['models']}
    weeks = (split['timestamp'] + 3*86400)//(7*86400)
    unique = np.unique(weeks)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    boot = rng.integers(0, len(unique), size=(cfg['bootstrap_replicates'], len(unique)))
    counts = np.array([(weeks == week).sum() for week in unique])
    comparisons = []
    for other in ['A', 'B', 'C_no_transfer', 'C_no_open']:
        difference = losses['C']-losses[other]
        sums = np.array([difference[weeks == week].sum() for week in unique])
        estimates = sums[boot].sum(1)/counts[boot].sum(1)
        comparisons.append(dict(comparison='C - '+other, difference=float(difference.mean()),
                                ci95=np.quantile(estimates,[.025,.975]).tolist(), weeks=len(unique)))
    result = dict(provenance=provenance, runs=fit['runs'], metrics=rows, paired_week_bootstrap=comparisons)
    Path('reports/market_metrics.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame([{k:v for k,v in row.items() if k != 'calibration'} for row in rows]).to_csv('reports/market_metrics.csv', index=False)
    np.savez_compressed(artifact/'test_predictions.npz', y=y, timestamp=split['timestamp'],
                        **{f'{k}-{s}':v for (k,s),v in predictions.items()})
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
