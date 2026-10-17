"""Full chronological epochs, joint energy score and a matched MSE control."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

H, CTX, K, LATENT = 8, 512, 8, 32
DEVICE = 'cuda'


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def checkpoint(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def energy(samples, target):
    z, y = samples.flatten(2), target.flatten(1)
    scale = math.sqrt(z.shape[-1])
    attraction = torch.linalg.vector_norm(z - y[:, None], dim=-1).mean(1) / scale
    if z.shape[1] == 1:
        return attraction
    pair = torch.cdist(z, z, compute_mode='donot_use_mm_for_euclid_dist')
    return attraction - .5 * pair.sum((1, 2)) / (z.shape[1] * (z.shape[1] - 1) * scale)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(128, 128, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(128, 128, 3, dilation=dilation)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)

    def forward(self, x):
        z = self.conv1(F.pad(x, (2 * self.dilation, 0)))
        z = F.silu(self.norm1(z.transpose(1, 2))).transpose(1, 2)
        z = self.conv2(F.pad(z, (2 * self.dilation, 0)))
        z = self.norm2(z.transpose(1, 2)).transpose(1, 2)
        return F.silu(x + z)


class TemporalEncoder(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.patch = nn.Linear(8 * 10, 128)
        self.side = nn.Parameter(torch.zeros(2, 128))
        if architecture == 'tcn':
            self.blocks = nn.Sequential(*[CausalBlock(d) for d in [1, 2, 4, 8, 16]])
        else:
            self.position = nn.Parameter(torch.randn(1, 128, 128) * .01)
            layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., activation='gelu',
                                               batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        v = x.reshape(len(x), 2, CTX, 10)
        # Both sides run far -> near gap. Each eight-candle patch retains all
        # individual OHLC/time coordinates; there is no candle averaging.
        v = torch.stack([v[:, 0], v[:, 1].flip(1)], 1).reshape(len(x), 2, 64, 80)
        v = self.patch(v) + self.side[None, :, None]
        if self.architecture == 'tcn':
            z = self.blocks(v.reshape(len(x) * 2, 64, 128).transpose(1, 2))[:, :, -1]
            context = z.reshape(len(x), 256)
        else:
            z = self.blocks(v.reshape(len(x), 128, 128) + self.position)
            context = torch.cat([z[:, 63], z[:, 127]], -1)
        return self.norm(context)


class Generator(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.encoder = TemporalEncoder(architecture)
        self.decoder = nn.Sequential(nn.Linear(256 + LATENT, 256), nn.SiLU(),
                                     nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, H * 4))
        nn.init.normal_(self.decoder[-1].weight, std=.005)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, base, noise):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = self.encoder(x)[:, None].expand(-1, noise.shape[1], -1)
            raw = self.decoder(torch.cat([context, noise], -1)).float().reshape(len(x), noise.shape[1], H, 4)
        b = base[:, None]
        op = b[..., 1] + raw[..., 0] * math.sqrt(H)
        close = b[..., 2] + raw[..., 1] * math.sqrt(H)
        upper = (b[..., 0] - torch.maximum(b[..., 1], b[..., 2])).clamp_min(0).sqrt()
        lower = (torch.minimum(b[..., 1], b[..., 2]) - b[..., 3]).clamp_min(0).sqrt()
        high = torch.maximum(op, close) + (upper + raw[..., 2]).square()
        low = torch.minimum(op, close) - (lower + raw[..., 3]).square()
        return torch.stack([high, op, close, low], -1)


class Data:
    def __init__(self, path):
        self.raw = np.load(path)
        self.prices = torch.as_tensor(self.raw['log_prices'], device=DEVICE)
        self.times = torch.as_tensor(self.raw['timestamp'], device=DEVICE)
        self.offsets = torch.as_tensor(np.r_[np.arange(-CTX, 0), np.arange(H, H + CTX)], device=DEVICE)
        self.hidden = torch.arange(H, device=DEVICE)
        self.splits = {s: (torch.as_tensor(self.raw[s + '_start'], device=DEVICE),
                           torch.as_tensor(self.raw[s + '_scale'], device=DEVICE)) for s in ['train', 'val', 'test']}

    def batch(self, split, begin, end):
        starts, scales = self.splits[split]
        return self.at(starts[begin:end], scales[begin:end])

    def at(self, starts, scale):
        center = self.prices[starts - 1, 2]
        visible = ((self.prices[starts[:, None] + self.offsets] - center[:, None, None]) / scale[:, None, None]).float()
        diff = torch.zeros_like(visible)
        diff[:, 1:CTX] = visible[:, 1:CTX] - visible[:, :CTX - 1]
        diff[:, CTX + 1:] = visible[:, CTX + 1:] - visible[:, CTX:-1]
        times = self.times[starts[:, None] + self.offsets]
        time_gap = torch.zeros_like(times, dtype=torch.float32)
        time_gap[:, 1:CTX] = torch.log1p((times[:, 1:CTX] - times[:, :CTX - 1]).float() / 60 - 1) / 10
        time_gap[:, CTX + 1:] = torch.log1p((times[:, CTX + 1:] - times[:, CTX:-1]).float() / 60 - 1) / 10
        time_offset = (2 / math.pi) * (((times - self.times[starts - 1, None]).float() / (60 * CTX)).atan())
        x = torch.cat([(2 / math.pi) * visible.atan(), (2 / math.pi) * diff.atan(),
                       time_gap[..., None], time_offset[..., None]], -1).flatten(1)
        y = ((self.prices[starts[:, None] + self.hidden] - center[:, None, None]) / scale[:, None, None]).float()
        left, right = visible[:, CTX - 1], visible[:, CTX]
        t = torch.arange(1, H + 1, device=DEVICE)[None, :, None]
        linear = left[:, None] + (right - left)[:, None] * t / (H + 1)
        close = right[:, None, 1] * t[:, :, 0] / H
        op = torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], 1)
        base = torch.stack([torch.maximum(linear[:, :, 0], torch.maximum(op, close)), op, close,
                            torch.minimum(linear[:, :, 3], torch.minimum(op, close))], -1)
        return x, y, base, linear


@torch.no_grad()
def validation(net, data, batch):
    net.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(9001)
    total = torch.zeros(3, device=DEVICE)
    n = len(data.splits['val'][0])
    for a in range(0, n, batch):
        x, y, base, _ = data.batch('val', a, a + batch)
        p = net(x, base, torch.randn(len(x), 32, LATENT, device=DEVICE, generator=rng))
        total += torch.stack([energy(p, y).sum(), (p - y[:, None]).square().mean((1, 2, 3)).sum(),
                              (p.mean(1) - y).square().mean((1, 2)).sum()])
    values = (total / n).cpu().tolist()
    return dict(validation_energy=values[0], validation_sample_mse=values[1], validation_mean_mse=values[2])


@torch.no_grad()
def evaluate(net, name, data, out, batch, residual=None):
    if net is not None:
        net.eval()
    rng = torch.Generator(device=DEVICE).manual_seed(9901)
    n = len(data.splits['test'][0])
    records, predictions = {}, {}
    for a in range(0, n, batch):
        x, y, base, linear = data.batch('test', a, a + batch)
        if net is not None:
            p = net(x, base, torch.randn(len(x), 128, LATENT, device=DEVICE, generator=rng))
        elif name == 'right_open_bridge':
            p = base[:, None]
        elif name == 'linear_bridge':
            p = linear[:, None]
        else:
            ix = torch.randint(len(residual), (len(x), 128), device=DEVICE, generator=rng)
            p = base[:, None] + residual[ix]
            p[..., 0] = torch.maximum(p[..., 0], torch.maximum(p[..., 1], p[..., 2]))
            p[..., 3] = torch.minimum(p[..., 3], torch.minimum(p[..., 1], p[..., 2]))
        mean = p.mean(1)
        bps = data.splits['test'][1][a:a + len(x)] * 1e4
        err = (mean - y) * bps[:, None, None]
        quant = torch.quantile(p, torch.tensor([.1, .9], device=DEVICE), dim=1)
        cover = ((y >= quant[0]) & (y <= quant[1])).float()
        inc = lambda z: torch.diff(z, dim=-1, prepend=torch.zeros_like(z[..., :1]))
        actual_steps, mean_steps, sample_steps = inc(y[:, :, 2]), inc(mean[:, :, 2]), inc(p[:, :, :, 2])
        nz = actual_steps.abs() > 1e-8
        valid = (p[..., 0] >= torch.maximum(p[..., 1], p[..., 2]) - 1e-6) & (p[..., 3] <= torch.minimum(p[..., 1], p[..., 2]) + 1e-6)
        values = dict(energy=energy(p, y), mse=err[:, :, [0, 2, 3]].square().mean((1, 2)),
                      mae=err[:, :, [0, 2, 3]].abs().mean((1, 2)), close_mse=err[:, :, 2].square().mean(1),
                      coverage=cover[:, :, 2].mean(1), coverage_all=cover.mean((1, 2)),
                      width=((quant[1, :, :, 2] - quant[0, :, :, 2]) * bps[:, None]).mean(1),
                      spread=(p[:, :, :, 2].std(1, unbiased=False) * bps[:, None]).mean(1),
                      actual_tv=(actual_steps.abs() * bps[:, None]).sum(1),
                      mean_tv=(mean_steps.abs() * bps[:, None]).sum(1),
                      sample_tv=(sample_steps.abs().mean(1) * bps[:, None]).sum(1),
                      direction_correct=((mean_steps.sign() == actual_steps.sign()) & nz).sum(1),
                      direction_count=nz.sum(1), invalid=(~valid).float().mean((1, 2)))
        for k, v in values.items():
            records.setdefault(k, []).append(v.cpu().numpy())
        for k, v in dict(mean=mean, quantiles=quant.permute(1, 0, 2, 3), first5=p[:, :5]).items():
            predictions.setdefault(k, []).append(v.cpu().numpy())
    records = {k: np.concatenate(v) for k, v in records.items()}
    np.savez_compressed(out / (name + '_per_case.npz'), **records)
    np.savez_compressed(out / (name + '_predictions.npz'), **{k: np.concatenate(v) for k, v in predictions.items()})
    metrics = {k: float(v.mean()) for k, v in records.items() if k not in ['actual_tv', 'mean_tv', 'sample_tv', 'direction_correct', 'direction_count']}
    metrics.update(mean_tv_ratio=float(records['mean_tv'].sum() / records['actual_tv'].sum()),
                   sample_tv_ratio=float(records['sample_tv'].sum() / records['actual_tv'].sum()),
                   direction=float(records['direction_correct'].sum() / records['direction_count'].sum()), windows=n)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--benchmark-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min-gain', type=float, default=.005)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    assert torch.cuda.is_available()
    torch.manual_seed(17)
    data = Data(args.data)
    n = len(data.splits['train'][0])
    initial_states = {}
    parameters = {}
    for architecture in ['tcn', 'transformer']:
        torch.manual_seed(17)
        initial = Generator(architecture).to(DEVICE)
        initial_states[architecture] = copy.deepcopy(initial.state_dict())
        parameters[architecture] = sum(p.numel() for p in initial.parameters())
    report = dict(status='running', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  parameters=parameters,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  settings=dict(epochs=args.epochs, batch=args.batch, samples=K, seed=17, lr_start=.0002,
                                lr_end=.00002, clip=5, gap=H, context=CTX, latent=LATENT,
                                precision='bf16 matmuls / fp32 loss, fp64 log differences', ordering='chronological, stride=1, no shuffle'),
                  windows={s: len(v[0]) for s, v in data.splits.items()}, trials={}, metrics={})
    def status(**kw):
        atomic_json(out / 'status.json', dict(elapsed_seconds=time.time() - started, **kw))
    if args.resume:
        old = json.loads((out / 'run.json').read_text())
        assert old['dataset_sha256'] == report['dataset_sha256']
        assert old['settings']['epochs'] == args.epochs and old['settings']['batch'] == args.batch
        report = old
        report['previous_fit_seconds'] = json.loads((out / 'status.json').read_text()).get('elapsed_seconds', 0.)
        report['prior_source_sha256'] = report['source_sha256']
        report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['settings']['early_stopping'] = dict(patience=args.patience, min_relative_gain=args.min_gain,
                                                      explicit_user_request=True)
        for stale in ['worker_result.json']:
            (out / stale).unlink(missing_ok=True)
    atomic_json(out / 'run.json', report)
    fixture = torch.tensor([[[[1.]], [[3.]]]], device=DEVICE, dtype=torch.float64, requires_grad=True)
    f = energy(fixture, torch.zeros(1, 1, 1, device=DEVICE, dtype=torch.float64)).sum()
    assert abs(f.item() - 1) < 1e-12
    assert torch.allclose(torch.autograd.grad(f, fixture)[0].flatten(), torch.tensor([1., 0.], device=DEVICE, dtype=torch.float64))
    # Hardware benchmark on train only; restore weights before either actual fit.
    report['benchmark'] = {}
    for architecture in ['tcn', 'transformer']:
        initial = Generator(architecture).to(DEVICE)
        initial.load_state_dict(initial_states[architecture])
        opt = torch.optim.Adam(initial.parameters(), lr=.0002)
        for i in range(13):
            if i == 3:
                torch.cuda.synchronize(); bench_start = time.time()
            x, y, base, _ = data.batch('train', 0, args.batch)
            opt.zero_grad(set_to_none=True)
            p = initial(x, base, torch.randn(len(x), K, LATENT, device=DEVICE))
            assert torch.isfinite(p).all()
            loss = energy(p, y).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(initial.parameters(), 5); opt.step()
        torch.cuda.synchronize()
        report['benchmark'][architecture] = dict(seconds_per_update=(time.time() - bench_start) / 10,
                                  estimated_epoch_seconds=(time.time() - bench_start) / 10 * math.ceil(n / args.batch))
        print(json.dumps(dict(architecture=architecture, **report['benchmark'][architecture])), flush=True)
    atomic_json(out / 'run.json', report)
    del initial, opt, x, y, base, p
    if args.benchmark_only:
        return
    for kind in ['tcn_energy', 'tcn_mse', 'transformer_energy', 'transformer_mse']:
        if kind in args.skip:
            assert (out / (kind + '_best.pt')).exists() and (out / (kind + '_last.pt')).exists()
            report['trials'][kind]['stopped'] = 'user requested skip after validation plateau'
            atomic_json(out / 'run.json', report)
            continue
        architecture, objective_kind = kind.split('_')
        net = Generator(architecture).to(DEVICE); net.load_state_dict(initial_states[architecture])
        opt = torch.optim.Adam(net.parameters(), lr=.0002)
        rng = torch.Generator(device=DEVICE).manual_seed(1702)
        curve, best, best_epoch = [], float('inf'), -1
        significant_best, stale_epochs = float('inf'), 0
        for epoch in range(args.epochs + 1):
            epoch_start = time.time()
            seen, total_loss, norms, clipped = 0, 0., 0., 0
            lr = .00002 + (.0002 - .00002) * .5 * (1 + math.cos(math.pi * max(0, epoch - 1) / max(1, args.epochs - 1)))
            for group in opt.param_groups:
                group['lr'] = lr
            if epoch:
                net.train()
                heartbeat = time.time()
                for begin in range(0, n, args.batch):
                    x, y, base, _ = data.batch('train', begin, begin + args.batch)
                    opt.zero_grad(set_to_none=True)
                    p = net(x, base, torch.randn(len(x), K, LATENT, device=DEVICE, generator=rng))
                    objective = energy(p, y).mean() if objective_kind == 'energy' else (p - y[:, None]).square().mean()
                    assert torch.isfinite(objective), (kind, epoch, begin)
                    objective.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5))
                    assert math.isfinite(norm)
                    opt.step()
                    total_loss += float(objective.detach()) * len(x)
                    seen += len(x); norms += norm; clipped += norm > 5
                    if time.time() - heartbeat > 30:
                        status(stage='training', model=kind, epoch=epoch, seen=seen, total=n)
                        heartbeat = time.time()
                assert seen == n, 'An epoch must visit every eligible training window exactly once'
            status(stage='validation', model=kind, epoch=epoch)
            row = dict(epoch=epoch, examples_seen=seen, seconds=time.time() - epoch_start,
                       lr=lr, train_loss=total_loss / seen if seen else None,
                       grad_norm=norms / math.ceil(n / args.batch) if seen else None,
                       clipping_fraction=clipped / math.ceil(n / args.batch) if seen else None,
                       **validation(net, data, min(args.batch, 256)))
            row['seconds'] = time.time() - epoch_start
            curve.append(row)
            score = row['validation_energy'] if objective_kind == 'energy' else row['validation_sample_mse']
            if score < best:
                best, best_epoch = score, epoch
                checkpoint(out / (kind + '_best.pt'), dict(model=net.state_dict(), epoch=epoch, score=score, settings=report['settings']))
            if score < significant_best * (1 - args.min_gain):
                significant_best, stale_epochs = score, 0
            elif epoch:
                stale_epochs += 1
            checkpoint(out / (kind + '_last.pt'), dict(model=net.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
                       rng=rng.get_state(), settings=report['settings']))
            atomic_json(out / (kind + '_curve.json'), curve)
            report['trials'][kind] = dict(best_epoch=best_epoch, best_validation=best, completed_epochs=epoch,
                                          examples_seen=sum(r['examples_seen'] for r in curve))
            atomic_json(out / 'run.json', report)
            status(stage='epoch_complete', model=kind, **row)
            print(json.dumps(dict(model=kind, **row)), flush=True)
            if epoch >= 4 and stale_epochs >= args.patience:
                report['trials'][kind]['stopped'] = f'validation plateau: <{args.min_gain:.1%} gain for {args.patience} epochs'
                atomic_json(out / 'run.json', report)
                break
        del net, opt
    # Validation-frozen models; test first accessed for scoring only now.
    # Empirical scenario bank: whole training-only residual trajectories.
    starts, scales = data.splits['train']
    rid = torch.as_tensor(np.linspace(0, n - 1, min(n, 131072), dtype=np.int64), device=DEVICE)
    bank = []
    for a in range(0, len(rid), 1024):
        _, y, base, _ = data.at(starts[rid[a:a + 1024]], scales[rid[a:a + 1024]])
        bank.append(y - base)
    residual = torch.cat(bank)
    for kind in ['tcn_energy', 'tcn_mse', 'transformer_energy', 'transformer_mse',
                 'tcn_energy_last', 'transformer_energy_last', 'right_open_bridge', 'linear_bridge', 'empirical_residual']:
        status(stage='test', model=kind)
        net = None
        if kind.startswith(('tcn_', 'transformer_')):
            net = Generator(kind.split('_')[0]).to(DEVICE)
            path = out / (kind + '.pt' if kind.endswith('_last') else kind + '_best.pt')
            net.load_state_dict(torch.load(path, weights_only=False)['model'])
        report['metrics'][kind] = evaluate(net, kind, data, out, 128, residual)
        atomic_json(out / 'run.json', report)
        print(json.dumps(dict(test_model=kind, metrics=report['metrics'][kind])), flush=True)
        del net
    starts, scales = data.splits['test']
    indices = np.linspace(0, len(starts) - 1, 12, dtype=np.int64)
    ii = torch.as_tensor(indices, device=DEVICE)
    _, truth, _, _ = data.at(starts[ii], scales[ii])
    vis = ((data.prices[starts[ii, None] + data.offsets] - data.prices[starts[ii] - 1, 2][:, None, None]) / scales[ii, None, None]).float()
    np.savez_compressed(out / 'examples.npz', indices=indices, truth=truth.cpu().numpy(), visible=vis.cpu().numpy(),
                        center=data.prices[starts[ii] - 1, 2].cpu().numpy(), scale=scales[ii].cpu().numpy())
    report.update(status='complete', seconds=time.time() - started + report.get('previous_fit_seconds', 0.), peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  loss_gradient_fixture_passed=True)
    atomic_json(out / 'run.json', report)
    status(stage='complete')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
