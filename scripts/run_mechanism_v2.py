"""Synthetic-only frozen mechanism diagnosis. Never accesses market data."""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crossing_memory import CrossingMemory, route_windows
from src.crossing_memory_v2 import TreeCrossingMemory, route_windows_v2
from src.synthetic_v2 import make_dataset, oracle_labels


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Classifier(nn.Module):
    def __init__(self, variant, cfg, gru_width=None):
        super().__init__()
        self.variant = variant
        if variant == 'gru':
            self.memory = nn.GRU(2, gru_width, batch_first=True)
            width = gru_width
        elif variant == 'v1':
            self.memory = CrossingMemory(2, hidden_dim=12, scales=3)
            width = self.memory.output_dim
        else:
            self.memory = TreeCrossingMemory(2, cfg['hidden_dim'], len(cfg['thresholds'])+1,
                                             transfer=variant != 'no_transfer', read_open=variant != 'no_open')
            width = self.memory.output_dim
        self.head = nn.Sequential(nn.Linear(width, cfg['head_dim']), nn.Tanh(), nn.Linear(cfg['head_dim'], 2))

    def forward(self, batch):
        x = batch['x']
        if self.variant == 'gru':
            h = self.memory(x)[0]
        elif self.variant == 'v1':
            h = self.memory(x, batch['v1_stats'], batch['v1_closures'])
        else:
            h = self.memory(x, batch['tree'])[0]
        return self.head(h[:, -1])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def matched_gru_width(cfg):
    target = parameter_count(Classifier('tree', cfg))
    width = min(range(4, 129), key=lambda w: abs(parameter_count(Classifier('gru', cfg, w))-target))
    return width


def pack(z, y, cfg):
    z = z.astype(np.float64)
    x = np.stack([np.diff(z, axis=1, prepend=z[:, :1]), (z-z[:, :1])/10.], -1).astype('float32')
    old_stats, old_closures = route_windows(100*np.exp(z*.0025), [.0025,.0075,.015])
    tree = route_windows_v2(z, cfg['thresholds'])
    return dict(x=torch.from_numpy(x), y=torch.from_numpy(y),
                v1_stats=torch.from_numpy(old_stats), v1_closures=torch.from_numpy(old_closures),
                tree={k: torch.from_numpy(v) for k,v in tree.items() if k in ['metadata','births','closures','observer','active']})


def subset(batch, ix):
    return {k: {a:v[ix] for a,v in value.items()} if isinstance(value, dict) else value[ix] for k,value in batch.items()}


@torch.no_grad()
def evaluate(model, batch):
    model.eval()
    logits = model(batch)
    y = batch['y']
    if torch.unique(y).numel() != 2:
        raise ValueError('Balanced accuracy requires both classes in this fixed diagnostic')
    prediction = logits.argmax(-1)
    balanced = torch.stack([(prediction[y == c] == c).float().mean() for c in [0,1]]).mean()
    return dict(loss=float(nn.functional.cross_entropy(logits,y)), accuracy=float((prediction==y).float().mean()),
                balanced_accuracy=float(balanced))


def datasets(task, seed, cfg):
    actual = 'hard_closed' if task in ['tiny_hard','random_hard'] else task
    n = cfg['tiny_size'] if task == 'tiny_hard' else cfg['train_size']
    z, y, hashes = make_dataset(actual,n,seed*100+1)
    np.testing.assert_array_equal(y,oracle_labels(actual,z))
    raw = {'train': (z,y)}
    info = {'train': hashes}
    exclude = set(hashes)
    if task == 'tiny_hard':
        raw.update(validation=(z.copy(),y.copy()), test=(z.copy(),y.copy()))
        info.update(validation=hashes, test=hashes)
    else:
        for offset,name in [(2,'validation'),(3,'test')]:
            zz,yy,hh = make_dataset(actual,cfg[name+'_size'],seed*100+offset,exclude_hashes=exclude)
            np.testing.assert_array_equal(yy,oracle_labels(actual,zz))
            assert not exclude.intersection(hh)
            exclude.update(hh)
            raw[name]=(zz,yy)
            info[name]=hh
    if task == 'random_hard':
        for offset,name in enumerate(raw):
            zz,yy = raw[name]
            yy = np.random.default_rng(seed*1000+offset+80).integers(0,2,len(yy),dtype=np.int64)
            raw[name]=(zz,yy)
    counts = {name: dict(n=len(y), positive=int(y.sum()), sha256=hashlib.sha256(z.tobytes()+y.tobytes()).hexdigest(),
                        unique_paths=len(set(info[name]))) for name,(z,y) in raw.items()}
    return {name:pack(z,y,cfg) for name,(z,y) in raw.items()}, counts


def fit(variant, seed, cfg, task, data, gru_width, artifact):
    torch.manual_seed(seed)
    model = Classifier(variant,cfg,gru_width)
    initial = {k:p.detach().clone() for k,p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    epochs = cfg['tiny_epochs'] if task == 'tiny_hard' else cfg['epochs']
    generator = torch.Generator().manual_seed(seed+10000)
    curve=[]
    best=float('inf')
    max_grad=0.
    started=time.perf_counter()
    best_state=None
    for epoch in range(epochs):
        model.train()
        for ix in torch.randperm(len(data['train']['y']),generator=generator).split(cfg['batch_size']):
            batch=subset(data['train'],ix)
            loss=nn.functional.cross_entropy(model(batch),batch['y'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss {task}/{variant}/{seed}/{epoch}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError('Nonfinite gradient')
                    max_grad=max(max_grad,float(p.grad.abs().max()))
            nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
            opt.step()
        train=evaluate(model,data['train'])
        val=evaluate(model,data['validation'])
        curve.append(dict(epoch=epoch+1,train=train,validation=val))
        score=train['loss'] if task == 'tiny_hard' else val['loss']
        if score < best:
            best=score
            best_epoch=epoch+1
            best_state=copy.deepcopy(model.state_dict())
        if (epoch+1)%20 == 0:
            print(json.dumps(dict(task=task,variant=variant,seed=seed,epoch=epoch+1,
                                  train_loss=train['loss'],train_ba=train['balanced_accuracy'],val_ba=val['balanced_accuracy'])),flush=True)
    final_test=evaluate(model,data['test'])
    changes={k:float((p.detach()-initial[k]).norm()) for k,p in model.named_parameters()}
    final_train=evaluate(model,data['train'])
    final_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_test=evaluate(model,data['test'])
    torch.save({'best':best_state,'final':final_state},artifact/f'{task}-{variant}-{seed}.pt')
    return dict(task=task,variant=variant,seed=seed,parameters=parameter_count(model),
                training_seconds=time.perf_counter()-started,best_epoch=best_epoch,epochs=epochs,
                optimizer_updates=epochs*((len(data['train']['y'])+cfg['batch_size']-1)//cfg['batch_size']),
                curve=curve,final_train=final_train,final_test=final_test,selected_test=selected_test,
                max_abs_gradient=max_grad,parameter_change_l2=changes,finite_gradients=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/mechanism-v2.json')
    parser.add_argument('--task',choices=['all','tiny_hard','easy_closed','medium_closed','hard_closed','hard_open','random_hard'],default='all')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    artifact=Path('artifacts/mechanism-v2')
    artifact.mkdir(parents=True,exist_ok=True)
    output=Path('reports/v2')
    output.mkdir(parents=True,exist_ok=True)
    files=['src/crossing_memory_v2.py','src/synthetic_v2.py','src/crossing_memory.py','scripts/run_mechanism_v2.py',args.config]
    provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    files_sha256={f:checksum(f) for f in files},python=sys.version,torch=torch.__version__,numpy=np.__version__)
    width=matched_gru_width(cfg)
    counts={v:parameter_count(Classifier(v,cfg,width)) for v in ['v1','tree','no_transfer','no_open','gru']}
    print(json.dumps(dict(gru_width=width,parameters=counts)),flush=True)
    tasks=list(cfg['tasks']) if args.task=='all' else [args.task]
    for task in tasks:
        runs=[]
        dataset_info=[]
        for seed in cfg['seeds']:
            data,info=datasets(task,seed,cfg)
            dataset_info.append(dict(seed=seed,splits=info))
            for variant in cfg['tasks'][task]:
                result=fit(variant,seed,cfg,task,data,width,artifact)
                runs.append(result)
                report=dict(provenance=provenance,config=cfg,task=task,gru_width=width,parameters=counts,
                            dataset=dataset_info,runs=runs,complete=len(runs)==len(cfg['seeds'])*len(cfg['tasks'][task]))
                (output/f'{task}.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(dict(finished=task,variant=variant,seed=seed,
                                      train=result['final_train'],test=result['selected_test'])),flush=True)


if __name__=='__main__':
    main()
