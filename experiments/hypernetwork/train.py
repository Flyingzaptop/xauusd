"""Fixed hypernetwork comparison on the precision-first monthly task."""
import argparse,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from classification_base import Data,save_json,checkpoint
from precision_base import update,collect
from policy import fit_policy,apply_policy,metrics,fixtures
from hyper_model import Model


def trunk_hash(net):
    return hashlib.sha256(torch.cat([v.detach().flatten() for k,v in net.state_dict().items() if k.startswith(('patch.','blocks.','norm.'))]).cpu().numpy().tobytes()).hexdigest()


def probe(net,data,ix):
    net.eval();grads=[];ys=[];xs=[]
    for i in ix:
        net.zero_grad(set_to_none=True);x,y=data.batch('train',int(i),int(i)+1);F.cross_entropy(net(x),y).backward()
        grads.append(torch.cat([p.grad.detach().flatten() for p in net.parameters()]));ys.append(int(y));xs.append(x)
    g=torch.stack(grads);norm=g.norm(dim=1);cos=(g@g.T)/torch.clamp(norm[:,None]*norm[None,:],min=1e-30)
    upper=torch.triu(torch.ones_like(cos,dtype=torch.bool),diagonal=1);labels=torch.tensor(ys,device='cuda');cross=upper&(labels[:,None]!=labels[None,:])
    result=dict(conflict_fraction=float((cos[upper]<-1e-6).float().mean()),cross_class_conflict_fraction=float((cos[cross]<-1e-6).float().mean()),
                mean_cosine=float(cos[upper].mean()),cancellation_ratio=float(g.mean(0).norm()/norm.mean()),dynamic_weights=net.dynamics(torch.cat(xs)))
    net.zero_grad(set_to_none=True);return result


@torch.no_grad()
def average_context(net,data):
    total=torch.zeros(256,device='cuda');n=len(data.starts['train']);net.eval()
    for a in range(0,n,512):
        x,_=data.batch('train',a,a+512)
        with torch.autocast('cuda',dtype=torch.bfloat16):c=net.context(x)
        total+=c.float().sum(0)
    return (total/n)[None,:]


def audit(data,ix):
    torch.manual_seed(912);static=Model('static').cuda();torch.manual_seed(912);hyper=Model('hyper').cuda()
    counts={k:sum(p.numel() for p in net.parameters()) for k,net in [('static',static),('hyper',hyper)]}
    assert counts=={'static':635730,'hyper':635539}
    assert trunk_hash(static)==trunk_hash(hyper)
    x=torch.cat([data.batch('train',int(i),int(i)+1)[0] for i in ix[:8]])
    y=torch.cat([data.batch('train',int(i),int(i)+1)[1] for i in ix[:8]])
    F.cross_entropy(hyper(x),y).backward()
    grad=float(torch.cat([p.grad.flatten() for p in hyper.hypernet.parameters()]).norm());assert grad>0
    stats=hyper.dynamics(x);assert stats['context_variation_norm']>0
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c=hyper.context(x);a,b=hyper.factors(c);h=hyper.head[1](hyper.head[0](c))
        factored=torch.bmm(a.float(),torch.bmm(b.float(),h.float().unsqueeze(-1))).squeeze(-1)/2
        explicit=torch.bmm(torch.bmm(a.float(),b.float())/2,h.float().unsqueeze(-1)).squeeze(-1)
        err=float((factored-explicit).abs().max());assert torch.allclose(factored,explicit,atol=1e-5,rtol=1e-4)
        original=hyper(x);hyper.fixed_context=c[:1].float();fixed=hyper(x);effect=float((original-fixed).abs().max());assert effect>0
    del static,hyper
    return dict(parameters=counts,initial_trunks_identical=True,hypernet_gradient_norm=grad,explicit_dynamic_matmul_max_error=err,conditioning_changes_logits_max=effect,initial_dynamic_weights=stats)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out;out.mkdir(exist_ok=True,parents=True);began=time.time();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(args.data);n=len(data.starts['train']);yv=data.y['val'].cpu().numpy();yt=data.y['train'].cpu().numpy()
    times=data.raw['timestamp'][data.starts['val']-1];early=times<int(np.datetime64('2024-02-05').astype('datetime64[s]').astype(int));late=~early
    probe_ix=np.concatenate([np.flatnonzero(yt==c)[np.linspace(0,int((yt==c).sum())-1,8,dtype=int)] for c in range(3)])
    assert len(np.unique(probe_ix))==24
    checks={**fixtures(),**audit(data,probe_ix)}
    run=dict(status='training',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
             model_sha256=hashlib.sha256(Path(__file__).with_name('hyper_model.py').read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks=checks,
             parameters=checks['parameters'],settings=dict(seeds=[17,29],models=['static','hyper'],epochs=12,batch=128,group=32,lr=[2e-4]*3+[1e-4]*3+[5e-5]*3+[2e-5]*3,optimizer='Adam',loss='ordinary CE',rank=4),
             selection_split=dict(checkpoint_windows=int(early.sum()),threshold_windows=int(late.sum())),probe_indices=probe_ix.tolist(),
             counts={s:np.bincount(data.y[s].cpu(),minlength=3).tolist() for s in data.starts},trials={},families={})
    initial={}
    for seed in [17,29]:
        for kind in ['static','hyper']:
            name=f'{kind}_{seed}';torch.manual_seed(seed);net=Model(kind).cuda()
            with torch.no_grad():net.head[-1].bias.copy_(torch.tensor(np.log(data.priors),device='cuda',dtype=torch.float32))
            sha=trunk_hash(net)
            if seed in initial:assert initial[seed]==sha
            initial[seed]=sha
            init_probe=probe(net,data,probe_ix);opt=torch.optim.Adam(net.parameters(),lr=2e-4);rng=np.random.default_rng(seed+1000);rows=[];best=-1.
            for epoch,lr in enumerate(run['settings']['lr'],1):
                for group in opt.param_groups:group['lr']=lr
                online=0.;seen=0
                for a in range(0,n,128):
                    b=min(a+128,n);online+=update(net,opt,data,a,b,'adam','ce',rng)*(b-a);seen+=b-a
                assert seen==n
                pt,_,train_cost=collect(net,data,'train','ce');pv,_,_=collect(net,data,'val','ce')
                pol=fit_policy(pv[early],yv[early],times[early]//86400);sm=metrics(apply_policy(pv[early],pol),yv[early])
                rows.append(dict(epoch=epoch,lr=lr,examples_seen=seen,train_cost=train_cost,selection_policy=pol,selection_metrics=sm,train_policy_metrics=metrics(apply_policy(pt,pol),yt)))
                if pol['selection_score']>best:
                    best=pol['selection_score'];best_epoch=epoch
                    checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),epoch=epoch,kind=kind))
                    np.savez_compressed(out/f'{name}_validation.npz',p=pv,target=yv,timestamp=times,early=early)
                checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,kind=kind))
                run['trials'][name]=dict(epochs=epoch,best_epoch=best_epoch,initial_trunk_sha256=sha,initial_probe=init_probe)
                save_json(out/f'{name}_curve.json',rows);save_json(out/'run.json',run)
                save_json(out/'status.json',dict(trial=name,epoch=epoch,best_epoch=best_epoch,train_cost=train_cost,selection_precision=sm['precision'],selection_signals=sm['signals'],elapsed=time.time()-began))
                if time.time()-began>1500:raise TimeoutError('25 minute watchdog')
            net.load_state_dict(torch.load(out/f'{name}_best.pt',weights_only=False)['model']);run['trials'][name]['best_probe']=probe(net,data,probe_ix)
            if kind=='hyper':
                net.fixed_context=average_context(net,data);p,_,_=collect(net,data,'val','ce')
                np.savez_compressed(out/f'{name}_fixed_validation.npz',p=p,target=yv,timestamp=times,early=early,context=net.fixed_context.cpu().numpy())
            save_json(out/'run.json',run);del net,opt
    for family in ['static','hyper','hyper_fixed']:
        names=[f'{family}_{s}' for s in [17,29]] if family!='hyper_fixed' else [f'hyper_{s}_fixed' for s in [17,29]]
        zs=[np.load(out/f'{name}_validation.npz') for name in names];p=np.mean([z['p'] for z in zs],axis=0)
        pol=fit_policy(p[late],yv[late],times[late]//86400)
        run['families'][family]=dict(policy=pol,calibration=metrics(apply_policy(p[late],pol),yv[late]),diagnostic_only=family=='hyper_fixed')
        np.savez_compressed(out/f'{family}_calibration.npz',p=p[late],target=yv[late],timestamp=times[late])
    run['selected_family']=max(['static','hyper'],key=lambda k:run['families'][k]['policy']['selection_score'])
    run['selection_frozen_before_test']=True;run['status']='evaluating';save_json(out/'run.json',run)
    for family in run['families']:
        kind='static' if family=='static' else 'hyper';ps=[]
        for seed in [17,29]:
            net=Model(kind).cuda();net.load_state_dict(torch.load(out/f'{kind}_{seed}_best.pt',weights_only=False)['model'])
            if family=='hyper_fixed':net.fixed_context=torch.tensor(np.load(out/f'hyper_{seed}_fixed_validation.npz')['context'],device='cuda')
            p,y,_=collect(net,data,'test','ce');ps.append(p);del net
        p=np.mean(ps,axis=0);pred=apply_policy(p,run['families'][family]['policy']);run['families'][family]['test']=metrics(pred,y)
        np.savez_compressed(out/f'{family}_test.npz',p=p,target=y,prediction=pred,delta_cents=data.delta['test'],timestamp=data.raw['timestamp'][data.starts['test']-1])
    run['status']='complete';run['seconds']=time.time()-began;save_json(out/'run.json',run);save_json(out/'status.json',dict(stage='complete',seconds=run['seconds']));print('COMPLETE',flush=True)

if __name__=='__main__':main()
