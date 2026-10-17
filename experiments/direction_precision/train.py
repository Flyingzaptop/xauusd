"""Precision-first training and chronologically separated validation selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from classification_base import Data,Classifier,save_json,checkpoint
from projection import coefficients,fixtures as projection_fixtures
from policy import fit_policy,apply_policy,metrics,fixtures


def loss_values(logits,y,objective):
    ce=F.cross_entropy(logits,y,reduction='none')
    if objective=='ce':return ce
    penalty=torch.zeros_like(ce)
    for c in [0,2]:
        other=[k for k in range(3) if k!=c]
        # -log(1-p_c), stable even when p_c is nearly one.
        fp_cost=F.softplus(logits[:,c]-torch.logsumexp(logits[:,other],dim=1))
        penalty+=fp_cost*(y!=c)
    return ce+9*penalty


def audit():
    z=torch.tensor([[2.,0.,0.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([1])
    g0=torch.autograd.grad(loss_values(z,y,'ce').sum(),z)[0]
    g1=torch.autograd.grad(loss_values(z,y,'fp9').sum(),z)[0]
    assert g1[0,0]>g0[0,0]>0
    assert torch.isfinite(loss_values(torch.tensor([[1000.,-1000.,-1000.]]),y,'fp9')).all()
    # Arithmetic-mean weighting of groups equals the whole batch loss gradient.
    z=torch.tensor([[1.,0.,-1.],[0.,1.,0.],[.2,.3,.4]],dtype=torch.float64,requires_grad=True);y=torch.tensor([0,1,2])
    full=torch.autograd.grad(loss_values(z,y,'fp9').mean(),z,retain_graph=True)[0]
    grouped=torch.autograd.grad((loss_values(z[:2],y[:2],'fp9').mean()*2+loss_values(z[2:],y[2:],'fp9').mean())/3,z)[0]
    err=float((full-grouped).abs().max());assert err<1e-12
    return dict(wrong_short_gradient_ce=float(g0[0,0]),wrong_short_gradient_fp9=float(g1[0,0]),group_gradient_error=err)


def update(net,opt,data,a,b,mode,objective,rng):
    net.train();gradients=[];sizes=[];total=0.
    for k in range(a,b,32):
        opt.zero_grad(set_to_none=True);x,y=data.batch('train',k,min(k+32,b));loss=loss_values(net(x),y,objective).mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();gradients.append(torch.cat([p.grad.flatten() for p in net.parameters()]));sizes.append(len(y));total+=float(loss.detach())*len(y)
    g=torch.stack(gradients);w=np.array(sizes)/sum(sizes);orders=[rng.permutation(len(w)) for _ in w]
    if mode=='pcgrad':w=w@coefficients((g@g.T).double().cpu().numpy(),orders)
    combined=torch.tensor(w,device='cuda',dtype=g.dtype)@g;offset=0
    for p in net.parameters():p.grad=combined[offset:offset+p.numel()].view_as(p);offset+=p.numel()
    torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
    return total/(b-a)


@torch.no_grad()
def collect(net,data,split,objective):
    net.eval();probs=[];cost=0.;n=len(data.starts[split])
    for a in range(0,n,512):
        x,y=data.batch(split,a,a+512);z=net(x)
        probs.append(z.softmax(1).cpu().numpy());cost+=float(loss_values(z,y,objective).sum())
    return np.concatenate(probs),data.y[split].cpu().numpy(),cost/n


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out;out.mkdir(exist_ok=True,parents=True);began=time.time();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(args.data);n=len(data.starts['train']);val_t=data.raw['timestamp'][data.starts['val']-1]
    cutoff=int(np.datetime64('2024-02-05').astype('datetime64[s]').astype(int));early=val_t<cutoff;late=~early
    assert early.sum()==265 and late.sum()==428
    run=dict(status='training',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
             gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks={**fixtures(),**audit(),**projection_fixtures()},
             task='Close[t+8]-Close[t], strict +/- $2, past512 only',settings=dict(seeds=[17,29],objectives=['ce','fp9'],optimizers=['adam','pcgrad'],
             epochs=12,lr=[2e-4]*3+[1e-4]*3+[5e-5]*3+[2e-5]*3,batch=128,group=32,fp_penalty=9,min_validation_signals_per_direction=5,min_validation_days=2),
             selection_split=dict(checkpoint_windows=int(early.sum()),threshold_windows=int(late.sum()),cutoff='2024-02-05'),
             counts={s:np.bincount(data.y[s].cpu(),minlength=3).tolist() for s in data.starts},trials={},families={})
    initial={}
    for seed in [17,29]:
        for objective in ['ce','fp9']:
            for mode in ['adam','pcgrad']:
                name=f'{objective}_{mode}_{seed}';torch.manual_seed(seed);net=Classifier().cuda()
                with torch.no_grad():net.head[-1].bias.copy_(torch.tensor(np.log(data.priors),device='cuda',dtype=torch.float32))
                sha=hashlib.sha256(torch.cat([p.flatten() for p in net.parameters()]).detach().cpu().numpy().tobytes()).hexdigest()
                if seed in initial:assert initial[seed]==sha
                initial[seed]=sha;run['parameters']=sum(p.numel() for p in net.parameters())
                opt=torch.optim.Adam(net.parameters(),lr=2e-4);rng=np.random.default_rng(seed+1000);rows=[];best=-1.
                for epoch,lr in enumerate(run['settings']['lr'],1):
                    for group in opt.param_groups:group['lr']=lr
                    online=0.;seen=0
                    for a in range(0,n,128):
                        b=min(a+128,n);online+=update(net,opt,data,a,b,mode,objective,rng)*(b-a);seen+=b-a
                    assert seen==n
                    pt,yt,train_cost=collect(net,data,'train',objective);pv,yv,val_cost=collect(net,data,'val',objective)
                    policy=fit_policy(pv[early],yv[early],val_t[early]//86400)
                    sm=metrics(apply_policy(pv[early],policy),yv[early]);tm=metrics(apply_policy(pt,policy),yt)
                    row=dict(epoch=epoch,lr=lr,examples_seen=seen,train_cost=train_cost,online_cost=online/n,selection_policy=policy,selection_metrics=sm,train_policy_metrics=tm)
                    rows.append(row)
                    if policy['selection_score']>best:
                        best=policy['selection_score'];best_epoch=epoch
                        checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),epoch=epoch))
                        np.savez_compressed(out/f'{name}_validation.npz',p=pv,target=yv,timestamp=val_t,early=early)
                    checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,projection_rng=rng.bit_generator.state))
                    run['trials'][name]=dict(epochs=epoch,best_epoch=best_epoch,best_selection_score=best,initial_sha256=sha)
                    save_json(out/f'{name}_curve.json',rows);save_json(out/'run.json',run)
                    save_json(out/'status.json',dict(trial=name,epoch=epoch,best_epoch=best_epoch,train_cost=train_cost,selection_precision=sm['precision'],selection_signals=sm['signals'],elapsed=time.time()-began))
                    if time.time()-began>1500:raise TimeoutError('25 minute fit watchdog')
                del net,opt
    # Epochs are frozen before threshold-fit labels are used.
    for objective in ['ce','fp9']:
        for mode in ['adam','pcgrad']:
            family=f'{objective}_{mode}';zs=[np.load(out/f'{family}_{s}_validation.npz') for s in [17,29]]
            p=np.mean([z['p'] for z in zs],axis=0);y=zs[0]['target']
            policy=fit_policy(p[late],y[late],val_t[late]//86400)
            run['families'][family]=dict(policy=policy,calibration=metrics(apply_policy(p[late],policy),y[late]))
            np.savez_compressed(out/f'{family}_calibration.npz',p=p[late],target=y[late],timestamp=val_t[late])
    run['selected_family']=max(run['families'],key=lambda k:run['families'][k]['policy']['selection_score'])
    run['selection_frozen_before_test']=True;run['status']='evaluating';save_json(out/'run.json',run)
    for family in run['families']:
        objective,mode=family.split('_');ps=[]
        for seed in [17,29]:
            net=Classifier().cuda();net.load_state_dict(torch.load(out/f'{family}_{seed}_best.pt',weights_only=False)['model'])
            p,y,_=collect(net,data,'test',objective);ps.append(p);del net
        p=np.mean(ps,axis=0);policy=run['families'][family]['policy'];pred=apply_policy(p,policy)
        run['families'][family]['test']=metrics(pred,y)
        np.savez_compressed(out/f'{family}_test.npz',p=p,target=y,prediction=pred,delta_cents=data.delta['test'],timestamp=data.raw['timestamp'][data.starts['test']-1])
    run['status']='complete';run['seconds']=time.time()-began;save_json(out/'run.json',run);save_json(out/'status.json',dict(stage='complete',seconds=run['seconds']));print('COMPLETE',flush=True)

if __name__=='__main__':main()
