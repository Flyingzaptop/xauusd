"""Fine-tune pretrained weights on stage TWO ONLY, then stop for review."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import Model
from data import Data
from metrics import Q,describe,baseline_predictions,pinball
from representation import decode_path

def save_json(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def checkpoint(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def legacy_hash(state):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in state.items() if not k.startswith('activity_heads.'))).hexdigest()
def objective(p,y):
    q=torch.tensor(Q,device=p.device,dtype=p.dtype);e=y.unsqueeze(-1)-p
    return torch.maximum(q*e,(q-1)*e).mean()
def init_heads(net,data,seed):
    quant=np.moveaxis(np.quantile(data.raw['train_y'],Q,axis=0),0,-1);g=torch.Generator(device='cuda').manual_seed(seed+9000)
    with torch.no_grad():
        for h,head in enumerate(net.activity_heads):
            head.weight.copy_(torch.randn(head.weight.shape,device='cuda',generator=g)*.001)
            gaps=np.concatenate([quant[h,:,:1],np.diff(quant[h],axis=-1)],axis=-1);gaps=np.maximum(gaps,1e-5)
            inv=gaps+np.log(-np.expm1(-gaps));head.bias.copy_(torch.tensor(inv.flatten(),device='cuda'))
def load_pretrained(kind,seed,data,folder):
    torch.manual_seed(seed);net=Model(kind).cuda();path=folder/f'{kind}_{seed}_best.pt';old=torch.load(path,weights_only=False)
    missing,unexpected=net.load_state_dict(old['model'],strict=False)
    assert set(missing)=={'activity_heads.0.weight','activity_heads.0.bias','activity_heads.1.weight','activity_heads.1.bias'} and not unexpected
    sha=legacy_hash(net.state_dict());assert sha==legacy_hash(old['model']);init_heads(net,data,seed)
    return net,dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stage1_epoch=old['epoch'],all_legacy_weights_exact=True,legacy_state_sha256=sha)
@torch.no_grad()
def collect(net,data,split,probe=False):
    net.eval();n=len(data.ids[split]);ids=None
    if probe:ids=np.linspace(0,n-1,min(1024,n),dtype=int);index=torch.tensor(ids,device='cuda');n=len(ids)
    ps=[];total=0.
    for a in range(0,n,256):
        x,y=data.batch(split,index[a:a+256]) if probe else data.batch(split,a,a+256);p=net(x);total+=float(objective(p,y))*len(y);ps.append(p.cpu().numpy())
    return total/n,np.concatenate(ps)
@torch.no_grad()
def old_skill(net,data,split,save_path=None):
    net.eval();ps=[];n=len(data.ids[split])
    for a in range(0,n,256):x,_=data.batch(split,a,a+256);ps.append(net.describe_past(x).cpu().numpy())
    p=np.concatenate(ps)
    if save_path is not None:np.savez_compressed(save_path,angles=p)
    path=decode_path(p,data.raw[split+'_anchor_log'],float(data.raw['s15']));err=np.abs(np.exp(path)-np.exp(data.raw[split+'_target_log']))
    return dict(path_mae_dollars=float(err.mean()),close_mae_dollars=float(err[:,:,2].mean()),note='Same frozen stage1 readout applied to changed encoder/head features; diagnostic, not stage1 retraining.')
@torch.no_grad()
def average_context(net,data):
    net.eval();total=torch.zeros(8,96,device='cuda');n=len(data.ids['train'])
    for a in range(0,n,256):
        x,_=data.batch('train',a,a+256)
        with torch.autocast('cuda',dtype=torch.bfloat16):c=net.contexts(x)
        total+=c.float().sum(0)
    return (total/n)[None]
def audit(data,folder):
    net,transfer=load_pretrained('hyper',17,data,folder);x,y=data.batch('train',0,8);net.eval();p=net(x);assert torch.equal(p,net(x)) and (p>0).all() and (p[...,1]>=p[...,0]).all() and (p[...,2]>=p[...,1]).all()
    objective(p,y).backward();encoder_gradient=sum(float(v.grad.norm()) for v in net.encoder.parameters() if v.grad is not None);generator_gradient=sum(float(v.grad.norm()) for v in net.hypernet.parameters() if v.grad is not None);assert encoder_gradient>0 and generator_gradient>0
    assert all(v.grad is None for v in net.readout.parameters());del net
    return dict(pretrained_loading=transfer,positive_ordered_quantiles=True,deterministic_eval=True,encoder_gradient=encoder_gradient,hypernetwork_gradient=generator_gradient,old_readout_frozen=True,stage=2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--pretrained',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True);began=time.time()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;data=Data(a.data);n=len(data.ids['train']);checks=audit(data,a.pretrained)
    run=dict(status='training',stage=2,no_stage1_extension=True,later_stages_disabled=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks=checks,data_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},settings=dict(seeds=[17,29],models=['static','hyper'],max_epochs=30,min_epochs=10,patience=5,min_relative_improvement=.002,batch=128,dropout=.25,head_only_epochs=3,backbone_lr=3e-5,head_lr=3e-4,optimizer='AdamW',weight_decay=.0001,clip=1.,objective='Mean normalized pinball q10/q50/q90 over two future horizons and two activity quantities'),trials={},families={},baselines={})
    for seed in [17,29]:
        for kind in ['static','hyper']:
            name=f'{kind}_{seed}';net,transfer=load_pretrained(kind,seed,data,a.pretrained);start_skill=old_skill(net,data,'val',out/f'{name}_past_before.npz');initial_cost,_=collect(net,data,'val')
            back=[p for k,p in net.named_parameters() if not k.startswith(('activity_heads.','readout.'))];heads=list(net.activity_heads.parameters());opt=torch.optim.AdamW([dict(params=back,lr=3e-5),dict(params=heads,lr=3e-4)],weight_decay=.0001)
            rows=[];best=float('inf');meaningful=float('inf');stale=0
            run['trials'][name]=dict(transfer=transfer,initial_validation_pinball=initial_cost,stage1_before=start_skill,epochs=0);save_json(out/'run.json',run)
            for epoch in range(1,31):
                joint=epoch>3
                for p in back:p.requires_grad_(joint)
                factor=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/29));opt.param_groups[0]['lr']=3e-5*factor;opt.param_groups[1]['lr']=3e-4*factor
                net.train();seen=0;online=0.;started=time.time()
                for i in range(0,n,128):
                    x,y=data.batch('train',i,i+128);opt.zero_grad(set_to_none=True);loss=objective(net(x),y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step();seen+=len(y);online+=float(loss.detach())*len(y)
                assert seen==n
                if not joint:assert legacy_hash(net.state_dict())==transfer['legacy_state_sha256']
                train_cost,_=collect(net,data,'train',True);val_cost,pv=collect(net,data,'val');vm=describe(pv,data.raw,'val')
                if val_cost<best:
                    best=val_cost;best_epoch=epoch;checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),kind=kind,stage=2,epoch=epoch));np.savez_compressed(out/f'{name}_validation.npz',prediction=pv)
                if best<meaningful*(1-.002):meaningful=best;stale=0
                else:stale+=1
                checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),kind=kind,stage=2,epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
                rows.append(dict(epoch=epoch,examples_seen=seen,phase='joint' if joint else 'head_only',online_loss=online/n,train_probe_pinball=train_cost,validation_pinball=val_cost,validation_metrics=vm,seconds=time.time()-started));save_json(out/f'{name}_curve.json',rows)
                run['trials'][name].update(epochs=epoch,best_epoch=best_epoch,best_validation_pinball=best);save_json(out/'run.json',run);save_json(out/'status.json',dict(trial=name,stage=2,epoch=epoch,best_epoch=best_epoch,train_pinball=train_cost,validation_pinball=val_cost,seconds=time.time()-began))
                if time.time()-began>1500:raise TimeoutError('25-minute stage-two worker cap')
                if epoch>=10 and stale>=5:break
            net.load_state_dict(torch.load(out/f'{name}_best.pt',weights_only=False)['model']);run['trials'][name]['stage1_after']=old_skill(net,data,'val',out/f'{name}_past_after.npz');run['trials'][name]['stop']='plateau' if epoch<30 else 'cap'
            if kind=='hyper':
                net.fixed_context=average_context(net,data);_,p=collect(net,data,'val');np.savez_compressed(out/f'{name}_fixed_validation.npz',prediction=p,context=net.fixed_context.cpu().numpy())
            save_json(out/'run.json',run);del net,opt
    for family in ['static','hyper','hyper_fixed']:
        names=[f'{family}_{s}' for s in [17,29]] if family!='hyper_fixed' else [f'hyper_{s}_fixed' for s in [17,29]];p=np.mean([np.load(out/f'{name}_validation.npz')['prediction'] for name in names],axis=0)
        run['families'][family]=dict(validation=describe(p,data.raw,'val'),diagnostic=family=='hyper_fixed');np.savez_compressed(out/f'{family}_validation.npz',prediction=p)
    for name,p in baseline_predictions(data.raw,'val').items():run['baselines'][name]=dict(validation=describe(p,data.raw,'val'))
    run['selected_family']=min(['static','hyper'],key=lambda k:run['families'][k]['validation']['pinball_normalized']);run['selected_baseline']=min(run['baselines'],key=lambda k:run['baselines'][k]['validation']['pinball_normalized']);run['selection_frozen_before_test']=True;run['status']='evaluating';save_json(out/'run.json',run)
    for family in run['families']:
        kind='static' if family=='static' else 'hyper';ps=[]
        for seed in [17,29]:
            net=Model(kind).cuda();net.load_state_dict(torch.load(out/f'{kind}_{seed}_best.pt',weights_only=False)['model'])
            if family=='hyper_fixed':net.fixed_context=torch.tensor(np.load(out/f'hyper_{seed}_fixed_validation.npz')['context'],device='cuda')
            _,p=collect(net,data,'test');ps.append(p);np.savez_compressed(out/f'{family}_{seed}_test.npz',prediction=p);del net
        p=np.mean(ps,axis=0);np.savez_compressed(out/f'{family}_test.npz',prediction=p);run['families'][family]['test']=describe(p,data.raw,'test')
    for name,p in baseline_predictions(data.raw,'test').items():run['baselines'][name]['test']=describe(p,data.raw,'test');np.savez_compressed(out/f'{name}_test.npz',prediction=p)
    run['status']='complete';run['seconds']=time.time()-began;save_json(out/'run.json',run);save_json(out/'status.json',dict(stage=2,status='complete',seconds=run['seconds'],next_stage='Await user review; no stage3'));print('STAGE 2 COMPLETE',flush=True)
if __name__=='__main__':main()
