"""Predeclared paired architecture x curriculum comparison, no test tuning."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import Data
from model import Model
from objective import objective,pinball

def save_json(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def checkpoint(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def trunk_hash(net):
    b=b''.join(v.detach().cpu().numpy().tobytes() for k,v in net.state_dict().items() if k.startswith(('towers.','context.')))
    return hashlib.sha256(b).hexdigest()
def templates(data):
    return {k:np.moveaxis(np.quantile(data.raw['train_'+k],[.1,.5,.9],axis=0),0,-1).astype('float32') for k in ['activity','coarse','final']}
def initialize_heads(net,data,temp):
    with torch.no_grad():
        for k,head in net.heads.items():
            head.weight.mul_(.01)
            if k=='past':raw=np.median(data.raw['train_past'],axis=0)
            else:
                q=temp[k];lower=np.maximum(q[...,1]-q[...,0],1e-4);upper=np.maximum(q[...,2]-q[...,1],1e-4)
                inv=lambda x:x+np.log(-np.expm1(-x))
                raw=np.stack([q[...,1],inv(lower),inv(upper)],-1)
            head.bias.copy_(torch.tensor(raw.flatten(),device='cuda'))

@torch.no_grad()
def evaluate(net,data,split,stage,probe=False):
    net.eval();values=[];loss=0.;aux=0.;n=len(data.ids[split]);indices=None
    if probe:indices=torch.tensor(np.linspace(0,n-1,min(1536,n),dtype=int),device='cuda');n=len(indices)
    for a in range(0,n,256):
        x,y=data.batch(split,indices[a:a+256],None) if probe else data.batch(split,a,a+256)
        p=net(x);loss+=float(objective(p,y,stage))*len(x[0])
        if stage==1:primary=F.smooth_l1_loss(p['past'],y['past'])
        else:key={2:'activity',3:'coarse',4:'final'}[stage];primary=pinball(p[key],y[key])
        aux+=float(primary)*len(x[0]);values.append(p['final'].cpu().numpy())
    return dict(objective=loss/n,primary=aux/n),np.concatenate(values)

@torch.no_grad()
def fixed_context(net,data):
    total=torch.zeros(192,device='cuda');n=len(data.ids['train']);net.eval()
    for a in range(0,n,256):
        x,_=data.batch('train',a,a+256)
        with torch.autocast('cuda',dtype=torch.bfloat16):c=net.features(x)
        total+=c.float().sum(0)
    return (total/n)[None]

def audit(data):
    torch.manual_seed(123);a=Model('static').cuda();torch.manual_seed(123);b=Model('hyper').cuda();assert trunk_hash(a)==trunk_hash(b)
    counts={k:sum(p.numel() for p in n.parameters()) for k,n in [('static',a),('hyper',b)]};assert abs(counts['static']-counts['hyper'])/counts['static']<.001
    x,y=data.batch('train',0,8);b.eval();p=b(x);assert torch.equal(p['final'],b(x)['final'])
    objective(p,y,4).backward();gradient=float(torch.cat([v.grad.flatten() for v in b.hypernet.parameters()]).norm());assert gradient>0
    for tower in b.towers:assert sum(float(v.grad.norm()) for v in tower.parameters() if v.grad is not None)>0
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):c=b.features(x);aa,bb=b.factors(c)
        dw=torch.bmm(aa.float(),bb.float())/2;variation=float((dw-dw.mean(0)).norm());assert variation>0
        b.fixed_context=c[:1].float();fixed=b(x)['final'];effect=float((p['final']-fixed).abs().max());assert effect>0
        b.fixed_context=None;b.train();effect_dropout=float((b(x)['final']-b(x)['final']).abs().max());assert effect_dropout>0
        assert (p['final'][...,0]<=p['final'][...,1]).all() and (p['final'][...,1]<=p['final'][...,2]).all()
        assert (p['final'][:,:,1,:]<=p['final'][:,:,0,:]).all() and (p['final'][:,:,0,:]<=p['final'][:,:,2,:]).all()
    del a,b
    return dict(parameters=counts,paired_trunks=True,generator_gradient_norm=gradient,generated_weight_variation=variation,conditioning_effect=effect,dropout_effect=effect_dropout,eval_deterministic=True,quantile_and_range_ordering=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    began=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(args.data);temp=templates(data);n=len(data.ids['train']);checks=audit(data)
    run=dict(status='training',gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks=checks,data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
             settings=dict(seeds=[17,29],dropout=.25,batch=128,optimizer='AdamW',weight_decay=.0001,gradient_clip=2.,stage_caps=[3,4,4,9],min_stage_epochs=[3,3,3,4],patience=3,minimum_relative_improvement=.005,quantiles=[.1,.5,.9]),trials={},families={})
    for seed in [17,29]:
        for kind in ['static','hyper']:
            budget=None;paired=None
            for mode in ['curriculum','direct']:
                name=f'{kind}_{mode}_{seed}';torch.manual_seed(seed);net=Model(kind).cuda();initialize_heads(net,data,temp)
                initial=trunk_hash(net)
                if paired is not None:assert paired==initial
                paired=initial;opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=.0001)
                schedule=list(zip([1,2,3,4],[3,4,4,9],[3,3,3,4])) if mode=='curriculum' else [(4,budget,min(6,budget))]
                rows=[];stages=[];epoch=0;best_final=None
                for stage,cap,min_epochs in schedule:
                    best=float('inf');meaningful=float('inf');stale=0;stage_path=out/f'{name}_stage{stage}.pt'
                    for local_epoch in range(1,cap+1):
                        epoch+=1;lr=2e-4 if epoch<=6 else (1e-4 if epoch<=12 else 5e-5)
                        for group in opt.param_groups:group['lr']=lr
                        net.train();total=0.;seen=0;epoch_started=time.time()
                        for a in range(0,n,128):
                            x,y=data.batch('train',a,a+128);opt.zero_grad(set_to_none=True);p=net(x);loss=objective(p,y,stage)
                            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2,error_if_nonfinite=True);opt.step()
                            count=len(x[0]);seen+=count;total+=float(loss.detach())*count
                        assert seen==n
                        val,pv=evaluate(net,data,'val',stage);train,_=evaluate(net,data,'train',stage,True)
                        if val['primary']<best:
                            best=val['primary'];best_local=local_epoch;best_global=epoch
                            checkpoint(stage_path,dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,stage=stage,kind=kind))
                            if stage==4:np.savez_compressed(out/f'{name}_validation.npz',prediction=pv,target=data.raw['val_final'])
                        if val['primary']<meaningful*(1-.005):meaningful=val['primary'];stale=0
                        else:stale+=1
                        row=dict(epoch=epoch,stage=stage,stage_epoch=local_epoch,lr=lr,examples_seen=seen,online_loss=total/n,train_probe=train,validation=val,seconds=time.time()-epoch_started)
                        rows.append(row);save_json(out/f'{name}_curve.json',rows)
                        run['trials'][name]=dict(epochs=epoch,stage=stage,initial_trunk_sha256=initial,stages=stages);save_json(out/'run.json',run)
                        save_json(out/'status.json',dict(trial=name,epoch=epoch,stage=stage,stage_epoch=local_epoch,train=train['primary'],val=val['primary'],seconds=time.time()-began))
                        if time.time()-began>3000:raise TimeoutError('50 minute worker budget')
                        if local_epoch>=min_epochs and stale>=3:break
                    ck=torch.load(stage_path,weights_only=False);net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
                    checkpoint(stage_path,dict(model=net.state_dict(),epoch=best_global,stage=stage,kind=kind))
                    stages.append(dict(stage=stage,epochs=local_epoch,best_global_epoch=best_global,best_stage_epoch=best_local,best_validation=best,stop='plateau' if local_epoch<cap else 'cap'))
                    if stage==4:best_final=best
                checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),kind=kind,mode=mode,selected_epoch=stages[-1]['best_global_epoch']))
                # Generated-weight variation and fixed-context ablation use train only.
                diagnostic=None
                if kind=='hyper':
                    net.eval();x,_=data.batch('train',torch.tensor(np.linspace(0,n-1,32,dtype=int),device='cuda'),None)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):c=net.features(x);aa,bb=net.factors(c)
                    dw=torch.bmm(aa.float(),bb.float())/2;diagnostic=dict(relative_variation=float((dw-dw.mean(0)).flatten(1).norm(dim=1).mean()/net.fc2.weight.norm()))
                    net.fixed_context=fixed_context(net,data);_,p=evaluate(net,data,'val',4)
                    np.savez_compressed(out/f'{name}_fixed_validation.npz',prediction=p,context=net.fixed_context.cpu().numpy())
                run['trials'][name]=dict(epochs=epoch,stages=stages,initial_trunk_sha256=initial,best_validation=best_final,generated_weights=diagnostic,epoch_cap=sum(c for _,c,_ in schedule));save_json(out/'run.json',run)
                if mode=='curriculum':budget=epoch
                del net,opt
    # All weights and family choices frozen before test labels are evaluated.
    yv=data.raw['val_final'];q=np.array([.1,.5,.9])
    def score(p,y):
        e=y[...,None]-p;return float(np.maximum(q*e,(q-1)*e).mean())
    for kind in ['static','hyper']:
        for mode in ['curriculum','direct']:
            family=f'{kind}_{mode}';ps=[np.load(out/f'{family}_{s}_validation.npz')['prediction'] for s in [17,29]];p=np.mean(ps,axis=0)
            run['families'][family]=dict(validation_pinball=score(p,yv),diagnostic=False)
            np.savez_compressed(out/f'{family}_validation.npz',prediction=p,target=yv)
            if kind=='hyper':
                pp=np.mean([np.load(out/f'{family}_{s}_fixed_validation.npz')['prediction'] for s in [17,29]],axis=0)
                run['families'][family+'_fixed']=dict(validation_pinball=score(pp,yv),diagnostic=True)
    run['selected_family']=min([k for k,v in run['families'].items() if not v['diagnostic']],key=lambda k:run['families'][k]['validation_pinball'])
    run['selection_frozen_before_test']=True;run['status']='evaluating';save_json(out/'run.json',run)
    for family in run['families']:
        parts=family.split('_');kind,mode=parts[:2];fixed=len(parts)==3;ps=[]
        for seed in [17,29]:
            name=f'{kind}_{mode}_{seed}';net=Model(kind).cuda();net.load_state_dict(torch.load(out/f'{name}_best.pt',weights_only=False)['model'])
            if fixed:net.fixed_context=torch.tensor(np.load(out/f'{name}_fixed_validation.npz')['context'],device='cuda')
            _,p=evaluate(net,data,'test',4);ps.append(p);np.savez_compressed(out/f'{family}_{seed}_test.npz',prediction=p);del net
        p=np.mean(ps,axis=0);np.savez_compressed(out/f'{family}_test.npz',prediction=p,target=data.raw['test_final']);run['families'][family]['test_pinball']=score(p,data.raw['test_final'])
    for split in ['val','test']:
        np.savez_compressed(out/f'baseline_{split}.npz',prediction=np.broadcast_to(temp['final'],(len(data.ids[split]),8,3,3)),target=data.raw[split+'_final'])
    run['status']='complete';run['seconds']=time.time()-began;save_json(out/'run.json',run);save_json(out/'status.json',dict(stage='complete',seconds=run['seconds']));print('COMPLETE',flush=True)
if __name__=='__main__':main()
