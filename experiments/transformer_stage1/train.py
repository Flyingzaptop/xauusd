"""Train only stage one, then stop for user review. No later-stage code."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from model import Model
from data import Data
from loss import values
from representation import decode_path

def save_json(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def checkpoint(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def common_hash(net):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in net.state_dict().items() if not k.startswith(('fc1.','fc2.','readout.','hypernet.')))).hexdigest()
def metrics(p,data,split,ids=None):
    truth=data.raw[split+'_target_log'];anchor=data.raw[split+'_anchor_log'];y=data.raw[split+'_angles']
    if ids is not None:truth=truth[ids];anchor=anchor[ids];y=y[ids]
    path=decode_path(p,anchor,float(data.raw['s15']));err=np.abs(np.exp(path)-np.exp(truth));high,op,cl,low=np.moveaxis(path,-1,0)
    valid=(high>=np.maximum(op,cl))&(low<=np.minimum(op,cl))
    return dict(angle_mse=float(np.mean((p-y)**2)),angle_mae_degrees=float(np.abs(p-y).mean()*90),path_mae_dollars=float(err.mean()),close_mae_dollars=float(err[:,:,2].mean()),last60_path_mae_dollars=float(err[:,-4:].mean()),valid_candle_fraction=float(valid.mean()),path_mae_by_block=err.mean((0,2)).tolist())

@torch.no_grad()
def evaluate(net,data,split,probe=False):
    net.eval();n=len(data.ids[split]);ids=None
    if probe:ids=np.linspace(0,n-1,min(1024,n),dtype=int);n=len(ids);index=torch.tensor(ids,device='cuda')
    total={k:0. for k in ['total','path','angle_mse']};pred=[]
    for a in range(0,n,256):
        x,y=data.batch(split,index[a:a+256]) if probe else data.batch(split,a,a+256)
        p=net(x);v=values(p,y)
        for k in total:total[k]+=float(v[k])*len(y)
        pred.append(p.cpu().numpy())
    p=np.concatenate(pred);return {**{k:v/n for k,v in total.items()},**metrics(p,data,split,ids)},p

@torch.no_grad()
def average_context(net,data):
    net.eval();n=len(data.ids['train']);total=torch.zeros(8,96,device='cuda')
    for a in range(0,n,256):
        x,_=data.batch('train',a,a+256)
        with torch.autocast('cuda',dtype=torch.bfloat16):c=net.contexts(x)
        total+=c.float().sum(0)
    return (total/n)[None]

def audit(data):
    torch.manual_seed(912);a=Model('static').cuda();torch.manual_seed(912);b=Model('hyper').cuda();assert common_hash(a)==common_hash(b)
    params={kind:sum(p.numel() for p in net.parameters()) for kind,net in [('static',a),('hyper',b)]};assert abs(params['static']-params['hyper'])/params['static']<.001
    x,y=data.batch('train',0,8);assert all(v.shape[-1]==37 and (v[:,:,:36].abs()<1).all() for v in x)
    b.eval();p=b(x);assert torch.equal(p,b(x));values(p,y)['total'].backward()
    grad=float(torch.cat([v.grad.flatten() for v in b.hypernet.parameters()]).norm());assert grad>0
    assert all(any(v.grad is not None and float(v.grad.norm())>0 for v in block.parameters()) for block in b.encoder.layers)
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):c=b.contexts(x);aa,bb=b.factors(c)
        dw=torch.bmm(aa.float(),bb.float())/2;var=float((dw.reshape(8,8,64,128)-dw.reshape(8,8,64,128).mean(0)).norm());assert var>0
        b.fixed_context=c[:1].float();effect=float((b(x)-p).abs().max());assert effect>0
        b.fixed_context=None;b.train();drop_effect=float((b(x)-b(x)).abs().max());assert drop_effect>0
    # Target is exactly a global zero minimum of the chosen deterministic loss.
    ideal=y.detach().clone().requires_grad_();loss=values(ideal,y)['total'];g=torch.autograd.grad(loss,ideal)[0];assert float(loss)==0 and float(g.norm())==0
    assert torch.isfinite(values(torch.full_like(y,.99),y)['total'])
    del a,b
    return dict(parameters=params,paired_initial_encoder=True,generator_gradient_norm=grad,generated_weight_variation=var,conditioning_changes_output=effect,dropout_changes_output=drop_effect,eval_deterministic=True,target_loss_and_gradient_zero=True,stage=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);began=time.time()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(args.data);n=len(data.ids['train']);checks=audit(data)
    run=dict(status='training',stage=1,later_stages_disabled=True,task='Describe the visible last 120 minutes as eight rolling 15-minute candles',gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks=checks,data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},settings=dict(models=['static','hyper'],seeds=[17,29],max_epochs=30,min_epochs=10,patience=5,min_relative_improvement=.002,batch=128,dropout=.25,optimizer='AdamW',weight_decay=.0001,gradient_clip=1.,lr='cosine 3e-4 to 3e-5 across cap',loss='Huber(cumulative decoded displacements)+0.1*MSE(angular vectors)'),trials={},families={})
    init={}
    for seed in [17,29]:
        for kind in ['static','hyper']:
            name=f'{kind}_{seed}';torch.manual_seed(seed);net=Model(kind).cuda();initial=common_hash(net)
            if seed in init:assert initial==init[seed]
            init[seed]=initial;opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.0001);rows=[];best=float('inf');meaningful=float('inf');stale=0
            start_train,_=evaluate(net,data,'train',True);start_val,_=evaluate(net,data,'val');run['trials'][name]=dict(epochs=0,initial_common_sha256=initial,initial_train=start_train,initial_validation=start_val);save_json(out/'run.json',run)
            for epoch in range(1,31):
                lr=3e-5+(3e-4-3e-5)*.5*(1+np.cos(np.pi*(epoch-1)/29))
                for group in opt.param_groups:group['lr']=lr
                net.train();online=0.;seen=0;started=time.time()
                for a in range(0,n,128):
                    x,y=data.batch('train',a,a+128);opt.zero_grad(set_to_none=True);v=values(net(x),y);loss=v['total'];assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();count=len(y);seen+=count;online+=float(loss.detach())*count
                assert seen==n
                tm,_=evaluate(net,data,'train',True);vm,pv=evaluate(net,data,'val')
                if vm['path_mae_dollars']<best:
                    best=vm['path_mae_dollars'];best_epoch=epoch;checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),kind=kind,epoch=epoch,stage=1))
                    np.savez_compressed(out/f'{name}_validation.npz',angles=pv)
                if best<meaningful*(1-.002):meaningful=best;stale=0
                else:stale+=1
                checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,kind=kind,stage=1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
                rows.append(dict(epoch=epoch,examples_seen=seen,lr=lr,online_loss=online/n,train_probe=tm,validation=vm,seconds=time.time()-started));save_json(out/f'{name}_curve.json',rows)
                run['trials'][name].update(epochs=epoch,best_epoch=best_epoch,best_validation_mae=best);save_json(out/'run.json',run);save_json(out/'status.json',dict(trial=name,stage=1,epoch=epoch,best_epoch=best_epoch,train_mae=tm['path_mae_dollars'],validation_mae=vm['path_mae_dollars'],seconds=time.time()-began))
                if time.time()-began>1500:raise TimeoutError('25 minute stage-one GPU budget')
                if epoch>=10 and stale>=5:break
            run['trials'][name]['stop']='plateau' if epoch<30 else 'cap';net.load_state_dict(torch.load(out/f'{name}_best.pt',weights_only=False)['model'])
            if kind=='hyper':
                net.eval();ix=torch.tensor(np.linspace(0,n-1,32,dtype=int),device='cuda');x,_=data.batch('train',ix)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):c=net.contexts(x);aa,bb=net.factors(c)
                dw=torch.bmm(aa.float(),bb.float()).reshape(32,8,64,128)/2
                run['trials'][name]['dynamic_relative_variation']=float((dw-dw.mean(0)).flatten(2).norm(dim=2).mean()/net.fc2.weight.detach().norm())
                net.fixed_context=average_context(net,data);_,p=evaluate(net,data,'val');np.savez_compressed(out/f'{name}_fixed_validation.npz',angles=p,context=net.fixed_context.cpu().numpy())
            save_json(out/'run.json',run);del net,opt
    for kind in ['static','hyper','hyper_fixed']:
        names=[f'{kind}_{s}' for s in [17,29]] if kind!='hyper_fixed' else [f'hyper_{s}_fixed' for s in [17,29]]
        p=np.mean([np.load(out/f'{name}_validation.npz')['angles'] for name in names],axis=0)
        run['families'][kind]=dict(validation=metrics(p,data,'val'),diagnostic=kind=='hyper_fixed');np.savez_compressed(out/f'{kind}_validation.npz',angles=p)
    run['selected_family']=min(['static','hyper'],key=lambda k:run['families'][k]['validation']['path_mae_dollars']);run['selection_frozen_before_test']=True;run['status']='evaluating';save_json(out/'run.json',run)
    for family in run['families']:
        kind='static' if family=='static' else 'hyper';ps=[]
        for seed in [17,29]:
            name=f'{kind}_{seed}';net=Model(kind).cuda();net.load_state_dict(torch.load(out/f'{name}_best.pt',weights_only=False)['model'])
            if family=='hyper_fixed':net.fixed_context=torch.tensor(np.load(out/f'{name}_fixed_validation.npz')['context'],device='cuda')
            _,p=evaluate(net,data,'test');ps.append(p);np.savez_compressed(out/f'{family}_{seed}_test.npz',angles=p);del net
        p=np.mean(ps,axis=0);run['families'][family]['test']=metrics(p,data,'test');np.savez_compressed(out/f'{family}_test.npz',angles=p)
    run['status']='complete';run['seconds']=time.time()-began;save_json(out/'run.json',run);save_json(out/'status.json',dict(stage=1,status='complete',seconds=run['seconds'],next_stage='Await user review; no automatic continuation'));print('STAGE 1 COMPLETE',flush=True)
if __name__=='__main__':main()
