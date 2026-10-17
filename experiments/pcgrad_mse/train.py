"""Deterministic TCN MSE, matched grouped-PCGrad ablation; see protocol.md."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from full_train import Data, Generator, LATENT, atomic_json, checkpoint
from projection import coefficients, diagnostics, fixtures


def parameters(net):
    return list(net.parameters())


def vector(net, gradients=False):
    return torch.cat([(p.grad if gradients else p.detach()).reshape(-1) for p in net.parameters()])


def forward(net, x, base):
    return net(x, base, torch.zeros(len(x), 1, LATENT, device=x.device))[:, 0]


def group_gradients(net, data, begin, end, group=32):
    gradients, losses, sizes = [], [], []
    for a in range(begin, end, group):
        net.zero_grad(set_to_none=True)
        x,y,base,_=data.batch('train',a,min(a+group,end))
        loss=(forward(net,x,base)-y).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        loss.backward()
        gradients.append(vector(net,True));losses.append(loss.detach());sizes.append(len(x))
    G=torch.stack(gradients)
    gram=(G@G.T).double().cpu().numpy()
    weights=np.asarray(sizes,dtype=np.float64)/sum(sizes)
    loss=float(torch.stack(losses) @ torch.tensor(weights,device=G.device,dtype=G.dtype))
    return G,gram,weights,loss


def update(net,opt,data,begin,end,mode,rng,record_actual=False):
    G,gram,w,loss=group_gradients(net,data,begin,end)
    orders=[rng.permutation(len(w)) for _ in w]
    c=coefficients(gram,orders)
    stats=diagnostics(gram,c,w)
    combined=(torch.as_tensor(w@c if mode=='pcgrad' else w,device=G.device,dtype=G.dtype)@G)
    offset=0
    for p in net.parameters():
        p.grad=combined[offset:offset+p.numel()].view_as(p)
        offset+=p.numel()
    norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),5.))
    if not math.isfinite(norm):raise RuntimeError('nonfinite gradient')
    before=vector(net) if record_actual else None
    opt.step()
    if before is not None:
        change=(G@(vector(net)-before)).detach().cpu().numpy()
        stats['adam_harm_fraction']=float(np.mean(change>1e-10))
    stats.update(loss=loss,grad_norm=norm,clipped=float(norm>5))
    return stats


@torch.no_grad()
def evaluate(net,data,split,out=None,name=None):
    if net is not None:net.eval()
    n=len(data.splits[split][0]); arrays={}
    predictions=[];truths=[];bases=[]
    for a in range(0,n,512):
        x,y,base,_=data.batch(split,a,min(a+512,n))
        p=forward(net,x,base) if net is not None else base
        scale=data.splits[split][1][a:a+len(x)]*1e4
        err=(p-y)*scale[:,None,None]
        predicted_steps=torch.diff(p[:,:,2],dim=1,prepend=torch.zeros_like(p[:,:1,2]))
        actual_steps=torch.diff(y[:,:,2],dim=1,prepend=torch.zeros_like(y[:,:1,2]))
        valid=(p[:,:,0]>=torch.maximum(p[:,:,1],p[:,:,2])-1e-6)&(p[:,:,3]<=torch.minimum(p[:,:,1],p[:,:,2])+1e-6)
        values=dict(normalized_mse=(p-y).square().mean((1,2)),hcl_mse=err[:,:,[0,2,3]].square().mean((1,2)),
                    close_mse=err[:,:,2].square().mean(1),hcl_mae=err[:,:,[0,2,3]].abs().mean((1,2)),
                    predicted_tv=(predicted_steps.abs()*scale[:,None]).sum(1),actual_tv=(actual_steps.abs()*scale[:,None]).sum(1),
                    invalid=(~valid).float().mean(1))
        for key,v in values.items():arrays.setdefault(key,[]).append(v.cpu().numpy())
        if out is not None:
            predictions.append(p.cpu().numpy());truths.append(y.cpu().numpy());bases.append(base.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    result={k:float(v.mean()) for k,v in arrays.items()}
    result['tv_ratio']=result['predicted_tv']/result['actual_tv']
    if out is not None:
        starts=data.splits[split][0].cpu().numpy()
        np.savez_compressed(out/f'{name}_cases.npz',**arrays,prediction=np.concatenate(predictions),target=np.concatenate(truths),base=np.concatenate(bases),
                            scale=data.splits[split][1].cpu().numpy(),timestamp=data.raw['timestamp'][starts],start=starts)
    return result


def probe(net,data,out,name):
    net.eval();n=len(data.splits['train'][0]);grads=[];losses=[]
    indices=np.linspace(0,n-1,32,dtype=int)
    for i in indices:
        net.zero_grad(set_to_none=True)
        x,y,base,_=data.batch('train',int(i),int(i)+1)
        loss=(forward(net,x,base)-y).square().mean();loss.backward()
        grads.append(vector(net,True));losses.append(float(loss.detach()))
    G=torch.stack(grads);gram=(G@G.T).double().cpu().numpy()
    rng=np.random.default_rng(551)
    c=coefficients(gram,[rng.permutation(len(G)) for _ in G])
    stats=diagnostics(gram,c,np.ones(len(G))/len(G))
    np.savez_compressed(out/f'{name}_probe.npz',gram=gram,indices=indices,loss=losses,projection_coefficients=c)
    net.zero_grad(set_to_none=True)
    return stats


def torch_fixture():
    # Independent weighted group gradients equal full-batch ordinary MSE gradients.
    torch.manual_seed(2)
    net=torch.nn.Linear(3,2).double()
    x=torch.randn(7,3,dtype=torch.float64);y=torch.randn(7,2,dtype=torch.float64)
    ((net(x)-y)**2).mean().backward();whole=vector(net,True)
    groups=[]
    for a,b in [(0,3),(3,7)]:
        net.zero_grad(set_to_none=True);((net(x[a:b])-y[a:b])**2).mean().backward();groups.append(vector(net,True))
    err=float(abs(whole-(groups[0]*3+groups[1]*4)/7).max())
    assert err<1e-12
    return err


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    started=time.time()
    def status(**kw):atomic_json(out/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    checks=fixtures();checks['group_gradient_equivalence_error']=torch_fixture()
    data=Data(args.data);n=len(data.splits['train'][0]);steps=math.ceil(n/128)
    report=dict(status='benchmark',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),checks=checks,
                settings=dict(seeds=[17,29],epochs=12,lr=[2e-4]*3+[1e-4]*3+[5e-5]*3+[2e-5]*3,batch=128,group=32,optimizer='Adam',clip=5,latent='fixed zero',architecture='TCN',
                              train_order='chronological all eligible windows',pcgrad='four contiguous example groups; not per-example',
                              projection='against original group gradients; independent shuffled orders; sample-count-weighted mean'),
                windows={s:len(v[0]) for s,v in data.splits.items()},trials={},metrics={})
    if (out/'run.json').exists():
        previous=json.loads((out/'run.json').read_text())
        assert previous['dataset_sha256']==report['dataset_sha256']
        assert previous['settings']==report['settings']
        report['trials']=previous.get('trials',{})
        report['metrics']=previous.get('metrics',{})
        report['benchmark_seconds_per_update']=previous['benchmark_seconds_per_update']
        report['estimated_fit_seconds']=previous['estimated_fit_seconds']
        report['prior_source_sha256']=previous['source_sha256']
        report['previous_fit_seconds']=sum(r['seconds'] for f in out.glob('*_curve.json') for r in json.loads(f.read_text())[1:])
    else:
        benchmark={}
        for mode in ['adam','pcgrad']:
            torch.manual_seed(17);net=Generator('tcn').cuda();opt=torch.optim.Adam(net.parameters(),lr=2e-4);rng=np.random.default_rng(891)
            for k in range(12):
                if k==3:torch.cuda.synchronize();t0=time.time()
                update(net,opt,data,0,128,mode,rng,record_actual=k%4==0)
            torch.cuda.synchronize();benchmark[mode]=(time.time()-t0)/9
            del net,opt
        report['benchmark_seconds_per_update']=benchmark
        estimate=sum(benchmark.values())*steps*12*2
        report['estimated_fit_seconds']=estimate
        atomic_json(out/'run.json',report)
        print(json.dumps(dict(benchmark=benchmark,estimated_fit_seconds=estimate)),flush=True)
        if estimate>1200:
            report['status']='budget_review_required';atomic_json(out/'run.json',report)
            raise RuntimeError('Benchmark exceeds predeclared budget; revise before fitting')
    report['status']='training'
    for seed in [17,29]:
        for mode in ['adam','pcgrad']:
            name=f'{mode}_{seed}';torch.manual_seed(seed);net=Generator('tcn').cuda()
            opt=torch.optim.Adam(net.parameters(),lr=2e-4);rng=np.random.default_rng(seed+1000)
            completed=0
            if (out/f'{name}_last.pt').exists():
                ck=torch.load(out/f'{name}_last.pt',map_location='cuda',weights_only=False)
                net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);completed=ck['epoch']
                curves=json.loads((out/f'{name}_curve.json').read_text())
                previous=report['trials'][name]
                best=previous['best_validation'];best_epoch=previous['best_epoch'];initial_probe=previous['initial_probe']
                if 'projection_rng' in ck:rng.bit_generator.state=ck['projection_rng']
                else:
                    for _ in range(completed):
                        for begin in range(0,n,128):
                            count=math.ceil(min(128,n-begin)/32)
                            for _ in range(count):rng.permutation(count)
                del ck
            else:
                status(stage='initial_probe',trial=name)
                initial_probe=probe(net,data,out,name+'_initial')
                best=float('inf');best_epoch=-1
                curves=[dict(epoch=0,validation=evaluate(net,data,'val'))]
            for epoch,lr in enumerate(report['settings']['lr'],1):
                if epoch<=completed:continue
                net.train();epoch_started=time.time();last_heartbeat=0;aggregate={};count={};seen=0
                for group in opt.param_groups:group['lr']=lr
                for step,begin in enumerate(range(0,n,128)):
                    end=min(begin+128,n)
                    stats=update(net,opt,data,begin,end,mode,rng,record_actual=step%64==0)
                    seen+=end-begin
                    for key,value in stats.items():
                        weight=end-begin if key=='loss' else 1
                        aggregate[key]=aggregate.get(key,0.)+value*weight;count[key]=count.get(key,0)+weight
                    if time.time()-last_heartbeat>30:
                        status(stage='training',trial=name,epoch=epoch,seen=seen,total=n,train_loss=aggregate['loss']/count['loss'])
                        last_heartbeat=time.time()
                    if time.time()-started>3500:raise TimeoutError('Training watchdog')
                assert seen==n
                stats={k:aggregate[k]/count[k] for k in aggregate}
                val=evaluate(net,data,'val')
                row=dict(epoch=epoch,lr=lr,examples_seen=seen,updates=steps,seconds=time.time()-epoch_started,train=stats,validation=val)
                curves.append(row)
                if val['normalized_mse']<best:
                    best=val['normalized_mse'];best_epoch=epoch
                    checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),epoch=epoch,validation=val))
                checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,projection_rng=rng.bit_generator.state))
                atomic_json(out/f'{name}_curve.json',curves)
                report['trials'][name]=dict(best_epoch=best_epoch,best_validation=best,completed_epochs=epoch,examples_seen=epoch*n,initial_probe=initial_probe)
                atomic_json(out/'run.json',report);status(stage='epoch_complete',trial=name,**row)
                print(json.dumps(dict(trial=name,**row)),flush=True)
            report['metrics'][name+'_last']=evaluate(net,data,'test',out,name+'_last')
            ck=torch.load(out/f'{name}_best.pt',map_location='cuda',weights_only=False);net.load_state_dict(ck['model'])
            report['trials'][name]['best_probe']=probe(net,data,out,name+'_best')
            report['metrics'][name]=evaluate(net,data,'test',out,name)
            atomic_json(out/'run.json',report)
            del net,opt,ck
    report['metrics']['bridge']=evaluate(None,data,'test',out,'bridge')
    report['status']='complete';report['seconds']=time.time()-started+report.get('previous_fit_seconds',0)
    report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    atomic_json(out/'run.json',report);status(stage='complete')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
