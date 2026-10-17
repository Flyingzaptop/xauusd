"""User-requested continuation: exhaust widths 13, 8, 5, 3, 2 without early stopping."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from full_train import Data,Generator as OriginalGenerator,atomic_json,checkpoint
from pcgrad_train import update,evaluate,forward,torch_fixture
from projection import fixtures
from capacity_model import Generator,parameter_count,width_ladder


def decide(curves,reference):
    arms={}
    for mode in ['adam','pcgrad']:
        early=[];late=[];train_down=[];epochs=[]
        for seed in [17,29]:
            rows=curves[f'{mode}_{seed}']
            er=min(rows[1:4],key=lambda r:r['validation']['normalized_mse'])
            lr=min(rows[4:],key=lambda r:r['validation']['normalized_mse'])
            early.append(er['validation']['normalized_mse']);late.append(lr['validation']['normalized_mse'])
            train_down.append(lr['train_fixed']['normalized_mse']<er['train_fixed']['normalized_mse'])
            epochs.append(lr['epoch'])
        old=np.mean([reference['trials'][f'{mode}_{s}']['best_validation'] for s in [17,29]])
        gain=1-float(np.mean(late)/np.mean(early))
        arms[mode]=dict(early=float(np.mean(early)),late=float(np.mean(late)),late_improvement_pct=100*gain,
                       late_selected_epochs=epochs,train_decreased=train_down,reference_best=float(old),
                       passed=bool(gain>=.002 and all(train_down) and np.mean(late)<=old*1.002))
    return dict(passed=all(v['passed'] for v in arms.values()),arms=arms)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True);args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);started=time.time()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reference=json.loads(args.reference.read_text());data=Data(args.data)
    assert hashlib.sha256(args.data.read_bytes()).hexdigest()==reference['dataset_sha256']
    n=len(data.splits['train'][0]);steps=math.ceil(n/128)
    def status(**kw):atomic_json(out/'status.json',dict(elapsed_seconds=time.time()-started,**kw))
    checks=fixtures();checks['group_gradient_error']=torch_fixture()
    torch.manual_seed(17);old=OriginalGenerator('tcn').cuda()
    torch.manual_seed(17);new=Generator(128).cuda()
    assert old.state_dict().keys()==new.state_dict().keys()
    assert all(torch.equal(old.state_dict()[k],new.state_dict()[k]) for k in old.state_dict())
    x,y,base,_=data.batch('train',0,8)
    with torch.no_grad():err=float((forward(old,x,base)-forward(new,x,base)).abs().max())
    assert err==0;checks['width128_output_max_error']=err
    del old,new,x,y,base
    report=dict(status='training',gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks=checks,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dataset_sha256=reference['dataset_sha256'],reference_parameters=parameter_count(128),
                ladder=[dict(width=w,parameters=parameter_count(w)) for w in width_ladder()],
                settings=dict(seeds=[17,29],epochs=12,lr=[2e-4]*3+[1e-4]*3+[5e-5]*3+[2e-5]*3,batch=128,group=32,clip=5),
                stages={},trials={},metrics={},windows=reference['windows'])
    atomic_json(out/'run.json',report)
    for width in [13, 8, 5, 3, 2]:
        stage_curves={};pair_hash={}
        for seed in [17,29]:
            for mode in ['adam','pcgrad']:
                name=f'w{width}_{mode}_{seed}'
                if time.time()-started>2600:raise TimeoutError('Capacity experiment budget exhausted')
                torch.manual_seed(seed);net=Generator(width).cuda()
                initial_hash=hashlib.sha256(torch.cat([p.detach().flatten() for p in net.parameters()]).cpu().numpy().tobytes()).hexdigest()
                if seed in pair_hash:assert initial_hash==pair_hash[seed]
                pair_hash[seed]=initial_hash
                opt=torch.optim.Adam(net.parameters(),lr=2e-4);rng=np.random.default_rng(seed+1000)
                curves=[dict(epoch=0,train_fixed=evaluate(net,data,'train'),validation=evaluate(net,data,'val'))]
                best=float('inf');best_epoch=-1
                status(stage='trial_start',trial=name,width=width,parameters=parameter_count(width))
                for epoch,lr in enumerate(report['settings']['lr'],1):
                    net.train();t0=time.time();sums={};counts={};seen=0
                    for group in opt.param_groups:group['lr']=lr
                    for step,begin in enumerate(range(0,n,128)):
                        end=min(begin+128,n)
                        stats=update(net,opt,data,begin,end,mode,rng,record_actual=step%64==0)
                        seen+=end-begin
                        for key,value in stats.items():
                            weight=end-begin if key=='loss' else 1
                            sums[key]=sums.get(key,0)+value*weight;counts[key]=counts.get(key,0)+weight
                    assert seen==n
                    train_fixed=evaluate(net,data,'train');val=evaluate(net,data,'val')
                    row=dict(epoch=epoch,lr=lr,examples_seen=seen,updates=steps,seconds=time.time()-t0,
                             train={k:v/counts[k] for k,v in sums.items()},train_fixed=train_fixed,validation=val)
                    curves.append(row)
                    if val['normalized_mse']<best:
                        best=val['normalized_mse'];best_epoch=epoch
                        checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),epoch=epoch,width=width,validation=val,train_fixed=train_fixed))
                    checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,width=width,projection_rng=rng.bit_generator.state))
                    atomic_json(out/f'{name}_curve.json',curves)
                    report['trials'][name]=dict(width=width,parameters=parameter_count(width),seed=seed,mode=mode,
                                               initial_sha256=initial_hash,completed_epochs=epoch,examples_seen=epoch*n,
                                               best_epoch=best_epoch,best_validation=best,
                                               last_train_fixed=train_fixed['normalized_mse'],last_validation=val['normalized_mse'])
                    atomic_json(out/'run.json',report)
                    status(stage='epoch_complete',trial=name,epoch=epoch,best_epoch=best_epoch,train=train_fixed['normalized_mse'],validation=val['normalized_mse'])
                stage_curves[f'{mode}_{seed}']=curves
                del net,opt
        decision=decide(stage_curves,reference)
        report['stages'][str(width)]=dict(parameters=parameter_count(width),decision=decision)
        atomic_json(out/'run.json',report)
        status(stage='width_complete',width=width,parameters=parameter_count(width),decision=decision)
        print(json.dumps(dict(width=width,parameters=parameter_count(width),decision=decision)),flush=True)
        if False:  # User requested all remaining smaller widths, regardless of criterion.
            report['stop_reason']='first width meeting delayed-improvement criterion in both arms';break
    else:
        report['stop_reason']='user-requested remaining widths completed through minimum nondegenerate width 2'
    report['status']='evaluating';report['decisions_frozen_before_test']=True
    report['selected_width']={mode:min(report['stages'],key=lambda w:np.mean([report['trials'][f'w{w}_{mode}_{s}']['best_validation'] for s in [17,29]])) for mode in ['adam','pcgrad']}
    atomic_json(out/'run.json',report)
    for name,trial in report['trials'].items():
        status(stage='test',trial=name)
        net=Generator(trial['width']).cuda()
        for version in ['best','last']:
            ck=torch.load(out/f'{name}_{version}.pt',map_location='cuda',weights_only=False);net.load_state_dict(ck['model'])
            label=name if version=='best' else name+'_last'
            report['metrics'][label]=evaluate(net,data,'test',out,label)
        del net,ck
        atomic_json(out/'run.json',report)
    report['metrics']['bridge']=evaluate(None,data,'test',out,'bridge')
    report['status']='complete';report['seconds']=time.time()-started;report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
    atomic_json(out/'run.json',report);status(stage='complete',stop_reason=report['stop_reason'])
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
