"""Past-only TCN classification: eight future candles, strict +/- $2 terminal return."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from capacity_model import Block
from projection import coefficients,fixtures
CTX,H=512,8
CLASSES=['SHORT','NONE','LONG']


def save_json(p,value):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)


def checkpoint(p,value):
    temp=p.with_suffix('.tmp');torch.save(value,temp);temp.replace(p)


class Data:
    def __init__(self,path):
        z=np.load(path);self.raw=z
        cents=np.rint(np.exp(z['log_prices'])*100).astype(np.int64)
        assert np.max(np.abs(np.exp(z['log_prices'])*100-cents))<1e-6
        self.starts={s:z[s+'_start'] for s in ['train','val','test']}
        changes=np.diff(z['log_prices'][:,2],prepend=z['log_prices'][0,2])
        cs=np.r_[0,np.cumsum(changes)];css=np.r_[0,np.cumsum(changes**2)]
        rawscale={}
        for s,ix in self.starts.items():
            # 511 close changes inside the 512 visible candles; no right-side prices.
            a,b=ix-CTX+1,ix
            mu=(cs[b]-cs[a])/(CTX-1)
            rawscale[s]=np.sqrt(np.maximum((css[b]-css[a])/(CTX-1)-mu**2,0))
            assert np.all(z['timestamp'][ix+H-1]-z['timestamp'][ix-1]==H*60)
        self.floor=max(float(np.quantile(rawscale['train'],.05)),1e-8)
        self.y={};self.delta={};self.scales={};self.ix={}
        for s,ix in self.starts.items():
            delta=cents[ix+H-1,2]-cents[ix-1,2]
            y=np.where(delta>200,2,np.where(delta < -200,0,1))
            self.y[s]=torch.tensor(y,device='cuda',dtype=torch.long)
            self.delta[s]=delta
            self.scales[s]=torch.tensor(np.maximum(rawscale[s],self.floor),device='cuda',dtype=torch.float32)
            self.ix[s]=torch.tensor(ix,device='cuda')
        counts=np.bincount(self.y['train'].cpu().numpy(),minlength=3)
        self.priors=counts/counts.sum()
        self.weights=torch.tensor(1/(3*self.priors),device='cuda',dtype=torch.float32)
        self.prices=torch.tensor(z['log_prices'],device='cuda')
        self.times=torch.tensor(z['timestamp'],device='cuda')
        self.offset=torch.arange(-CTX,0,device='cuda')
    def batch(self,s,a,b):
        ix=self.ix[s][a:b];scale=self.scales[s][a:b]
        visible=(self.prices[ix[:,None]+self.offset]-self.prices[ix-1,2,None,None])/scale[:,None,None]
        v=visible.float();diff=torch.zeros_like(v);diff[:,1:]=v[:,1:]-v[:,:-1]
        ts=self.times[ix[:,None]+self.offset];gap=torch.zeros_like(ts,dtype=torch.float32)
        gap[:,1:]=torch.log((ts[:,1:]-ts[:,:-1]).float()/60)/10
        elapsed=(2/math.pi)*(((ts-self.times[ix-1,None]).float()/(60*CTX)).atan())
        x=torch.cat([(2/math.pi)*v.atan(),(2/math.pi)*diff.atan(),gap[:,:,None],elapsed[:,:,None]],-1)
        return x,self.y[s][a:b]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch=nn.Linear(80,128)
        self.blocks=nn.Sequential(*[Block(128,d) for d in [1,2,4,8,16]])
        self.norm=nn.LayerNorm(256)
        self.head=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,3))
    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=self.blocks(self.patch(x.reshape(len(x),64,80)).transpose(1,2))
            return self.head(self.norm(torch.cat([z[:,:,-1],z.mean(-1)],-1))).float()


def metrics(logits,y):
    a=logits-logits.max(1,keepdims=True);logp=a-np.log(np.exp(a).sum(1,keepdims=True));pred=logits.argmax(1)
    cm=np.bincount(y*3+pred,minlength=9).reshape(3,3)
    recall=np.diag(cm)/np.maximum(cm.sum(1),1);precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-20)
    losses=-logp[np.arange(len(y)),y]
    return dict(accuracy=float((pred==y).mean()),balanced_accuracy=float(recall.mean()),macro_f1=float(f1.mean()),
                balanced_nll=float(np.mean([losses[y==k].mean() for k in range(3)])),nll=float(losses.mean()),
                precision=precision.tolist(),recall=recall.tolist(),f1=f1.tolist(),confusion=cm.tolist(),predicted_counts=cm.sum(0).tolist(),support=cm.sum(1).tolist())


@torch.no_grad()
def evaluate(net,data,split,out=None,name=None):
    net.eval();pred=[]
    for a in range(0,len(data.starts[split]),512):
        x,y=data.batch(split,a,a+512);pred.append(net(x).cpu().numpy())
    logits=np.concatenate(pred);truth=data.y[split].cpu().numpy()
    # Weighted CE estimates a posterior under uniform class priors. Correct it
    # back to train priors for a separate natural-frequency decision rule.
    natural=logits-np.log(data.weights.cpu().numpy())[None,:]
    result=dict(balanced=metrics(logits,truth),natural=metrics(natural,truth))
    if out is not None:
        ix=data.starts[split]
        np.savez_compressed(out/f'{name}_cases.npz',logits=logits,natural_logits=natural,target=truth,
                            delta_cents=data.delta[split],timestamp=data.raw['timestamp'][ix-1],start=ix)
    return result


def update(net,opt,data,a,b,mode,rng):
    net.train();grads=[];sizes=[];total=0
    for k in range(a,b,32):
        opt.zero_grad(set_to_none=True);x,y=data.batch('train',k,min(k+32,b))
        loss=(F.cross_entropy(net(x),y,reduction='none')*data.weights[y]).mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();grads.append(torch.cat([p.grad.flatten() for p in net.parameters()]));sizes.append(len(y));total+=float(loss.detach())*len(y)
    g=torch.stack(grads);w=np.array(sizes)/sum(sizes)
    orders=[rng.permutation(len(w)) for _ in w]
    if mode=='pcgrad':w=w@coefficients((g@g.T).double().cpu().numpy(),orders)
    combined=torch.tensor(w,device='cuda',dtype=g.dtype)@g
    pos=0
    for p in net.parameters():p.grad=combined[pos:pos+p.numel()].view_as(p);pos+=p.numel()
    torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
    return total/(b-a)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    out=args.out;out.mkdir(exist_ok=True,parents=True);began=time.time()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Data(args.data);n=len(data.starts['train']);checks=fixtures()
    report=dict(status='training',classes=CLASSES,horizon_candles=8,threshold_cents=200,none_includes_boundary=True,
                task='terminal Close[t+8]-Close[t]; only 512 past candles; no reconstruction',
                loss='cross entropy, per-example inverse train-frequency class weights, arithmetic mean',
                optimizer='Adam with/without grouped PCGrad',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                train_priors=data.priors.tolist(),class_weights=data.weights.cpu().tolist(),scale_floor=data.floor,checks=checks,
                settings=dict(seeds=[17,29],epochs=12,batch=128,group=32,lr=[2e-4]*3+[1e-4]*3+[5e-5]*3+[2e-5]*3),trials={},metrics={})
    report['counts']={s:np.bincount(data.y[s].cpu(),minlength=3).tolist() for s in data.starts}
    # Boundary and causality check: changing all future prices/times leaves features identical.
    ix=int(data.starts['train'][0]);x0,_=data.batch('train',0,1)
    backup=data.prices[ix:ix+H].clone();data.prices[ix:ix+H]+=100
    x1,_=data.batch('train',0,1);data.prices[ix:ix+H]=backup
    assert torch.equal(x0,x1);checks['future_price_perturbation_max_feature_error']=float(abs(x0-x1).max())
    assert np.array_equal(np.where(np.array([-201,-200,0,200,201])>200,2,np.where(np.array([-201,-200,0,200,201]) < -200,0,1)),[0,1,1,1,2])
    init={}
    for seed in [17,29]:
        for mode in ['adam','pcgrad']:
            name=f'{mode}_{seed}';torch.manual_seed(seed);net=Classifier().cuda()
            report['parameters']=sum(p.numel() for p in net.parameters())
            sha=hashlib.sha256(torch.cat([p.flatten() for p in net.parameters()]).detach().cpu().numpy().tobytes()).hexdigest()
            if seed in init:assert init[seed]==sha
            init[seed]=sha
            opt=torch.optim.Adam(net.parameters(),lr=2e-4);rng=np.random.default_rng(seed+1000);best=float('inf');rows=[]
            for epoch,lr in enumerate(report['settings']['lr'],1):
                for group in opt.param_groups:group['lr']=lr
                loss=0;seen=0
                for a in range(0,n,128):
                    b=min(a+128,n);loss+=update(net,opt,data,a,b,mode,rng)*(b-a);seen+=b-a
                assert seen==n
                train=evaluate(net,data,'train');val=evaluate(net,data,'val')
                row=dict(epoch=epoch,lr=lr,examples_seen=seen,online_weighted_ce=loss/n,train=train,validation=val);rows.append(row)
                if val['balanced']['balanced_nll']<best:
                    best=val['balanced']['balanced_nll'];best_epoch=epoch
                    checkpoint(out/f'{name}_best.pt',dict(model=net.state_dict(),epoch=epoch))
                checkpoint(out/f'{name}_last.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,projection_rng=rng.bit_generator.state))
                report['trials'][name]=dict(best_epoch=best_epoch,best_validation_balanced_nll=best,epochs=epoch,initial_sha256=sha)
                save_json(out/f'{name}_curve.json',rows);save_json(out/'run.json',report)
                save_json(out/'status.json',dict(trial=name,epoch=epoch,train_nll=train['balanced']['balanced_nll'],val_nll=val['balanced']['balanced_nll'],val_macro_f1=val['balanced']['macro_f1'],elapsed=time.time()-began))
                if time.time()-began>1500:raise TimeoutError('classification 25-minute watchdog')
            del net,opt
    report['selection_frozen_before_test']=True;save_json(out/'run.json',report)
    for name in report['trials']:
        net=Classifier().cuda()
        for version in ['best','last']:
            net.load_state_dict(torch.load(out/f'{name}_{version}.pt',weights_only=False)['model'])
            label=name if version=='best' else name+'_last'
            report['metrics'][label]=evaluate(net,data,'test',out,label)
        del net
    truth=data.y['test'].cpu().numpy()
    report['baselines']={'always_none':metrics(np.tile([0.,1.,0.],(len(truth),1)),truth),
                         'train_prior':metrics(np.tile(np.log(data.priors),(len(truth),1)),truth),
                         'uniform':metrics(np.zeros((len(truth),3)),truth)}
    report['status']='complete';report['seconds']=time.time()-began
    save_json(out/'run.json',report);save_json(out/'status.json',dict(stage='complete',seconds=report['seconds']));print('COMPLETE',flush=True)

if __name__=='__main__':main()
