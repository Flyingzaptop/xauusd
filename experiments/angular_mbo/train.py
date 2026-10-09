"""Chronological complete passes, validation-only selection, one final July evaluation."""
import argparse
import hashlib
import logging
import os
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from config import *
from model import Model,event_features,loss,decode

LOG=logging.getLogger('mbo.train'); DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BATCH=16

class Data:
    def __init__(self):
        z=np.load(OUT/'candles.npz');self.ts=z['ts'];self.lp=z['lp'];self.a=z['angles']
        self.scale=json.loads((OUT/'scale.json').read_text());self.s0=self.scale['S0']
        self.days=json.loads((OUT/'manifest.json').read_text())['days']
    def batches(self,split,variant,batch_size=BATCH,limit=None):
        lo,hi={'train':(START,VALID),'validation':(VALID,TEST),'test':(TEST,END)}[split]
        count=0
        for day in self.days:
            if not day['samples'] or not lo<=day['date']<hi:continue
            meta=np.load(CACHE/(day['date']+'.npz'));ix=meta['ix'];gc=meta['gc']
            raw=np.load(day['file'],mmap_mode='r') if variant=='mbo' else None
            for b in range(0,len(ix),batch_size):
                jj=ix[b:b+batch_size]; xx=self.a[jj[:,None]+np.arange(-CONTEXT,0)[None,:]].transpose(0,2,1)
                left=self.lp[jj-1];truth=self.lp[jj]
                target=(2/np.pi)*np.arctan((truth[:,COLS]-left[:,COLS])/self.s0)
                pack=dict(x=xx,gc=gc[b:b+len(jj)],left=left,truth=truth,target=target)
                if raw is not None:
                    pack['events'],pack['mask']=event_features(raw[b:b+len(jj)],self.ts[jj].astype(np.uint64)*10**9,self.s0)
                t={k:torch.as_tensor(np.array(v,copy=True),device=DEVICE,dtype=torch.bool if k=='mask' else torch.float32) for k,v in pack.items()}
                # Anchor subtraction prevents float32 cancellation at log-price ~8; targets stay relative.
                t['left']=torch.as_tensor(left-left[:,2:3],device=DEVICE,dtype=torch.float32)
                t['truth']=torch.as_tensor(truth-left[:,2:3],device=DEVICE,dtype=torch.float32)
                yield jj,t
                count+=1
                if limit and count>=limit:return
            if raw is not None:del raw

def predict(net,b):return net(b['x'],b.get('gc'),b.get('events'),b.get('mask'))

def metrics(pred,jj,data):
    left=data.lp[jj-1];truth=data.lp[jj]
    target=(2/np.pi)*np.arctan((truth[:,COLS]-left[:,COLS])/data.s0)
    path=left[:,COLS]+data.s0*np.tan(np.clip(pred,-1+1e-5,1-1e-5)*(np.pi/2))
    err=path-truth[:,COLS]
    dollars=np.exp(path)-np.exp(truth[:,COLS])
    direction=(np.sign(path[:,1]-left[:,2])==np.sign(truth[:,2]-left[:,2])).mean()
    invalid=(path[:,0]<np.maximum(left[:,2],path[:,1]))|(path[:,2]>np.minimum(left[:,2],path[:,1]))
    return dict(samples=len(jj),angular_mse=float(np.mean((pred-target)**2)),price_mse_bps2=float(np.mean(err**2)*1e8),
        price_mae_usd=float(np.mean(abs(dollars))),close_mse_bps2=float(np.mean(err[:,1]**2)*1e8),
        close_movement_ratio=float(np.mean(abs(path[:,1]-left[:,2]))/max(np.mean(abs(truth[:,2]-left[:,2])),1e-12)),
        close_direction_accuracy=float(direction),invalid_candle_fraction=float(invalid.mean()),
        per_channel_mse_bps2=dict(zip(['H','C','L'],(np.mean(err**2,axis=0)*1e8).tolist())))

@torch.no_grad()
def evaluate(net,data,split,limit=None):
    net.eval();pred=[];indices=[]
    for jj,b in data.batches(split,net.variant,limit=limit):
        p=predict(net,b);assert torch.isfinite(p).all();pred.append(p.cpu().numpy());indices.append(jj)
    p=np.concatenate(pred);jj=np.concatenate(indices)
    return metrics(p,jj,data),p,jj

def fit(variant,data,epochs=30,smoke=False):
    torch.manual_seed(17);np.random.seed(17);torch.set_num_threads(4)
    net=Model(variant).to(DEVICE); opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2,min_lr=1e-6)
    path=OUT/('smoke' if smoke else 'models')/variant;path.mkdir(parents=True,exist_ok=True)
    best=float('inf');stale=0;curve=[];best_epoch=0
    expected=sum(d['samples'] for d in data.days if START<=d['date']<VALID)
    for epoch in range(1,epochs+1):
        net.train();seen=0;total=0.;start=time.time();last=start
        for jj,b in data.batches('train',variant,limit=3 if smoke else None):
            opt.zero_grad(set_to_none=True);pred=predict(net,b)
            value=loss(pred,b['target'],b['left'],b['truth'],data.s0)
            assert torch.isfinite(value),'Nonfinite loss'
            value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            seen+=len(jj);total+=float(value.detach())*len(jj)
            if time.time()-last>30:
                dump(OUT/'progress.json',dict(stage='training',variant=variant,epoch=epoch,seen=seen,total=expected,
                    last_target_utc=str(np.datetime64(int(data.ts[jj[-1]]),'s')),loss=total/seen,elapsed_seconds=time.time()-start))
                last=time.time()
        if not smoke:assert seen==expected,'Every eligible training minute must appear once per epoch'
        validation,_,_=evaluate(net,data,'train' if smoke else 'validation',limit=2 if smoke else None)
        score=validation['price_mse_bps2'];sched.step(score)
        row=dict(epoch=epoch,train_samples=seen,train_loss=total/seen,validation=validation,
                 seconds=time.time()-start,learning_rate=opt.param_groups[0]['lr']);curve.append(row)
        if score<best:
            best=score;stale=0;best_epoch=epoch
            torch.save(dict(model=net.state_dict(),variant=variant,epoch=epoch,scale=data.scale,
                parameters=sum(p.numel() for p in net.parameters()),protocol=dict(context=CONTEXT,events=EVENTS,
                cutoff='strict ts_recv < target minute start',dropout=.25,initial_lr=1e-4)),path/'best.pt')
        else:stale+=1
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            epoch=epoch,curve=curve,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE=='cuda' else []),path/'latest.pt')
        dump(path/'curve.json',curve);LOG.info('%s EPOCH %s',variant,row)
        if stale>=5:break
    return dict(variant=variant,epochs=len(curve),best_epoch=best_epoch,best_validation_mse_bps2=best,
        train_samples_per_epoch=expected,parameters=sum(p.numel() for p in net.parameters()))

def run(smoke=False,epochs=30):
    data=Data();results=[]
    if smoke:
        for variant in ['m1','gc','mbo']:results.append(fit(variant,data,1,True))
        dump(OUT/'smoke_training.json',results);return
    for variant in ['m1','gc','mbo']:results.append(fit(variant,data,epochs))
    tests={};reference=None;test_predictions={}
    for variant in ['m1','gc','mbo']:
        ck=torch.load(OUT/'models'/variant/'best.pt',map_location=DEVICE,weights_only=False)
        net=Model(variant).to(DEVICE);net.load_state_dict(ck['model'])
        tests[variant],pred,ix=evaluate(net,data,'test')
        if reference is not None:assert np.array_equal(ix,reference)
        reference=ix;test_predictions[variant]=pred;np.savez(OUT/f'test_{variant}.npz',indices=ix,angles=pred)
    tests['zero_angles']=metrics(np.zeros((len(reference),3)),reference,data)
    train_ix=np.concatenate([np.load(CACHE/(d['date']+'.npz'))['ix'] for d in data.days if d['samples'] and START<=d['date']<VALID])
    targets=(2/np.pi)*np.arctan((data.lp[train_ix][:,COLS]-data.lp[train_ix-1][:,COLS])/data.s0)
    tests['train_mean_angles']=metrics(np.broadcast_to(targets.mean(0),(len(reference),3)),reference,data)
    # Paired UTC-day bootstrap: temporally correlated minutes are resampled as days.
    days=data.ts[reference]//86400;unique=np.unique(days);errors={}
    for name,p in test_predictions.items():
        path=data.lp[reference-1][:,COLS]+data.s0*np.tan(np.clip(p,-1+1e-5,1-1e-5)*(np.pi/2))
        errors[name]=np.mean((path-data.lp[reference][:,COLS])**2,axis=1)*1e8
    comparisons={};rng=np.random.default_rng(1729)
    for other in ['m1','gc']:
        delta=errors['mbo']-errors[other]
        sums=np.array([delta[days==d].sum() for d in unique]);ns=np.array([(days==d).sum() for d in unique])
        draws=rng.integers(len(unique),size=(2000,len(unique)))
        boots=sums[draws].sum(1)/ns[draws].sum(1)
        comparisons['mbo_minus_'+other]=dict(mean_price_mse_delta_bps2=float(delta.mean()),
            paired_day_bootstrap_95ci=np.quantile(boots,[.025,.975]).tolist(),days=len(unique),negative_is_better=True)
    dump(OUT/'results.json',dict(training=results,test=tests,comparisons=comparisons,
        selected_on='June validation price MSE, no July tuning',seed=17))
    manifest=json.loads((OUT/'manifest.json').read_text());manifest['status']='complete';dump(OUT/'manifest.json',manifest)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args();run(args.smoke,args.epochs)
