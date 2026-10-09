"""Matched full-data objective ablation; validation selection, then one final test."""
import sys,time,json,logging,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import *

LOG=logging.getLogger('reconstruction_v2');GAPS=[1,5,10,20,40];BATCH=16;EPOCHS=4

def save_model(net,path,kind,epoch,val_score):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=net.state_dict(),kind=kind,epoch=epoch,validation_score=val_score,
        code_sha256=old.ref.sha(Path(__file__)),core_sha256=old.ref.sha(Path(__file__).with_name('core.py'))),path)

@torch.no_grad()
def predict_model(net,data,lp,ts,anchors,lengths,kind):
    net.eval();angles=[];paths=[]
    for start in range(0,len(anchors),BATCH):
        b=batch(data,lp,ts,anchors[start:start+BATCH],lengths[start:start+BATCH])
        a,p=output(net,b,kind);angles.append(a.cpu().numpy());paths.append(p.cpu().numpy())
    return np.concatenate(angles),np.concatenate(paths)

def baseline_arrays(data,lp,ts,anchors,lengths,right_open=False):
    paths=[];angles=[]
    for start in range(0,len(anchors),128):
        b=batch(data,lp,ts,anchors[start:start+128],lengths[start:start+128])
        p=bridge(b,right_open);a=encode_path(p,b['left'],b['s0'])
        paths.append(p.cpu().numpy());angles.append(a.cpu().numpy())
    return np.concatenate(angles),np.concatenate(paths)

@torch.no_grad()
def validation(net,data,lp,ts,anchors,kind,bridge_losses):
    rows=[]
    for n in GAPS:
        lengths=np.full(len(anchors),n);a,p=predict_model(net,data,lp,ts,anchors,lengths,kind)
        target=lp[anchors[:,None]+np.arange(1,MAX+1)]-lp[anchors,2][:,None,None]
        pe=(p[:,:n,COLS]-target[:,:n,COLS])**2*1e8
        score=float(pe.mean())
        rows.append(dict(gap=n,price_HCL_mse_bps2=score,ratio_to_linear_bridge=score/bridge_losses[n]))
    return float(np.mean([r['ratio_to_linear_bridge'] for r in rows])),rows

def fit(kind,data,lp,ts,pieces,anchors,ck,bridge_losses):
    net=build(kind,ck);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
    folder=OUT/'models'/kind;folder.mkdir(parents=True,exist_ok=True)
    score,rows=validation(net,data,lp,ts,anchors,kind,bridge_losses);best=score;best_epoch=0
    save_model(net,folder/'best.pt',kind,0,score);old.dump(folder/'validation_epoch_0.json',rows)
    curve=[]
    for epoch in range(1,EPOCHS+1):
        aa,nn=old.make_spans(pieces['train'],40,41000+epoch);seen=0;sums={};started=time.time();last_log=started
        net.train()
        for start in range(0,len(aa),BATCH):
            a=aa[start:start+BATCH];n=nn[start:start+BATCH];b=batch(data,lp,ts,a,n)
            opt.zero_grad(set_to_none=True);pred,path=output(net,b,kind);terms=losses(pred,path,b)
            loss=terms['angle'] if kind=='A' else terms['total'];assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
            count=int(n.sum());seen+=count
            for key,value in terms.items():sums[key]=sums.get(key,0.)+float(value.detach())*count
            if time.time()-last_log>=30:
                progress=dict(variant=kind,epoch=epoch,epochs=EPOCHS,targets_seen=seen,targets_per_epoch=old.target_count(pieces['train']),seconds=time.time()-started)
                old.dump(OUT/'progress.json',progress);LOG.info('PROGRESS %s',progress);last_log=time.time()
        assert seen==old.target_count(pieces['train'])
        score,rows=validation(net,data,lp,ts,anchors,kind,bridge_losses)
        old.dump(folder/f'validation_epoch_{epoch}.json',rows)
        if score<best:best=score;best_epoch=epoch;save_model(net,folder/'best.pt',kind,epoch,score)
        save_model(net,folder/f'epoch_{epoch}.pt',kind,epoch,score)
        row=dict(epoch=epoch,targets=seen,updates=(len(aa)+BATCH-1)//BATCH,seconds=time.time()-started,
            validation_ratio_to_bridge=score,**{f'train_{k}':v/seen for k,v in sums.items()})
        curve.append(row);pd.DataFrame(curve).to_csv(folder/'curve.csv',index=False);LOG.info('EPOCH %s %s',kind,row)
    del net,opt
    if DEVICE=='cuda':torch.cuda.empty_cache()
    return dict(kind=kind,best_epoch=best_epoch,validation_ratio_to_bridge=best,epochs=EPOCHS)

def score_predictions(lp,ts,anchors,lengths,angles,paths,name,s0):
    true=lp[anchors[:,None]+np.arange(1,MAX+1)]-lp[anchors,2][:,None,None]
    prev=lp[anchors[:,None]+np.arange(MAX)]-lp[anchors,2][:,None,None]
    y=(2/np.pi)*np.arctan((true[:,:,COLS]-prev[:,:,COLS])/s0)
    mask=np.arange(MAX)[None,:]<lengths[:,None]
    metrics=[];candle=[];aggrows=[]
    groups={'ALL':mask}
    for n in np.unique(lengths):groups[f'gap_{n}']=mask&(lengths[:,None]==n)
    months=pd.to_datetime(ts[anchors+1],unit='s',utc=True).strftime('%Y-%m').to_numpy()
    for month in np.unique(months):groups[month]=mask&(months[:,None]==month)
    pprev=np.concatenate([np.zeros((len(paths),1)),paths[:,:-1,2]],1)
    tprev=np.concatenate([np.zeros((len(paths),1)),true[:,:-1,2]],1)
    invalid=(paths[:,:,0]<paths.max(2))|(paths[:,:,3]>paths.min(2))
    for group,m in groups.items():
        for j,ch in enumerate(['H','C','L']):
            col=COLS[j];ae=angles[:,:,j][m]-y[:,:,j][m];pe=(paths[:,:,col][m]-true[:,:,col][m])*10000
            metrics.append(dict(model=name,group=group,channel=ch,n=int(m.sum()),angle_mse=float(np.mean(ae**2)),
                angle_mae_degrees=float(np.mean(abs(ae))*90),price_mse_bps2=float(np.mean(pe**2)),price_mae_bps=float(np.mean(abs(pe)))))
        true_tv=float(np.sum(abs(true[:,:,2]-tprev)[m]));pred_tv=float(np.sum(abs(paths[:,:,2]-pprev)[m]))
        nz=(true[:,:,2]-tprev)[m]!=0
        candle.append(dict(model=name,group=group,invalid_fraction=float(invalid[m].mean()),
            close_total_variation_ratio=pred_tv/max(true_tv,1e-12),
            body_mae_bps=float(np.mean(abs((paths[:,:,2]-paths[:,:,1])-(true[:,:,2]-true[:,:,1]))[m])*10000),
            range_mae_bps=float(np.mean(abs((paths[:,:,0]-paths[:,:,3])-(true[:,:,0]-true[:,:,3]))[m])*10000),
            close_step_direction_accuracy=float(np.mean(np.sign((paths[:,:,2]-pprev)[m][nz])==np.sign((true[:,:,2]-tprev)[m][nz])))))
    pp=torch.as_tensor(paths);tt=torch.as_tensor(true);nn=torch.as_tensor(lengths);minutes=torch.as_tensor(ts[anchors+1]//60)
    for k in [3,5,10]:
        ap,valid=aggregate(pp,nn,minutes,k);at,_=aggregate(tt,nn,minutes,k)
        e=(ap-at).numpy();valid=valid.numpy()
        for j,ch in enumerate(['H','C','L']):
            values=e[:,:,COLS[j]][valid]*10000
            aggrows.append(dict(model=name,timeframe=f'M{k}',channel=ch,complete_bars=len(values),
                price_mse_bps2=float(np.mean(values**2)),price_mae_bps=float(np.mean(abs(values)))))
    ae=((angles-y)**2).mean(2);pe=((paths[:,:,COLS]-true[:,:,COLS])**2).mean(2)*1e8
    ce=(paths[:,:,2]-true[:,:,2])**2*1e8
    percase={key:(value*mask).sum(1)/lengths for key,value in [('angle_HCL',ae),('price_HCL',pe),('price_Close',ce)]}
    return metrics,candle,aggrows,percase

def evaluate(data,lp,ts,pieces,ck):
    anchors,lengths=old.make_spans(pieces['test'],40,991)
    (OUT/'predictions').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'test_origins.npz',anchors=anchors,lengths=lengths)
    methods={};original=build('A',ck)
    methods['original']=predict_model(original,data,lp,ts,anchors,lengths,'A');del original
    for kind in ['A','B','C']:
        net=build(kind,ck);saved=torch.load(OUT/'models'/kind/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(saved['model'])
        methods[kind]=predict_model(net,data,lp,ts,anchors,lengths,kind);del net
    methods['linear_bridge']=baseline_arrays(data,lp,ts,anchors,lengths)
    methods['right_open_bridge']=baseline_arrays(data,lp,ts,anchors,lengths,True)
    rows=[];shapes=[];aggregates=[];percase={}
    for name,(angles,paths) in methods.items():
        assert np.isfinite(angles).all() and np.isfinite(paths).all()
        np.savez_compressed(OUT/'predictions'/f'{name}.npz',angles=angles,relative_log_prices=paths)
        m,s,a,p=score_predictions(lp,ts,anchors,lengths,angles,paths,name,ck['scale']['S0'])
        rows+=m;shapes+=s;aggregates+=a;percase[name]=p
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(shapes).to_csv(OUT/'shape_metrics.csv',index=False)
    pd.DataFrame(aggregates).to_csv(OUT/'multiscale_metrics.csv',index=False)
    days,inv=np.unique(ts[anchors+1]//86400,return_inverse=True);count=np.bincount(inv,weights=lengths)
    rng=np.random.default_rng(4141);draw=rng.integers(0,len(days),(2000,len(days)));comparisons=[]
    for kind in ['A','B','C']:
        for base in ['original','linear_bridge','right_open_bridge']:
            for metric in percase[kind]:
                delta=percase[kind][metric]-percase[base][metric]
                total=np.bincount(inv,weights=delta*lengths);boot=total[draw].sum(1)/count[draw].sum(1)
                lo,hi=np.quantile(boot,[.025,.975]);value=float(np.average(delta,weights=lengths));den=float(np.average(percase[base][metric],weights=lengths))
                comparisons.append(dict(model=kind,baseline=base,metric=metric,delta=value,ci_low=lo,ci_high=hi,improvement_pct=-value/den*100))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    return anchors,lengths,methods

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'run.log',encoding='utf-8')])
    diagnosis=json.loads((OUT/'diagnostic/results.json').read_text(encoding='utf-8'))
    assert diagnosis['status']=='complete'
    if diagnosis['results'][0]['best_price_mae_bps']>.1:raise RuntimeError('Memorization has not established decoder capacity; investigate before full-data runs')
    lp,ts,pieces,anchors,_,info=old.read_data();ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    old.dump(OUT/'data_audit.json',info);old.dump(OUT/'scale.json',ck['scale'])
    manifest=dict(status='running',started_utc=str(pd.Timestamp.now(tz='UTC')),epochs_per_variant=EPOCHS,variants=['A','B','C'],
        source_sha256=info['source_sha256'],initial_checkpoint_sha256=old.ref.sha(CHECKPOINT),
        code_sha256=old.ref.sha(Path(__file__)),core_sha256=old.ref.sha(Path(__file__).with_name('core.py')),
        protocol_sha256=old.ref.sha(Path(__file__).with_name('protocol.md')),no_sampling=True,lr=1e-4,dropout=.25,weight_decay=.01)
    old.dump(OUT/'manifest.json',manifest)
    data=old.MaskedData(lp,ck['scale']['S0'],ck['scale']['H'])
    bridge_losses={}
    for n in GAPS:
        _,p=baseline_arrays(data,lp,ts,anchors,np.full(len(anchors),n))
        target=lp[anchors[:,None]+np.arange(1,MAX+1)]-lp[anchors,2][:,None,None]
        bridge_losses[n]=float(np.mean((p[:,:n,COLS]-target[:,:n,COLS])**2)*1e8)
    selection=[]
    for kind in ['A','B','C']:
        selection.append(fit(kind,data,lp,ts,pieces,anchors,ck,bridge_losses));old.dump(OUT/'selection.json',selection)
    manifest['phase']='final_evaluation';old.dump(OUT/'manifest.json',manifest)
    evaluate(data,lp,ts,pieces,ck)
    manifest.update(status='evaluation_complete',finished_utc=str(pd.Timestamp.now(tz='UTC')),elapsed_seconds=time.time()-start,
        total_training_targets=old.target_count(pieces['train'])*EPOCHS*3,
        metrics_sha256=old.ref.sha(OUT/'metrics.csv'))
    old.dump(OUT/'manifest.json',manifest);LOG.info('COMPLETE %s',manifest)

if __name__=='__main__':main()
