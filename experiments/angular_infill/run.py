"""Curriculum reconstruction of angular candles from512 left/right context tokens."""
from __future__ import annotations
import os,sys,json,time,argparse,importlib.util,logging,base64
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/angular_infill_512_20261008'
spec=importlib.util.spec_from_file_location('walk_reference',ROOT/'experiments/angular_walkforward/run.py')
wf=importlib.util.module_from_spec(spec);spec.loader.exec_module(wf)
ref=wf.ref;DEVICE=wf.DEVICE;COLS=[0,2,3];LEFT=512;RIGHT=512;MAX_GAP=40;BATCH=16;EPOCHS=30
STAGES=[5,10,20,40];VAL_LENGTHS=[1,2,3,4,5,10,20,40];LOG=logging.getLogger('infill')
CHECKPOINT=ROOT/'reports/angular_walkforward_20261008/checkpoints/epoch_30.pt'

def dump(path,value):wf.dump(path,value)

def partitions(lp,ts,cut,val_end):
    limits=np.unique(np.r_[0,np.flatnonzero(np.diff(ts)!=60)+1,np.searchsorted(ts,cut),np.searchsorted(ts,val_end),len(ts)])
    pieces={'train':[],'val':[],'test':[]}
    for a,b in zip(limits[:-1],limits[1:]):
        if b<=a:continue
        name='train' if ts[b-1]<cut else ('val' if ts[a]<val_end else 'test')
        if b-a>LEFT+55+RIGHT:pieces[name].append((int(a),int(b)))
    return pieces

def target_count(pieces):return sum(b-RIGHT-(a+LEFT+55) for a,b in pieces)

def make_spans(pieces,cap,seed):
    rng=np.random.default_rng(seed);anchors=[];lengths=[]
    for a,b in pieces:
        cursor=a+LEFT+55;stop=b-RIGHT
        while cursor<stop:
            n=min(int(rng.integers(1,cap+1)),stop-cursor)
            anchors.append(cursor-1);lengths.append(n);cursor+=n
    anchors=np.asarray(anchors,np.int64);lengths=np.asarray(lengths,np.int64)
    assert lengths.sum()==target_count(pieces)
    assert len(anchors)==0 or np.all(np.diff(anchors)>0)
    return anchors,lengths

def read_data():
    source=ROOT/'data/M1/xauusd_m1.parquet';df=pd.read_parquet(source)
    price=df[['high','open','close','low']].to_numpy(float)
    assert np.isfinite(price).all() and (price>0).all()
    assert ((price[:,0]>=price.max(1))&(price[:,3]<=price.min(1))).all()
    ts=df.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64);assert np.all(np.diff(ts)>0)
    ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    previous=json.loads((CHECKPOINT.parents[1]/'manifest.json').read_text(encoding='utf-8'))
    assert ref.sha(source)==previous['source_sha256']
    scale=ck['scale'];boundary=pd.Timestamp(scale['fit_end_exclusive']);validation_end=boundary+pd.DateOffset(months=3)
    lp=np.log(price);pieces=partitions(lp,ts,int(boundary.timestamp()),int(validation_end.timestamp()))
    candidates=np.concatenate([np.arange(a+LEFT+55-1,b-RIGHT-MAX_GAP) for a,b in pieces['val'] if b-RIGHT-MAX_GAP>a+LEFT+55-1])
    val_anchors=np.sort(np.random.default_rng(1701).choice(candidates,min(512,len(candidates)),replace=False))
    assert len(val_anchors)>=128,'Insufficient validation windows with512+512 context'
    info=dict(source_sha256=previous['source_sha256'],source_rows=len(ts),train_boundary=str(boundary),validation_end=str(validation_end),
        final_test_end=str(df.timestamp.max()),left_tokens=LEFT,right_tokens=RIGHT,raw_left_support=LEFT+55,
        eligible_hidden_targets={k:target_count(v) for k,v in pieces.items()},continuous_segments={k:len(v) for k,v in pieces.items()},
        validation_anchors=len(val_anchors),init_checkpoint=str(CHECKPOINT),init_checkpoint_sha256=ref.sha(CHECKPOINT),
        training_sampling=False,training_shuffle=False,exposure='Previously researched data; validation and final test split for this experiment')
    return lp,ts,pieces,val_anchors,ck,info

class MaskedData:
    def __init__(self,lp,s0,H,device=DEVICE):
        self.base=ref.AngularHistory(lp,s0,H,device=device);self.a=self.base.a;self.device=device;self.s0=s0
        self.rel=torch.arange(-LEFT+1,MAX_GAP+RIGHT+1,device=device)
        self.past=self.rel[:,None]-torch.as_tensor(ref.LAGS,device=device)[None,:]
        self.h=torch.arange(1,MAX_GAP+1,device=device)
    def x(self,anchors,lengths,right=True):
        anchors=torch.as_tensor(anchors,device=self.device,dtype=torch.long);n=torch.as_tensor(lengths,device=self.device,dtype=torch.long)
        ix=(anchors[:,None]+self.rel[None,:]).clamp(0,len(self.a)-1)
        value=self.a[ix]
        visible=(self.rel[None,:]<=0)
        if right:visible=visible|((self.rel[None,:]>n[:,None])&(self.rel[None,:]<=n[:,None]+RIGHT))
        else:visible=visible.expand(len(n),-1)
        pv=(self.past[None,:,:]<=0)
        if right:pv=pv|((self.past[None,:,:]>n[:,None,None])&(self.past[None,:,:]<=n[:,None,None]+RIGHT))
        available=(visible[:,:,None]&pv)[:,:,:,None].expand(-1,-1,-1,4).flatten(2)
        values=value.masked_fill(~available,0)
        hidden=((self.rel[None,:]>=1)&(self.rel[None,:]<=n[:,None])).float()
        return torch.cat([values,available.float(),hidden[:,:,None]],2).transpose(1,2).contiguous()
    def y(self,anchors,lengths):
        anchors=torch.as_tensor(anchors,device=self.device,dtype=torch.long);n=torch.as_tensor(lengths,device=self.device,dtype=torch.long)
        target=self.a[anchors[:,None]+self.h[None,:]][:,:,COLS]
        return target,self.h[None,:]<=n[:,None]

class Encoder(nn.Module):
    def __init__(self,old):
        super().__init__();self.input=nn.Conv1d(73,64,1)
        self.blocks=nn.Sequential(*[wf.Block(64,2**i) for i in range(10)])
        for block in self.blocks[7:]:nn.init.zeros_(block.norm2.weight);nn.init.zeros_(block.norm2.bias)
        own=self.state_dict()
        for key in own:
            if key=='input.weight':
                own[key].zero_();own[key][:,:36].copy_(old[key])
            elif key in old and own[key].shape==old[key].shape:own[key].copy_(old[key])
        self.load_state_dict(own)
    def forward(self,x):return self.blocks(self.input(x))

class Infiller(nn.Module):
    def __init__(self,old=None):
        super().__init__();old=wf.TCN().state_dict() if old is None else old
        self.left=Encoder(old);self.right=Encoder(old);self.head=nn.Conv1d(128,3,1)
        with torch.no_grad():
            self.head.weight.zero_();self.head.weight[:,:64].copy_(old['head.weight']);self.head.bias.copy_(old['head.bias'])
    def forward(self,x):
        left=self.left(x);left=F.pad(left,(1,0))[:,:,:x.shape[-1]]
        right=self.right(x.flip(-1)).flip(-1)
        return ((1-1e-6)*torch.tanh(self.head(torch.cat([left,right],1))))[:,:,LEFT:LEFT+MAX_GAP].transpose(1,2)

def decode(angles,left,s0):
    delta=s0*np.tan(np.pi*np.clip(angles.astype(float),-1+1e-6,1-1e-6)/2)
    result=np.empty((*angles.shape[:2],4),float)
    result[:,:,COLS]=left[:,None,COLS]+np.cumsum(delta,axis=1)
    result[:,0,1]=left[:,2];result[:,1:,1]=result[:,:-1,2]
    return result

def path_angles(path,left,s0):
    previous=np.concatenate([left[:,None,:],path[:,:-1]],axis=1)
    return ref.encode_next(path,previous,s0)[:,:,COLS]

@torch.no_grad()
def predict(net,data,anchors,lengths,right=True):
    net.eval();allp=[]
    for i in range(0,len(anchors),BATCH):allp.append(net(data.x(anchors[i:i+BATCH],lengths[i:i+BATCH],right)).cpu().numpy())
    return np.concatenate(allp)

@torch.no_grad()
def causal_rollout(old_model,lp,anchors,steps,scale):
    old_model.eval();outputs=[];offset=torch.arange(128,device=DEVICE);lags=torch.as_tensor(ref.LAGS,device=DEVICE)
    denom=torch.as_tensor(scale['S0']*ref.LAGS**scale['H'],device=DEVICE,dtype=torch.float32)
    for start in range(0,len(anchors),64):
        a=anchors[start:start+64];history=lp[a[:,None]+np.arange(-182,1)]
        centered=history-lp[a,2][:,None,None]
        buf=torch.zeros((len(a),183+steps,4),device=DEVICE);buf[:,:183]=torch.as_tensor(centered,device=DEVICE,dtype=torch.float32)
        predictions=[]
        for j in range(steps):
            end=182+j;ids=end-127+offset
            delta=(buf[:,ids,:,None].transpose(2,3)-buf[:,ids[:,None]-lags[None,:],:])/denom[None,None,:,None]
            features=((2/np.pi)*torch.atan(delta)).flatten(2).transpose(1,2).contiguous()
            p=old_model(features)[:,:,-1];predictions.append(p.cpu().numpy())
            buf[:,end+1,COLS]=buf[:,end,COLS]+scale['S0']*torch.tan(p*(np.pi/2))
            buf[:,end+1,1]=buf[:,end,2]
        outputs.append(np.stack(predictions,axis=1))
    return np.concatenate(outputs)

def errors(lp,anchors,lengths,pred,s0):
    h=np.arange(1,pred.shape[1]+1);mask=h[None,:]<=lengths[:,None]
    actual=lp[anchors[:,None]+h];previous=lp[anchors[:,None]+h-1]
    y=ref.encode_next(actual,previous,s0)[:,:,COLS]
    decoded=decode(pred,lp[anchors],s0)
    ae=(pred-y)**2;pe=(decoded-actual)**2*1e8
    return dict(angle=float(ae[mask].mean()),price_HCL=float(pe[:,:,COLS][mask].mean()),price_Close=float(pe[:,:,2][mask].mean())),decoded,y,mask

def gate_check(rows,cap,stage_epochs,streak):
    at_cap=[r for r in rows if r['gap']==cap][0]
    passed=at_cap['price_HCL_ratio']<=.8 and at_cap['price_Close_ratio']<=.8 and all(r['angle_ratio']<=1 for r in rows)
    streak=streak+1 if passed else 0
    return passed,streak,stage_epochs>=3 and streak>=2

def validate(net,data,lp,anchors,cap,causal,scale):
    rows=[]
    for n in VAL_LENGTHS:
        if n>cap:continue
        lengths=np.full(len(anchors),n);p=predict(net,data,anchors,lengths)
        ours,_,_,_=errors(lp,anchors,lengths,p,scale['S0'])
        baseline,_,_,_=errors(lp,anchors,lengths,causal,scale['S0'])
        rows.append(dict(gap=n,**ours,**{k+'_ratio':ours[k]/baseline[k] for k in ours}))
    return rows

def scheduler(opt):return torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=3,threshold=.001,min_lr=1e-6)

def fit(data,lp,pieces,val_anchors,ck,resume=False):
    torch.manual_seed(17);net=Infiller(ck['model']).to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01);sched=scheduler(opt)
    old=wf.TCN().to(DEVICE);old.load_state_dict(ck['model']);old.eval()
    cache=OUT/'validation_causal.npz'
    if cache.exists():
        cached=np.load(cache);assert np.array_equal(cached['anchors'],val_anchors)
        causal=cached['angles'];assert causal.shape==(len(val_anchors),MAX_GAP,3)
    else:
        causal=causal_rollout(old,lp,val_anchors,MAX_GAP,ck['scale']);np.savez_compressed(cache,anchors=val_anchors,angles=causal)
    del old
    first=1;stage=0;stage_epochs=0;streak=0;curve=[];trained_cap=5
    if resume and (OUT/'resume.json').exists():
        last=torch.load(json.loads((OUT/'resume.json').read_text())['path'],map_location='cpu',weights_only=False)
        assert last['code_sha256']==ref.sha(Path(__file__))
        net.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);sched.load_state_dict(last['scheduler'])
        first=last['epoch']+1;stage=last['next_stage'];stage_epochs=last['next_stage_epochs'];streak=last['next_streak'];curve=last['curve'];trained_cap=last['trained_cap']
        torch.set_rng_state(last['cpu_rng'])
        if DEVICE=='cuda':torch.cuda.set_rng_state_all(last['cuda_rng'])
    for epoch in range(first,EPOCHS+1):
        cap=STAGES[stage];anchors,lengths=make_spans(pieces['train'],cap,17000+epoch)
        stage_epochs+=1;trained_cap=cap;net.train();total=0.;seen=0;started=time.time();last_log=started;lr=opt.param_groups[0]['lr']
        for start in range(0,len(anchors),BATCH):
            a=anchors[start:start+BATCH];n=lengths[start:start+BATCH];x=data.x(a,n);y,mask=data.y(a,n)
            opt.zero_grad(set_to_none=True);p=net(x);squared=(p-y).square()*mask[:,:,None]
            loss=squared.sum()/(mask.sum()*3);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
            total+=squared.detach().sum().item();seen+=int(n.sum())
            if time.time()-last_log>=30:
                progress=dict(epoch=epoch,epochs=EPOCHS,stage_max=cap,hidden_targets_seen=seen,hidden_targets_per_epoch=target_count(pieces['train']),
                    spans_done=start+len(a),spans_per_epoch=len(anchors),loss=total/(seen*3),lr=lr,seconds=time.time()-started)
                dump(OUT/'progress.json',progress);LOG.info('TRAIN %s',progress);last_log=time.time()
        assert seen==target_count(pieces['train']);train_loss=total/(seen*3);sched.step(train_loss)
        validation=validate(net,data,lp,val_anchors,cap,causal,ck['scale'])
        passed,streak,advance=gate_check(validation,cap,stage_epochs,streak)
        row=dict(epoch=epoch,stage_max=cap,stage_epochs=stage_epochs,targets=seen,spans=len(anchors),updates=(len(anchors)+BATCH-1)//BATCH,
            train_angle_mse=train_loss,learning_rate=lr,seconds=time.time()-started,gate_passed=passed,passing_streak=streak)
        curve.append(row);pd.DataFrame(curve).to_csv(OUT/'training_curve.csv',index=False)
        dump(OUT/f'validation_epoch_{epoch:02d}.json',dict(epoch=epoch,cap=cap,metrics=validation,passed=passed,advance=advance))
        if advance and stage<len(STAGES)-1:
            stage+=1;stage_epochs=0;streak=0;sched=scheduler(opt)
        saved=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,trained_cap=trained_cap,
            next_stage=stage,next_stage_epochs=stage_epochs,next_streak=streak,curve=curve,scale=ck['scale'],left=LEFT,right=RIGHT,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE=='cuda' else [],code_sha256=ref.sha(Path(__file__)))
        path=OUT/'checkpoints'/f'epoch_{epoch:02d}.pt';path.parent.mkdir(exist_ok=True);temp=path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(path)
        dump(OUT/'resume.json',dict(path=str(path),completed_epoch=epoch,trained_max_gap=cap,next_max_gap=STAGES[stage]))
        LOG.info('EPOCH COMPLETE %s; next maximum=%s',row,STAGES[stage])
    return net,trained_cap,curve

def bridges(lp,anchors,lengths,s0):
    left=lp[anchors];right=lp[anchors+lengths+1]
    paths={};h=np.arange(1,MAX_GAP+1)[None,:,None]
    linear=left[:,None,:]+(right-left)[:,None,:]*h/(lengths[:,None,None]+1)
    linear[:,0,1]=left[:,2];linear[:,1:,1]=linear[:,:-1,2];paths['linear_bridge']=linear
    op=linear.copy();op[:,:,2]=left[:,None,2]+(right[:,1]-left[:,2])[:,None]*h[:,:,0]/lengths[:,None]
    op[:,0,1]=left[:,2];op[:,1:,1]=op[:,:-1,2];paths['right_open_bridge']=op
    return {name:path_angles(path,left,s0) for name,path in paths.items()}

def evaluate(net,data,lp,ts,pieces,cap,ck):
    a,n=make_spans(pieces['test'],cap,991);old=wf.TCN().to(DEVICE);old.load_state_dict(ck['model']);old.eval()
    LOG.info('FINAL EVALUATION %s gaps, %s hidden candles, maximum=%s',len(a),int(n.sum()),cap)
    preds=dict(infiller=predict(net,data,a,n),right_removed=predict(net,data,a,n,right=False))
    causal=causal_rollout(old,lp,a,cap,ck['scale']);preds['causal_rollout']=np.pad(causal,((0,0),(0,MAX_GAP-cap),(0,0)))
    preds.update(bridges(lp,a,n,ck['scale']['S0']))
    np.savez_compressed(OUT/'test_predictions.npz',anchors=a,lengths=n,**preds)
    rows=[];shapes=[];percase={};truth=None
    for name,p in preds.items():
        _,decoded,y,mask=errors(lp,a,n,p,ck['scale']['S0']);truth=lp[a[:,None]+np.arange(1,MAX_GAP+1)]
        ae=(p-y)**2;pe=(decoded-truth)**2*1e8
        percase[name]=dict(angle=np.sum(ae*mask[:,:,None],axis=(1,2))/(n*3),
            price_HCL=np.sum(pe[:,:,COLS]*mask[:,:,None],axis=(1,2))/(n*3),price_Close=np.sum(pe[:,:,2]*mask,axis=1)/n)
        selectors={'ALL':mask}
        for length in np.unique(n):selectors[f'gap_{length}']=mask&(n[:,None]==length)
        for h in range(1,cap+1):selectors[f'position_{h}']=mask&(np.arange(MAX_GAP)[None,:]==h-1)
        for group,select in selectors.items():
            for j,ch in enumerate(['H','C','L']):
                actual=y[:,:,j][select];pred=p[:,:,j][select];err=pred-actual;col=COLS[j]
                variance=float(np.var(actual));nonzero=actual!=0
                rows.append(dict(model=name,group=group,channel=ch,n=int(select.sum()),angle_mse=float(np.mean(err**2)),
                    angle_mae_degrees=float(np.mean(abs(err))*90),angle_r2=1-float(np.mean(err**2))/variance if variance else None,
                    angle_sign_accuracy=float(np.mean(np.sign(actual[nonzero])==np.sign(pred[nonzero]))),
                    price_mse_bps2=float(pe[:,:,col][select].mean()),price_mae_bps=float(np.mean(abs(decoded[:,:,col][select]-truth[:,:,col][select]))*10000)))
        valid=(decoded[:,:,0]>=decoded.max(2))&(decoded[:,:,3]<=decoded.min(2))
        shapes.append(dict(model=name,invalid_ohlc_fraction=float(1-valid[mask].mean()),
            range_mae_bps=float(np.mean(abs((decoded[:,:,0]-decoded[:,:,3])-(truth[:,:,0]-truth[:,:,3]))[mask])*10000),
            body_mae_bps=float(np.mean(abs((decoded[:,:,2]-decoded[:,:,1])-(truth[:,:,2]-truth[:,:,1]))[mask])*10000)))
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(shapes).to_csv(OUT/'candle_metrics.csv',index=False)
    days,inv=np.unique(ts[a+1]//86400,return_inverse=True);counts=np.bincount(inv,weights=n)
    draw=np.random.default_rng(701).integers(0,len(days),(2000,len(days)));comparisons=[]
    for name in preds:
        if name=='infiller':continue
        for metric in percase['infiller']:
            delta=percase['infiller'][metric]-percase[name][metric]
            sums=np.bincount(inv,weights=delta*n);boot=sums[draw].sum(1)/counts[draw].sum(1)
            lo,hi=np.quantile(boot,[.025,.975]);mean=float(np.average(delta,weights=n));base=float(np.average(percase[name][metric],weights=n))
            comparisons.append(dict(baseline=name,metric=metric,delta=mean,ci_low=lo,ci_high=hi,improvement_pct=-mean/base*100))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    # Recompute ALL angular metrics directly from saved arrays and raw labels.
    saved=np.load(OUT/'test_predictions.npz');checks=0
    for name in preds:
        p=saved[name];actual=ref.encode_next(lp[a[:,None]+np.arange(1,MAX_GAP+1)],lp[a[:,None]+np.arange(MAX_GAP)],ck['scale']['S0'])[:,:,COLS]
        for j,ch in enumerate(['H','C','L']):
            score=float(np.mean((p[:,:,j][mask]-actual[:,:,j][mask])**2));reported=[r['angle_mse'] for r in rows if r['model']==name and r['group']=='ALL' and r['channel']==ch][0]
            assert abs(score-reported)<1e-9;checks+=1
    dump(OUT/'verification.json',dict(status='passed',saved_metric_checks=checks,hidden_targets=int(n.sum()),test_gaps=len(a),
        decoder_anchors='Only last observed left candle; internal prices integrated from predicted angles',max_gap=cap))
    return a,n,preds,pd.DataFrame(rows),pd.DataFrame(shapes),pd.DataFrame(comparisons)

def report(lp,ts,info,scale,curve,cap,evaluation,elapsed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a,n,preds,metrics,shapes,comparisons=evaluation
    selected=np.unique(np.quantile(np.flatnonzero(n==cap) if np.any(n==cap) else np.arange(len(a)),[.15,.5,.85]).astype(int))
    fig,axes=plt.subplots(1,len(selected),figsize=(15,4.5),squeeze=False)
    for ax,i in zip(axes[0],selected):
        anchor=a[i];length=n[i];base=lp[anchor,2];x=np.arange(-5,length+7)
        ax.plot(x,(lp[anchor+x,2]-base)*10000,'k.-',label='Actual Close')
        for name,color in [('infiller','#d65b3a'),('causal_rollout','#8a91a3'),('linear_bridge','#298eb5')]:
            path=decode(preds[name][i:i+1],lp[anchor:anchor+1],scale['S0'])[0,:length,2]
            ax.plot(np.arange(1,length+1),(path-base)*10000,'o-',label=name,color=color,ms=3)
        ax.axvspan(.5,length+.5,color='#dddddd',alpha=.4);ax.set(title=f'{pd.to_datetime(ts[anchor],unit="s",utc=True):%Y-%m-%d %H:%M}; gap{length}',xlabel='Minute relative to left anchor',ylabel='Log-price (bps)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'examples.png',dpi=160);plt.close(fig)
    image=base64.b64encode((OUT/'examples.png').read_bytes()).decode()
    def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.6g}',border=0)
    headline=comparisons[(comparisons.baseline=='causal_rollout')&(comparisons.metric=='price_Close')].iloc[0]
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Angular infilling512</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:32px auto;padding:0 24px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}}img{{max-width:100%}}</style>
<h1>Восстановление пропусков:512 шагов слева и справа</h1>
<p>Финальный обученный максимум: <b>{cap} свечей</b>. MSE восстановленного Close относительно причинного прогноза: улучшение {headline.improvement_pct:.3f}%.
Это восстановление с доступом к будущему, а не прогноз торговли.</p>
<p>Стартовые пропуски1–5, затем1–10,1–20,1–40 только после validation-gate. На каждом этапе минимум3 эпохи и два успешных замера подряд:
на максимальной длине снижение ценовой MSE H/C/L и Close не менее20%, угловая ошибка не хуже причинной модели на всех проверяемых длинах.
Если gate не пройден, окно не расширялось. Полная история переходов: training_curve.csv и validation_epoch_*.json.</p>
<p>Train заканчивается {info['train_boundary']}; validation до {info['validation_end']}; оставшийся период до {info['final_test_end']} — final test.
Период ранее исследовался. Контекст512+512 и разрывы данных оставили следующее количество допустимых скрытых целей: {info['eligible_hidden_targets']}.
Каждая допустимая обучающая цель включена один раз за эпоху. Пропуски динамические, порядок хронологический.</p>
<p>Двунаправленная TCN инициализирована предыдущим epoch30, вход только углы и маски. Dropout25%, начальный LR1e-4, AdamW weight_decay0.01.
Loss — MSE углов H/C/L только в пропусках. Цены интегрированы от последней видимой свечи; истинные скрытые предыдущие цены не использовались.
Open каждой внутренней свечи равен предыдущему предсказанному Close. Обучение и оценка текущего запуска заняли {elapsed/3600:.2f} ч.</p>
<h2>Общие ошибки</h2>{table(metrics[metrics.group=='ALL'])}
<h2>Парные сравнения</h2><p>Отрицательная delta — лучше. Интервалы95% по дням,2000 повторов, без поправки на множественные сравнения.</p>{table(comparisons)}
<p>linear_bridge использует известные цены по обе стороны. right_open_bridge дополнительно использует Open первой правой свечи как оценку последнего скрытого Close.
right_removed — проверка той же сети с удалённым правым контекстом, вне её обучающего распределения, а не отдельно обученный прогнозист.</p>
<h2>Свечная геометрия</h2>{table(shapes)}
<h2>Примеры</h2><img src="data:image/png;base64,{image}">
<h2>Эпохи и curriculum</h2>{table(pd.DataFrame(curve))}
<p>Артефакты: checkpoints/, test_predictions.npz, metrics.csv, comparisons.csv, verification.json. Ошибки по каждой длине и позиции — в metrics.csv.</p>
<pre>python experiments/angular_infill/run.py train
python experiments/angular_infill/run.py train --resume</pre></html>'''
    (OUT/'results.html').write_text(text,encoding='utf-8')

def main(resume=False):
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'run.log',encoding='utf-8')])
    lp,ts,pieces,val_anchors,ck,info=read_data();dump(OUT/'data_audit.json',info);dump(OUT/'scale.json',ck['scale'])
    np.savez_compressed(OUT/'partition_segments.npz',**{k:np.asarray(v) for k,v in pieces.items()},val_anchors=val_anchors)
    manifest=dict(status='running',started_utc=str(pd.Timestamp.now(tz='UTC')),script_sha256=ref.sha(Path(__file__)),protocol_sha256=ref.sha(Path(__file__).with_name('protocol.md')),
        source_sha256=info['source_sha256'],left_context=LEFT,right_context=RIGHT,stages=STAGES,epochs=EPOCHS,initial_lr=1e-4,dropout=.25,
        parameters=sum(v.numel() for v in Infiller(ck['model']).parameters()))
    dump(OUT/'manifest.json',manifest);LOG.info('AUDIT %s',info);LOG.info('MODEL %s',manifest)
    data=MaskedData(lp,ck['scale']['S0'],ck['scale']['H'])
    net,cap,curve=fit(data,lp,pieces,val_anchors,ck,resume)
    evaluation=evaluate(net,data,lp,ts,pieces,cap,ck)
    report(lp,ts,info,ck['scale'],curve,cap,evaluation,time.time()-start)
    manifest.update(status='complete',trained_max_gap=cap,finished_utc=str(pd.Timestamp.now(tz='UTC')),elapsed_seconds=time.time()-start,
        metrics_sha256=ref.sha(OUT/'metrics.csv'),final_checkpoint_sha256=ref.sha(OUT/'checkpoints/epoch_30.pt'))
    dump(OUT/'manifest.json',manifest);LOG.info('COMPLETE %s',manifest)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train'],nargs='?',default='train');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    main(args.resume)
