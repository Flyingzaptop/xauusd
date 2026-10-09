"""Full chronological training of a dense causal angular TCN; all eligible origins."""
from __future__ import annotations
import os,sys,time,json,hashlib,logging,argparse,importlib.util,base64
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/angular_walkforward_20261008'
spec=importlib.util.spec_from_file_location('angular_reference',ROOT/'experiments/angular_tcn/run.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
CONTEXT=128;CHUNK=2048;EPOCHS=30;TARGET=[0,2,3];NAMES=['H','C','L'];LOG=logging.getLogger('full_walk')

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8');temp.replace(path)

def chunks_for(origins):
    cuts=np.r_[0,np.flatnonzero(np.diff(origins)!=1)+1,len(origins)];chunks=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        for pos in range(int(lo),int(hi),CHUNK):chunks.append((int(origins[pos]),min(CHUNK,int(hi)-pos)))
    return chunks

class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.dilation=dilation
        self.conv=nn.Conv1d(width,width,2,dilation=dilation);self.point=nn.Conv1d(width,width,1)
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.drop=nn.Dropout(.25)
    def forward(self,x):
        z=self.conv(F.pad(x,(self.dilation,0)))
        z=self.drop(F.gelu(self.norm1(z.transpose(1,2)).transpose(1,2)))
        z=self.drop(F.gelu(self.norm2(self.point(z).transpose(1,2)).transpose(1,2)))
        return x+z

class TCN(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Conv1d(36,64,1)
        self.blocks=nn.Sequential(*[Block(64,d) for d in [1,2,4,8,16,32,64]])
        self.head=nn.Conv1d(64,3,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):return (1-1e-6)*torch.tanh(self.head(self.blocks(self.input(x))))

def seed_all():
    torch.manual_seed(17);np.random.seed(17)

def load_data():
    source=ROOT/'data/M1/xauusd_m1.parquet';df=pd.read_parquet(source)
    ts=df.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
    p=df[['high','open','close','low']].to_numpy(float);assert np.all(np.diff(ts)>0)
    origins,good=ref.eligible(ts,p);assert good.all(),'Invalid source candles require an explicit audit'
    boundary=pd.Timestamp(df.timestamp.max())-pd.DateOffset(years=1);cut=int(boundary.timestamp())
    train=origins[ts[origins+1]<cut];test=origins[ts[origins+1]>=cut]
    assert len(train)>0 and len(test)>0 and ts[train[-1]+1]<ts[test[0]+1]
    lp=np.log(p);segments=np.cumsum(np.r_[True,np.diff(ts)!=60]);scales=[]
    for k in ref.LAGS:
        ix=np.arange(k,np.searchsorted(ts,cut));ix=ix[segments[ix]==segments[ix-k]]
        r=lp[ix,2]-lp[ix-k,2];scales.append(float(np.median(abs(r-np.median(r)))))
    H,intercept=np.polyfit(np.log(ref.LAGS),np.log(scales),1);s0=float(np.exp(intercept))
    scale=dict(S0=s0,H=float(H),lags=ref.LAGS.tolist(),MAD=scales,g=1,fit_end_exclusive=str(boundary))
    train_chunks=chunks_for(train);test_chunks=chunks_for(test)
    for ix,chunks in [(train,train_chunks),(test,test_chunks)]:
        expanded=np.concatenate([np.arange(start,start+n) for start,n in chunks])
        assert np.array_equal(expanded,ix),'Chunking must cover every origin exactly once, in order'
    years=pd.to_datetime(ts,unit='s',utc=True).year.to_numpy()
    info=dict(source_sha256=ref.sha(source),rows=len(ts),source_start=str(df.timestamp.min()),source_end=str(df.timestamp.max()),
        holdout_start=str(boundary),train_origins=len(train),holdout_origins=len(test),train_chunks=len(train_chunks),
        train_first_target=str(df.timestamp.iloc[train[0]+1]),train_last_target=str(df.timestamp.iloc[train[-1]+1]),
        holdout_first_target=str(df.timestamp.iloc[test[0]+1]),holdout_last_target=str(df.timestamp.iloc[test[-1]+1]),
        eligible_total=len(origins),excluded_for_gaps_or_warmup=int(len(ts)-1-len(origins)),
        train_targets_by_year={str(y):int(np.sum(years[train+1]==y)) for y in np.unique(years[train+1])},
        exact_coverage_verified=True,origin_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
        context_tokens=128,raw_history_bars=183,shuffle=False,sampling=False,holdout_exposure='Previously researched data; reserved only for this run')
    return lp,ts,years,train,test,train_chunks,test_chunks,scale,info

def xy(data,start,n):
    x=data.a[start-127:start+n].T.unsqueeze(0).contiguous()
    y=data.a[start+1:start+n+1][:,TARGET]
    return x,y

def setup_optimizer(net):
    opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,threshold=.001,min_lr=1e-6)
    return opt,sched

def throughput(data,chunks):
    seed_all();net=TCN().to(DEVICE);opt,_=setup_optimizer(net);start=time.time();count=0
    for begin,n in chunks[:40]:
        x,y=xy(data,begin,n);opt.zero_grad(set_to_none=True)
        loss=(net(x)[0,:,-n:].T-y).square().mean();loss.backward();opt.step();count+=n
    if DEVICE=='cuda':torch.cuda.synchronize()
    result=dict(benchmark_chunks=min(40,len(chunks)),targets=count,seconds=time.time()-start,discarded_model=True)
    result['targets_per_second']=count/result['seconds'];dump(OUT/'throughput.json',result)
    LOG.info('BENCHMARK %s',result);del net,opt
    return result

def save_checkpoint(net,opt,sched,epoch,scale,curve):
    state=dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,
        scale=scale,curve=curve,seed=17,targets=NAMES,context=128,width=64,dropout=.25,
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if DEVICE=='cuda' else [],
        source_code_sha256=ref.sha(Path(__file__)))
    path=OUT/'checkpoints'/f'epoch_{epoch:02d}.pt';path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
    dump(OUT/'resume.json',dict(checkpoint=str(path),completed_epoch=epoch))

def fit(data,chunks,train,years,scale,resume):
    seed_all();net=TCN().to(DEVICE);opt,sched=setup_optimizer(net);curve=[];first=1
    if resume and (OUT/'resume.json').exists():
        path=json.loads((OUT/'resume.json').read_text())['checkpoint'];ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['source_code_sha256']==ref.sha(Path(__file__)) and ck['scale']==scale
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['cpu_rng']);
        if DEVICE=='cuda':torch.cuda.set_rng_state_all(ck['cuda_rng'])
        first=ck['epoch']+1;curve=ck['curve'];LOG.info('RESUME epoch%s',first)
    for epoch in range(first,EPOCHS+1):
        net.train();started=time.time();last_log=started;total=np.zeros(3);seen=0;year_loss={};lr=opt.param_groups[0]['lr']
        for step,(begin,n) in enumerate(chunks,1):
            x,y=xy(data,begin,n);opt.zero_grad(set_to_none=True)
            pred=net(x)[0,:,-n:].T;errors=(pred-y).square();loss=errors.mean()
            assert torch.isfinite(loss),'Nonfinite training loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            sums=errors.detach().sum(0).cpu().numpy().astype(float);total+=sums;seen+=n
            target_years=years[begin+1:begin+n+1]
            if target_years[0]==target_years[-1]:
                yy=int(target_years[0]);rec=year_loss.setdefault(yy,[0,np.zeros(3)]);rec[0]+=n;rec[1]+=sums
            else:
                err=errors.detach().cpu().numpy()
                for yy in np.unique(target_years):
                    mask=target_years==yy;rec=year_loss.setdefault(int(yy),[0,np.zeros(3)]);rec[0]+=int(mask.sum());rec[1]+=err[mask].sum(0)
            if time.time()-last_log>=20 or step==len(chunks):
                progress=dict(epoch=epoch,epochs=EPOCHS,targets_seen=seen,targets_per_epoch=len(train),chunks_done=step,chunks_per_epoch=len(chunks),
                    last_training_year=int(target_years[-1]),angular_mse=float(total.mean()/seen),lr=lr,epoch_seconds=time.time()-started)
                dump(OUT/'progress.json',progress);LOG.info('PROGRESS %s',progress);last_log=time.time()
        assert seen==len(train)
        value=float(total.mean()/seen);sched.step(value)
        row=dict(epoch=epoch,targets=seen,updates=len(chunks),angular_mse=value,loss_H=total[0]/seen,loss_C=total[1]/seen,loss_L=total[2]/seen,
            learning_rate=lr,next_learning_rate=opt.param_groups[0]['lr'],seconds=time.time()-started)
        curve.append(row);pd.DataFrame(curve).to_csv(OUT/'training_curve.csv',index=False)
        pd.DataFrame([dict(year=y,targets=v[0],H=v[1][0]/v[0],C=v[1][1]/v[0],L=v[1][2]/v[0]) for y,v in sorted(year_loss.items())]).to_csv(OUT/f'train_years_epoch_{epoch:02d}.csv',index=False)
        save_checkpoint(net,opt,sched,epoch,scale,curve)
        LOG.info('EPOCH COMPLETE %s',row)
    net.eval();return net,curve

@torch.no_grad()
def infer(net,data,chunks):
    net.eval();outputs=[]
    for start,n in chunks:
        x,_=xy(data,start,n);outputs.append(net(x)[0,:,-n:].T.cpu().numpy())
    return np.concatenate(outputs)

def score(net,data,lp,ts,train,test,test_chunks,scale):
    s0=scale['S0'];target=ref.encode_next(lp[test+1],lp[test],s0)
    three=infer(net,data,test_chunks);four=np.empty_like(target);four[:,TARGET]=three
    # Open is supplied deterministically, outside the neural network.
    four[:,1]=ref.encode_next(lp[test,2],lp[test,1],s0)
    constants=dict(mean_angle=ref.encode_next(lp[train+1],lp[train],s0).mean(0),median_offsets=np.median(lp[train+1]-lp[train,2:3],axis=0))
    baselines=ref.baseline_angles(lp,test,s0,constants);baselines.pop('repeat_angle')
    for values in baselines.values():values[:,1]=four[:,1]
    preds=dict(TCN=four,**baselines);rows=[];shapes=[];comparisons=[]
    for name,p in preds.items():
        for rr,ss in ref.metrics(target,p,lp[test],lp[test+1],s0,ts[test+1],name,'last_year'):rows+=rr;shapes.append(ss)
    np.savez_compressed(OUT/'holdout_predictions.npz',origins=test,timestamp=ts[test+1],target=target,**preds)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(shapes).to_csv(OUT/'candle_metrics.csv',index=False)
    days,inv=np.unique(ts[test+1]//86400,return_inverse=True);counts=np.bincount(inv);rng=np.random.default_rng(1717)
    sampled=rng.integers(0,len(days),(2000,len(days)))
    def loss(p):
        a=(p[:,TARGET]-target[:,TARGET])**2
        e=(ref.decode_log(p,lp[test],s0)-lp[test+1])**2*1e8
        return dict(angle_HCL=a.mean(1),angle_Close=a[:,1],price_HCL=e[:,TARGET].mean(1),price_Close=e[:,2])
    ours=loss(four)
    for name,p in baselines.items():
        b=loss(p)
        for key in ours:
            difference=ours[key]-b[key];sums=np.bincount(inv,weights=difference)
            boot=sums[sampled].sum(1)/counts[sampled].sum(1);lo,hi=np.quantile(boot,[.025,.975])
            comparisons.append(dict(baseline=name,metric=key,delta=float(difference.mean()),ci_low=lo,ci_high=hi,days=len(days),
                improvement_pct=float((1-ours[key].mean()/b[key].mean())*100)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
    # Recompute scores independently in float64 from saved predictions.
    checks=0
    for name,p in preds.items():
        for j,ch in enumerate(ref.CHANNELS):
            reported=[r for r in rows if r['model']==name and r['period']=='ALL' and r['channel']==ch][0]
            actual=float(np.mean((p[:,j].astype(float)-target[:,j])**2))
            assert abs(reported['angle_mse']-actual)<1e-9;checks+=1
    dump(OUT/'verification.json',dict(status='passed',all_train_origins_once_per_epoch=True,epochs=EPOCHS,
        metric_checks=checks,holdout_origins_scored=len(test),open_not_trained=True,holdout_used_for_training_or_selection=False))
    return pd.DataFrame(rows),pd.DataFrame(shapes),pd.DataFrame(comparisons)

def report(info,scale,curve,metrics,shapes,comparisons,elapsed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for ch in NAMES:ax.plot([r['epoch'] for r in curve],[r[f'loss_{ch}'] for r in curve],label=ch)
    ax.set(xlabel='Complete chronological epoch',ylabel='Training angular MSE',title='Every eligible origin visited once per epoch; training diagnostics only');ax.legend();fig.tight_layout()
    fig.savefig(OUT/'training.png',dpi=160);plt.close(fig)
    img=base64.b64encode((OUT/'training.png').read_bytes()).decode()
    summary=metrics[(metrics.period=='ALL')&(metrics.model=='TCN')];summary.to_csv(OUT/'summary.csv',index=False)
    def table(d):return d.to_html(index=False,float_format=lambda v:f'{v:.6g}',border=0)
    text=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Full chronological angular TCN</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:36px auto;padding:0 24px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}}img{{max-width:100%}}</style>
<h1>TCN: все точки, хронологически, 30 полных эпох</h1>
<p>Обучено {info['train_origins']:,} точек за каждую эпоху, всего {info['train_origins']*EPOCHS:,} предъявлений.
Без выборки и перемешивания. Период целей: {info['train_first_target']} — {info['train_last_target']}.
Последний год: {info['holdout_first_target']} — {info['holdout_last_target']}, {info['holdout_origins']:,} оценённых точек без прореживания.</p>
<p>Причинная TCN width64, контекст128 угловых шагов (183 исходных свечи), 36 входных каналов, выход3 угла H/C/L.
Open=последний Close вне сети. Dropout25%, AdamW LR1e-4, weight_decay0.01, gradient clip1.
LR может только уменьшаться по тренировочному плато. Оценён checkpoint последней,30-й эпохи.</p>
<p>Каждая эпоха — проход по ВСЕМ допустимым точкам; исключены только разрывы/недостаточная история.
Свёртки вычисляются по последовательным блокам, прогноз есть у каждой минуты. Обновление весов происходит после блока, не после каждой свечи.
Повторные эпохи используют уже виденные исторические ответы: тренировочные потери не являются независимым walk-forward-бэктестом.</p>
<p>Последний год не участвовал в подборе настроек, весов или числа эпох. Но этот рыночный период исследовался ранее, поэтому он не является глобально нетронутым.
Масштаб рассчитан только на train: S0={scale['S0']:.9g}, H={scale['H']:.6f}. Время текущего запуска {elapsed/60:.1f} минут.</p>
<h2>Следующая свеча: результаты модели</h2>{table(summary)}
<h2>Все базовые сравнения</h2>{table(metrics[metrics.period=='ALL'])}
<h2>Форма свечи</h2>{table(shapes[shapes.period=='ALL'])}
<h2>Парные сравнения H/C/L и Close</h2><p>Отрицательная delta — меньше ошибка модели. 95% интервалы по UTC-дням,2000 bootstrap-повторов, без поправки на множественные проверки.</p>{table(comparisons)}
<h2>Ход обучения</h2><img src="data:image/png;base64,{img}">{table(pd.DataFrame(curve))}
<p>Числа в degrees относятся к выбранной нормированной логарифмической шкале. Ценовые ошибки измеряются отдельно.
Невалидные OHLC не исправлялись. Комиссии, спред и прибыльность не оценивались.</p>
<h2>Артефакты</h2><p>checkpoints/epoch_30.pt — модель и optimizer; training_curve.csv — точное число примеров и LR каждой эпохи;
train_years_epoch_*.csv — потери по годам; split_indices.npz — все точки; holdout_predictions.npz — все прогнозы;
data_audit.json,manifest.json,verification.json — аудит и проверки.</p>
<pre>python experiments/angular_walkforward/run.py train
python experiments/angular_walkforward/run.py train --resume</pre></html>'''
    (OUT/'results.html').write_text(text,encoding='utf-8')

def main(resume=False):
    OUT.mkdir(parents=True,exist_ok=True);started=time.time()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'run.log',encoding='utf-8')])
    lp,ts,years,train,test,chunks,test_chunks,scale,info=load_data()
    dump(OUT/'data_audit.json',info);dump(OUT/'scale.json',scale);np.savez_compressed(OUT/'split_indices.npz',train=train,holdout=test)
    manifest=dict(status='running',started_utc=str(pd.Timestamp.now(tz='UTC')),source_sha256=info['source_sha256'],script_sha256=ref.sha(Path(__file__)),
        protocol_sha256=ref.sha(Path(__file__).with_name('protocol.md')),epochs=EPOCHS,parameters=sum(p.numel() for p in TCN().parameters()),
        initial_lr=1e-4,dropout=.25,weight_decay=.01,train_origins=info['train_origins'],targets=NAMES,torch=torch.__version__,device=DEVICE)
    dump(OUT/'manifest.json',manifest);LOG.info('AUDIT %s',info);LOG.info('MODEL %s',manifest)
    data=ref.AngularHistory(lp,scale['S0'],scale['H'])
    if not resume:
        bench=throughput(data,chunks)
        LOG.info('Estimated full epoch from initial benchmark: %.1f seconds; actual timings may vary',len(train)/bench['targets_per_second'])
    net,curve=fit(data,chunks,train,years,scale,resume)
    metrics,shapes,comparisons=score(net,data,lp,ts,train,test,test_chunks,scale)
    report(info,scale,curve,metrics,shapes,comparisons,time.time()-started)
    manifest.update(status='complete',finished_utc=str(pd.Timestamp.now(tz='UTC')),elapsed_seconds=time.time()-started,
        total_training_presentations=sum(r['targets'] for r in curve),optimizer_updates=sum(r['updates'] for r in curve),
        final_checkpoint_sha256=ref.sha(OUT/'checkpoints/epoch_30.pt'),metrics_sha256=ref.sha(OUT/'metrics.csv'))
    dump(OUT/'manifest.json',manifest);LOG.info('COMPLETE %s',manifest)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train'],nargs='?',default='train');p.add_argument('--resume',action='store_true');args=p.parse_args()
    main(args.resume)
