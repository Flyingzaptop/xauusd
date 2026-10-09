"""One bounded, reproducible test of privileged OHLC reconstruction distillation."""
from __future__ import annotations
import argparse, copy, hashlib, json, logging, math, os, platform, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports' / 'angular_distillation_20261008'
LAGS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55])
HORIZONS = [1, 5, 10, 20]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG = logging.getLogger('experiment')

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(OUT/'run.log', encoding='utf-8')])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    LOG.info('device=%s torch=%s', DEVICE, torch.__version__)

def sample(a, maximum, seed):
    return np.sort(np.random.default_rng(seed).choice(a, min(maximum, len(a)), replace=False))

def eligible(ts, p):
    good = np.isfinite(p).all(1) & (p > 0).all(1)
    good &= (p[:, 1] >= p.max(1)) & (p[:, 2] <= p.min(1))
    breaks = np.r_[True, np.diff(ts) != 60] | ~good | np.r_[False, ~good[:-1]]
    segments = np.cumsum(breaks)
    origins = np.arange(74, len(p)-40)
    origins = origins[(segments[origins-74] == segments[origins+40]) & good[origins]]
    return origins, good, segments

def load_market():
    source = ROOT/'data/M1/xauusd_m1.parquet'
    d = pd.read_parquet(source)
    dt = pd.to_datetime(d.timestamp, utc=True)
    ts = dt.to_numpy(dtype='datetime64[s]').astype(np.int64)
    p = d[['open','high','low','close']].to_numpy(np.float64)
    assert np.all(np.diff(ts) > 0), 'Source timestamps must be unique and sorted'
    origins, good, segments = eligible(ts, p)
    b1,b2,b3 = [int(pd.Timestamp(x,tz='UTC').timestamp()) for x in ['2025-01-01','2026-01-01','2026-09-01']]
    train = origins[ts[origins+40] < b1]
    val = origins[(ts[origins-74] >= b1) & (ts[origins+40] < b2)]
    test = origins[(ts[origins-74] >= b2) & (ts[origins+40] < b3)]
    tail = origins[ts[origins-74] >= b3]
    splits = dict(train=sample(train,100000,101),val=sample(val,12000,102),test=test[::5],tail=tail[::5])
    audit = dict(rows=len(d), start=str(dt.min()), end=str(dt.max()), source_sha256=sha(source),
        invalid_rows=int((~good).sum()), duplicates=int(dt.duplicated().sum()),
        nonminute_transitions=int((np.diff(ts)!=60).sum()),
        eligible_counts=dict(train=len(train),val=len(val),test=len(test),tail=len(tail)),
        sample_counts={k:len(v) for k,v in splits.items()},
        exposure='Jan-Aug 2026 already exposed in earlier research per existing README; September exposure otherwise unknown.')
    logp = np.log(p)
    scales=[]
    for k in LAGS:
        ix=np.arange(k,np.searchsorted(ts,b1)); ix=ix[segments[ix]==segments[ix-k]]
        r=logp[ix,3]-logp[ix-k,3]
        scales.append(float(np.median(np.abs(r-np.median(r)))))
    H,intercept=np.polyfit(np.log(LAGS),np.log(scales),1)
    S0=float(np.exp(intercept))
    fitted=S0*LAGS**H
    scale=dict(S0=S0,H=float(H),lags=LAGS.tolist(),MAD=scales,fitted=fitted.tolist(),
        R2=float(1-np.sum((np.log(scales)-np.log(fitted))**2)/np.sum((np.log(scales)-np.mean(np.log(scales)))**2)),
        fit_end_exclusive='2025-01-01',g=1)
    dump(OUT/'data_audit.json',audit);dump(OUT/'scale.json',scale)
    np.savez_compressed(OUT/'market_split_indices.npz',**splits)
    LOG.info('AUDIT %s',json.dumps(audit));LOG.info('SCALE %s',json.dumps(scale))
    return logp,ts,splits,S0,float(H)

class Features:
    def __init__(self, logp, s0, exponent, rep):
        # Center before float32 conversion to avoid loss of small price differences.
        self.logp=torch.as_tensor(logp-logp[0,3],device=DEVICE,dtype=torch.float32)
        self.s0=s0;self.rep=rep
        self.rel=torch.arange(-19,41,device=DEVICE)
        self.lags=torch.as_tensor(LAGS,device=DEVICE)
        self.den=torch.as_tensor(s0*LAGS**exponent,device=DEVICE,dtype=torch.float32)
        self.yd=torch.as_tensor(s0*np.sqrt(np.arange(1,21)),device=DEVICE,dtype=torch.float32)
        self.hh=torch.arange(1,21,device=DEVICE)
        self.masks={}
        for mode in ['teacher','student']:
            visible=(self.rel<=0) | ((self.rel>=21) if mode=='teacher' else torch.zeros_like(self.rel,dtype=torch.bool))
            start=self.rel[:,None]-self.lags[None,:]
            start_visible=(start<=0) | ((start>=21) if mode=='teacher' else torch.zeros_like(start,dtype=torch.bool))
            ml=visible[:,None]&start_visible
            self.masks[mode]=torch.cat([visible[:,None].expand(-1,4),ml[:,:,None].expand(-1,-1,4).reshape(60,36)],1)

    def x(self, origins, mode):
        idx=torch.as_tensor(origins,device=DEVICE,dtype=torch.long)
        ix=idx[:,None]+self.rel[None,:]
        now=self.logp[ix]
        anchor=self.logp[idx,3]
        level=(now-anchor[:,None,None])/(self.s0*math.sqrt(20))
        prev=self.logp[ix[:,:,None]-self.lags[None,None,:]]
        delta=(now[:,:,None,:]-prev)/self.den[None,None,:,None]
        if self.rep=='angular':delta=torch.atan(delta)*(2/math.pi)
        f=torch.cat([level,delta.flatten(2)],2)
        mask=self.masks[mode][None,:,:].expand(len(idx),-1,-1)
        f=f.masked_fill(~mask,0)
        return torch.cat([f,mask.to(f.dtype)],2).flatten(1)

    def y(self, origins):
        idx=torch.as_tensor(origins,device=DEVICE,dtype=torch.long)
        y=(self.logp[idx[:,None]+self.hh[None,:]]-self.logp[idx,3][:,None,None])/self.yd[None,:,None]
        return y.flatten(1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4800,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),
            nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,80))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)

def seed_all(seed):
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def new_model(seed,state=None):
    seed_all(seed);m=Net().to(DEVICE)
    if state is not None:m.load_state_dict(state)
    return m

@torch.no_grad()
def predict(model, features, ix, mode='student',batch=1024):
    model.eval();out=[]
    for start in range(0,len(ix),batch):out.append(model(features.x(ix[start:start+batch],mode)).cpu().numpy())
    return np.concatenate(out).reshape(-1,20,4)

@torch.no_grad()
def validation(model,features,ix,mode):
    model.eval();se=0.;n=0
    for start in range(0,len(ix),1024):
        j=ix[start:start+1024]
        diff=(model(features.x(j,mode))-features.y(j)).reshape(-1,20,4)
        se+=diff[:,np.array(HORIZONS)-1,3].square().sum().item();n+=len(j)*4
    return se/n

def fit(features,splits,seed,name,epochs,folder,state=None,teacher=None,lam=0.,snapshot=None):
    mode='teacher' if name=='T' else 'student'
    m=new_model(seed,state);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
    best_score=float('inf');best=None;rows=[];snap=None
    t0=time.time()
    if teacher is not None:teacher.eval()
    for epoch in range(epochs):
        m.train();order=np.random.default_rng(seed*1000+epoch).permutation(splits['train'])
        loss_sum=0.;sup_sum=0.
        for start in range(0,len(order),512):
            j=order[start:start+512];x=features.x(j,mode);y=features.y(j)
            opt.zero_grad(set_to_none=True);pred=m(x);supervised=(pred-y).square().mean()
            if teacher is not None:
                with torch.no_grad():soft=teacher(features.x(j,'teacher'))
                loss=(1-lam)*supervised+lam*(pred-soft).square().mean()
            else:loss=supervised
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            loss_sum+=loss.item()*len(j);sup_sum+=supervised.item()*len(j)
        score=validation(m,features,splits['val'],mode)
        row=dict(epoch=epoch+1,objective=loss_sum/len(order),supervised_loss=sup_sum/len(order),val_score=score,seconds=time.time()-t0)
        rows.append(row)
        if score<best_score:
            best_score=score;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};best_epoch=epoch+1
        if snapshot and epoch+1==snapshot:
            snap=dict(state=copy.deepcopy(best),val_score=best_score,best_epoch=best_epoch)
            torch.save(dict(**snap,seed=seed,name='A'),folder/'A.pt')
        LOG.info('%s seed=%s %s epoch=%d/%d loss=%.5f val=%.5f best=%.5f seconds=%.1f',folder.parent.name,seed,name,epoch+1,epochs,row['supervised_loss'],score,best_score,row['seconds'])
        pd.DataFrame(rows).to_csv(folder/f'{name}_curve.csv',index=False)
    result=dict(state=best,val_score=best_score,best_epoch=best_epoch)
    torch.save(dict(**result,seed=seed,name=name),folder/f'{name}.pt')
    return result,snap

def leakage_checks(logp,s0,H):
    ix=np.array([100]);small=logp[:200].copy()
    f=Features(small,s0,H,'angular')
    teacher0=f.x(ix,'teacher').cpu().numpy();student0=f.x(ix,'student').cpu().numpy()
    changed=small.copy();changed[101:121]+=.01
    f1=Features(changed,s0,H,'angular')
    hidden_error=float(np.max(np.abs(teacher0-f1.x(ix,'teacher').cpu().numpy())))
    changed=small.copy();changed[101:141]+=.02
    f2=Features(changed,s0,H,'angular')
    future_error=float(np.max(np.abs(student0-f2.x(ix,'student').cpu().numpy())))
    teacher_future=float(np.max(np.abs(teacher0-f2.x(ix,'teacher').cpu().numpy())))
    unsafe0=(small[121,3]-small[108,3])/(s0*13**H)
    unsafe1=(f1.logp[121,3]-f1.logp[108,3]).item()/(s0*13**H)
    recovered=small[121,3]-unsafe0*(s0*13**H)
    result=dict(teacher_hidden_mutation_max_abs_diff=hidden_error,
        student_future_mutation_max_abs_diff=future_error,teacher_right_mutation_diff=teacher_future,
        unsafe_visible_lag_feature_change=abs(unsafe1-unsafe0),
        unsafe_hidden_logprice_recovery_error=float(abs(recovered-small[108,3])))
    assert hidden_error==0 and future_error==0 and teacher_future>0
    assert result['unsafe_visible_lag_feature_change']>0
    dump(OUT/'leakage_checks.json',result);LOG.info('MASK CHECKS %s',result)
    del f,f1,f2

def synthetic(kind):
    rng=np.random.default_rng(330 if kind=='random_walk' else 331);n=100000
    eps=rng.normal(0,.0002,n);r=eps.copy()
    if kind=='predictable_ar':
        for i in range(1,n):r[i]=.65*r[i-1]+eps[i]
    close=np.cumsum(r);op=np.r_[0.,close[:-1]]
    upper=np.abs(rng.normal(0,.00005,n));lower=np.abs(rng.normal(0,.00005,n))
    logp=np.column_stack([op,np.maximum(op,close)+upper,np.minimum(op,close)-lower,close])
    # Disjoint raw support, including teacher's future context.
    splits=dict(train=sample(np.arange(74,59960),12000,1),val=sample(np.arange(60074,79960),4000,2),test=np.arange(80074,99960,5))
    trainr=r[:60000];s0=float(np.median(abs(trainr-np.median(trainr))))
    ts=np.arange(n,dtype=np.int64)*60
    return logp,ts,splits,s0,.5

def train_suite(kind,logp,ts,splits,s0,H,reps,seeds,epochs):
    meta={}
    for rep in reps:
        features=Features(logp,s0,H,rep)
        for seed in seeds:
            folder=OUT/'models'/kind/rep/str(seed);folder.mkdir(parents=True,exist_ok=True)
            key=f'{rep}/{seed}';meta[key]={}
            # A and A_long share one uninterrupted trajectory.
            long,a=fit(features,splits,seed,'A_long',epochs*2,folder,snapshot=epochs)
            meta[key]['A']={k:v for k,v in a.items() if k!='state'}
            meta[key]['A_long']={k:v for k,v in long.items() if k!='state'}
            tr,_=fit(features,splits,seed,'T',epochs,folder)
            meta[key]['T']={k:v for k,v in tr.items() if k!='state'}
            b,_=fit(features,splits,seed,'B',epochs,folder,state=tr['state'])
            meta[key]['B']={k:v for k,v in b.items() if k!='state'}
            teacher=new_model(seed,tr['state']);teacher.eval()
            for name,lam in [('C25',.25),('C75',.75)]:
                c,_=fit(features,splits,seed,name,epochs,folder,teacher=teacher,lam=lam)
                meta[key][name]={k:v for k,v in c.items() if k!='state'}
            del teacher
            dump(OUT/f'{kind}_training_selection.json',meta)
        del features
    selection={}
    for rep in reps:
        scores={name:float(np.mean([meta[f'{rep}/{s}'][name]['val_score'] for s in seeds])) for name in ['C25','C75']}
        selection[rep]=dict(selected=min(scores,key=scores.get),validation_scores=scores)
    dump(OUT/f'{kind}_frozen_lambda.json',selection)
    return selection

def truth(logp,ix):return (logp[ix[:,None]+np.arange(1,21)]-logp[ix,3][:,None,None])*10000

def metric_rows(y,p,kind,split,rep,seed,method,ts,ix):
    rows=[]
    periods={'ALL':np.ones(len(ix),dtype=bool)}
    if kind=='market':
        labels=pd.to_datetime(ts[ix],unit='s',utc=True).strftime('%Y-%m').to_numpy()
        periods.update({month:labels==month for month in np.unique(labels)})
    for period,mask in periods.items():
        yy=y[mask];pp=p[mask]
        for h in HORIZONS:
            a=yy[:,h-1,3];b=pp[:,h-1,3];err=b-a;nz=a!=0
            corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>1e-12 else 0.
            mse=float(np.mean(err**2));zero=float(np.mean(a**2))
            rows.append(dict(kind=kind,split=split,representation=rep,seed=seed,method=method,period=period,
                horizon=h,n=len(a),mse_bps2=mse,mae_bps=float(np.mean(abs(err))),zero_mse_bps2=zero,
                skill_vs_zero=1-mse/zero,direction_accuracy=float(np.mean(np.sign(a[nz])==np.sign(b[nz]))),
                correlation=corr,ohlc_mse_bps2=float(np.mean((pp[:,h-1]-yy[:,h-1])**2))))
    return rows

def evaluate_suite(kind,logp,ts,splits,s0,H,reps,seeds,selection):
    rows=[];prediction_paths=[]
    for rep in reps:
        features=Features(logp,s0,H,rep)
        for split in ['test','tail']:
            if split not in splits or not len(splits[split]):continue
            ix=splits[split];y=truth(logp,ix)
            pred_dir=OUT/'predictions'/kind/rep/split;pred_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pred_dir/'truth.npz',origins=ix,timestamp=ts[ix],y_bps=y.astype(np.float32))
            for seed in seeds:
                for name in ['A','A_long','T','B','C25','C75']:
                    ck=torch.load(OUT/'models'/kind/rep/str(seed)/f'{name}.pt',map_location='cpu',weights_only=False)
                    model=new_model(seed,ck['state'])
                    pred=predict(model,features,ix,'teacher' if name=='T' else 'student')
                    pred=pred*(s0*np.sqrt(np.arange(1,21))*10000)[None,:,None]
                    rows+=metric_rows(y,pred,kind,split,rep,seed,name,ts,ix)
                    path=pred_dir/f'{name}_{seed}.npz'
                    np.savez_compressed(path,p_bps=pred.astype(np.float32))
                    prediction_paths.append(path)
                    if name==selection[rep]['selected']:
                        rows+=metric_rows(y,pred,kind,split,rep,seed,'C_selected',ts,ix)
                    violation=(pred[:,:,1]<pred.max(2)-1e-6)|(pred[:,:,2]>pred.min(2)+1e-6)
                    LOG.info('EVAL %s %s %s seed%s %s close20 MSE=%.5f OHLC violation=%.4f',kind,rep,split,seed,name,np.mean((pred[:,19,3]-y[:,19,3])**2),violation.mean())
                    dump(pred_dir/f'{name}_{seed}_diagnostics.json',dict(ohlc_violation_fraction=float(violation.mean())))
                    del model
            zero=np.zeros_like(y)
            bridge=(logp[ix+21]-logp[ix,3][:,None])[:,None,:]*np.arange(1,21)[None,:,None]/21*10000
            for name,p in [('zero',zero),('bridge_oracle',bridge)]:rows+=metric_rows(y,p,kind,split,rep,-1,name,ts,ix)
        del features
    return rows

def linear_baseline(logp,ts,splits,s0,H):
    # A convex baseline on the exact same causal features. Solve dual-free ridge
    # on the 20 left tokens (feature masks constant, omitted) with train moments.
    rows=[]
    for rep in ['log','angular']:
        f=Features(logp,s0,H,rep)
        def design(ix):
            arr=[]
            for i in range(0,len(ix),1024):
                x=f.x(ix[i:i+1024],'student').reshape(-1,60,80)[:,:20,:40].flatten(1)
                arr.append(x.cpu().numpy())
            return np.concatenate(arr).astype(np.float64)
        x=design(splits['train']);mean=x.mean(0);sd=np.maximum(x.std(0),1e-5)
        x=(x-mean)/sd;x=np.column_stack([np.ones(len(x)),x]);
        yy=truth(logp,splits['train'])[:,np.array(HORIZONS)-1,3]
        gram=x.T@x/len(x);rhs=x.T@yy/len(x)
        xv=(design(splits['val'])-mean)/sd;xv=np.column_stack([np.ones(len(xv)),xv])
        yv=truth(logp,splits['val'])[:,np.array(HORIZONS)-1,3]
        best=None;best_score=float('inf')
        for penalty in [.01,.1,1.,10.]:
            regularizer=np.eye(gram.shape[0])*penalty;regularizer[0,0]=0
            coef=np.linalg.solve(gram+regularizer,rhs)
            score=np.mean(((xv@coef-yv)/(s0*np.sqrt(HORIZONS)*10000))**2)
            if score<best_score:best_score=score;best=coef;best_penalty=penalty
        dump(OUT/f'ridge_{rep}.json',dict(penalty=best_penalty,val_score=best_score,train_only_standardization=True))
        for split in ['test','tail']:
            ix=splits[split];xx=(design(ix)-mean)/sd;xx=np.column_stack([np.ones(len(xx)),xx])
            y=truth(logp,ix);p=np.zeros_like(y);p[:,np.array(HORIZONS)-1,3]=xx@best
            rr=metric_rows(y,p,'market',split,rep,-1,'ridge',ts,ix)
            # Ridge forecasts only close at the four horizons; no OHLC claim.
            for r in rr:r['ohlc_mse_bps2']=None
            rows+=rr
        del f,x,xv
    return rows

def bootstrap_comparisons(kind,reps,seeds,selection):
    rows=[];rng=np.random.default_rng(20261008)
    for rep in reps:
        for split in ['test','tail']:
            folder=OUT/'predictions'/kind/rep/split
            if not folder.exists():continue
            truthfile=np.load(folder/'truth.npz');y=truthfile['y_bps'];ts=truthfile['timestamp']
            days,inverse=np.unique(ts//86400,return_inverse=True);counts=np.bincount(inverse)
            for candidate in ['C_selected','B']:
                actual=selection[rep]['selected'] if candidate=='C_selected' else candidate
                for base in ['A','A_long']:
                    ps=[np.load(folder/f'{actual}_{s}.npz')['p_bps'] for s in seeds]
                    pb=[np.load(folder/f'{base}_{s}.npz')['p_bps'] for s in seeds]
                    for h in [0]+HORIZONS:
                        if h==0:
                            jj=np.array(HORIZONS)-1
                            ce=np.mean([np.mean((p[:,jj,3]-y[:,jj,3])**2/np.array(HORIZONS),axis=1) for p in ps],axis=0)
                            be=np.mean([np.mean((p[:,jj,3]-y[:,jj,3])**2/np.array(HORIZONS),axis=1) for p in pb],axis=0)
                        else:
                            ce=np.mean([(p[:,h-1,3]-y[:,h-1,3])**2 for p in ps],axis=0)
                            be=np.mean([(p[:,h-1,3]-y[:,h-1,3])**2 for p in pb],axis=0)
                        delta=ce-be;daily=np.bincount(inverse,weights=delta)
                        sampled=rng.integers(0,len(days),(2000,len(days)))
                        boot=daily[sampled].sum(1)/counts[sampled].sum(1)
                        lo,hi=np.quantile(boot,[.025,.975])
                        rows.append(dict(kind=kind,representation=rep,split=split,candidate=candidate,baseline=base,
                            horizon=h,days=len(days),delta_mse_bps2=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),
                            relative_improvement_pct=float(-delta.mean()/be.mean()*100),
                            note='horizon0=primary mean SE/h over 1,5,10,20 (bps2/min); UTC day bootstrap; seed-average losses; unadjusted diagnostic CI'))
                    del ps,pb
    return rows

def write_report(metrics,comparisons,elapsed):
    m=pd.DataFrame(metrics);c=pd.DataFrame(comparisons)
    m.to_csv(OUT/'metrics.csv',index=False);c.to_csv(OUT/'paired_comparisons.csv',index=False)
    summary=m[(m.kind=='market')&(m.period=='ALL')].groupby(['split','representation','method','horizon'],as_index=False).agg(
        mse_bps2=('mse_bps2','mean'),mae_bps=('mae_bps','mean'),skill_vs_zero=('skill_vs_zero','mean'),direction_accuracy=('direction_accuracy','mean'),correlation=('correlation','mean'))
    summary.to_csv(OUT/'summary.csv',index=False)
    audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    scale=json.loads((OUT/'scale.json').read_text(encoding='utf-8'))
    selected=json.loads((OUT/'market_frozen_lambda.json').read_text(encoding='utf-8'))
    # Tables do not depend on tabulate being installed.
    def mdtable(df):
        cols=list(df.columns);out=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join(f'{v:.5g}' if isinstance(v,float) else str(v) for v in row)+' |')
        return '\n'.join(out)
    final20=summary[(summary.split=='test')&(summary.horizon==20)]
    comp20=c[(c.kind=='market')&(c.split=='test')&(c.horizon==20)]
    main=c[(c.kind=='market')&(c.split=='test')&(c.candidate=='C_selected')&(c.baseline=='A')&(c.horizon==0)]
    found=bool((main.ci_high<0).any())
    interpretation=('Есть локальные улучшения с отрицательной верхней границей диагностического интервала; проверьте горизонт, seed и сравнение с A_long. Это не независимое подтверждение.' if found else
        'Устойчивого статистически различимого преимущества дистилляции над прямым обучением в этом прогоне не обнаружено.')
    text=f'''# XAUUSD: реконструкция пропуска и teacher → student

Завершённый ограниченный исследовательский прогон, 2026-10-08. Время: {elapsed/60:.1f} мин.

{interpretation}

## Что именно проверено

Левый контекст 20 минут; скрыты следующие 20; teacher дополнительно видит 20 минут справа.
Из-за Fibonacci-lags историческая поддержка начинается за 74 минуты до точки прогноза.
Student на применении видит только прошлое. Предсказываются все 20 OHLC-свечей в
логарифмических координатах относительно последнего известного Close.

Сравнены A (прямое обучение), A_long (вдвое больше обновлений), B (веса teacher,
затем причинное дообучение), C25/C75 (дистилляция с весом 0.25/0.75), C_selected
(выбор веса только по validation), T (учитель с будущим), ridge, нулевой return и
bridge_oracle (линейная интерполяция с известной правой границей).

Log и angular используют одинаковые 9 лагов и 40 каналов плюс маски доступности.
Это проверяет arctan, но НЕ доказывает оптимальность Fibonacci-сетки.
Архитектура — небольшой MLP, не Transformer; 3 seed, 12 эпох, A_long 24.
T/B/A имеют одинаковый параметрический бюджет, но B и C требуют затрат teacher.

## Данные и границы вывода

- Исходный ряд: {audit['rows']:,} строк, {audit['start']} — {audit['end']}.
- Train до 2025; validation весь 2025; исследовательская оценка январь–август 2026.
- Сэмплы: {audit['sample_counts']}. Обучение на фиксированной выборке окон, не на каждом возможном окне.
- Все окна непрерывны по минутам и полностью находятся внутри своей части, включая правый контекст.
- Январь–август 2026 уже фигурировали в прежних исследованиях проекта. Это известный исследовательский период, НЕ нетронутый тест.
- Сентябрьский хвост короткий (1–4 сентября); история его использования неизвестна. Отдельная диагностика, не подтверждение.
- S0={scale['S0']:.8g}, H={scale['H']:.5f}, R² масштабирования={scale['R2']:.5f}; оценены только на train.
- Выбор весов дистилляции: {selected}.

## 20 минут: среднее по seed, январь–август 2026

{mdtable(final20[['representation','method','mse_bps2','mae_bps','skill_vs_zero','direction_accuracy','correlation']])}

MSE измеряется в квадрате базисных пунктов log-return; меньше лучше. Skill=1−MSE/MSE нулевого прогноза;
отрицательный skill означает проигрыш прогнозу «цена не изменится». Среднее метрик seed — не ансамбль.
Direction accuracy исключает ровно нулевые фактические returns; это не калибровка вероятностей.

## Парные сравнения на 20 минутах

{mdtable(comp20[['representation','candidate','baseline','delta_mse_bps2','ci_low','ci_high','relative_improvement_pct','days']])}

Отрицательная delta означает преимущество кандидата. Интервалы: 2000 bootstrap-повторов по UTC-дням,
сначала усредняются парные потери трёх seed. Интервалы диагностические, без поправки на множественные
сравнения; междневная зависимость остаётся ограничением. Все горизонты и месяцы — в metrics.csv.

## Контроли и ограничения

Случайное блуждание и предсказуемый AR(1) проходят тот же pipeline с меньшим бюджетом (1 seed, 6 эпох).
Отдельно проверено точное равенство входов teacher при изменении скрытого участка и student при
изменении всего будущего. Намеренно небезопасный lag-признак демонстрирует утечку исходной схемы.
Числа контролей находятся в synthetic_summary.csv, assertions — в leakage_checks.json.

Teacher получает правую границу и решает более лёгкую задачу; его качество не является качеством прогноза.
Soft labels teacher на train являются in-sample; cross-fitting не применялся.
OHLC-порядок выходов не навязывался, частота нарушений записана в prediction diagnostics.
Нет оценки P&L, spread/slippage, вероятностной калибровки или гарантии оптимальности архитектуры.
Результат относится к заданной модели, бюджету, горизонту и источнику M1. L2/MBO не использовались.

## Воспроизведение и артефакты

Из корня проекта: `python experiments/angular_distillation/run_experiment.py`

- Протокол: experiments/angular_distillation/protocol.md.
- Полные метрики: metrics.csv; агрегаты: summary.csv; интервалы: paired_comparisons.csv.
- Даты/хеш данных: data_audit.json; параметры шкалы: scale.json; воспроизводимость: manifest.json.
- Контроль масок: leakage_checks.json; checkpoint selection: *_training_selection.json и *_frozen_lambda.json.
- Кривые/веса: models/; предсказания: predictions/; подробный лог: run.log.
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')
    synth=m[(m.kind!='market')&(m.period=='ALL')].copy();synth.to_csv(OUT/'synthetic_summary.csv',index=False)
    import html
    htmltext='<!doctype html><html lang="ru"><meta charset="utf-8"><title>XAUUSD experiment</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;color:#182533;padding:0 24px}table{border-collapse:collapse;font-size:13px;width:100%;margin:24px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f7}pre{white-space:pre-wrap;line-height:1.6}h1{font-size:28px}</style>'
    htmltext+='<h1>XAUUSD: teacher → student</h1><p>'+html.escape(interpretation)+'</p>'
    htmltext+='<p>Полные данные и ограничения приведены ниже. Январь–август 2026 — ранее использованный исследовательский период.</p>'
    htmltext+='<h2>20 минут: средние результаты</h2>'+final20.to_html(index=False,float_format=lambda x:f'{x:.5g}')
    htmltext+='<h2>Парные сравнения</h2>'+comp20.to_html(index=False,float_format=lambda x:f'{x:.5g}')
    htmltext+='<h2>Все горизонты</h2>'+summary.to_html(index=False,float_format=lambda x:f'{x:.5g}')
    htmltext+='<h2>Полный отчёт</h2><pre>'+html.escape(text)+'</pre></html>'
    (OUT/'report.html').write_text(htmltext,encoding='utf-8')
    return summary

def main():
    setup();start=time.time()
    dump(OUT/'manifest.json',dict(status='running',start_utc=pd.Timestamp.now(tz='UTC'),python=sys.version,
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),
        device=DEVICE,gpu=torch.cuda.get_device_name(0) if DEVICE=='cuda' else None,
        script_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('protocol.md')),
        parameter_count=sum(p.numel() for p in Net().parameters()),seeds=[17,29,43]))
    logp,ts,splits,s0,H=load_market();leakage_checks(logp,s0,H)
    metrics=[];comparisons=[]
    for kind in ['random_walk','predictable_ar']:
        data=synthetic(kind);lp,st,sp,ss,hh=data
        sel=train_suite(kind,lp,st,sp,ss,hh,['angular'],[17],6)
        metrics+=evaluate_suite(kind,lp,st,sp,ss,hh,['angular'],[17],sel)
        comparisons+=bootstrap_comparisons(kind,['angular'],[17],sel)
        pd.DataFrame(metrics).to_csv(OUT/'control_metrics.csv',index=False)
    # Controls and all market variants are fitted before market held-out scoring.
    selection=train_suite('market',logp,ts,splits,s0,H,['log','angular'],[17,29,43],12)
    metrics+=evaluate_suite('market',logp,ts,splits,s0,H,['log','angular'],[17,29,43],selection)
    metrics+=linear_baseline(logp,ts,splits,s0,H)
    comparisons+=bootstrap_comparisons('market',['log','angular'],[17,29,43],selection)
    summary=write_report(metrics,comparisons,time.time()-start)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='complete',end_utc=str(pd.Timestamp.now(tz='UTC')),elapsed_seconds=time.time()-start,
        metrics_rows=len(metrics),comparison_rows=len(comparisons),
        report_sha256=sha(OUT/'report.md'),metrics_sha256=sha(OUT/'metrics.csv'))
    dump(OUT/'manifest.json',manifest)
    LOG.info('COMPLETE elapsed=%.1f minutes; reports=%s',(time.time()-start)/60,OUT)
    print(summary[(summary.split=='test')&(summary.horizon==20)].to_string(index=False))

if __name__=='__main__':main()
