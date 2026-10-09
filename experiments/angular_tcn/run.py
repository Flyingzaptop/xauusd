"""Angular-history causal TCN: four angles -> next OHLC candle, no distillation."""
from __future__ import annotations
import os,sys,json,hashlib,logging,time,argparse,base64,html
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/angular_tcn_20261008'
OLD=ROOT/'reports/angular_distillation_20261008'
CHANNELS=['H','O','C','L'];LAGS=np.array([1,2,3,5,8,13,21,34,55])
CONTEXT=128;SUPPORT=182;SEEDS=[17,29,43]
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LOGGER=logging.getLogger('angular_tcn')

def write_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def encode_next(current,previous,s0):
    return (2/np.pi)*np.arctan((current-previous)/s0)

def decode_log(angles,previous,s0):
    return previous+s0*np.tan(np.pi*np.clip(angles,-1+1e-6,1-1e-6)/2)

def eligible(ts,p):
    good=np.isfinite(p).all(1)&(p>0).all(1)&(p[:,0]>=p.max(1))&(p[:,3]<=p.min(1))
    cut=np.r_[True,np.diff(ts)!=60]|~good|np.r_[False,~good[:-1]]
    seg=np.cumsum(cut);ix=np.arange(SUPPORT,len(p)-1)
    ix=ix[(seg[ix-SUPPORT]==seg[ix+1])&good[ix]]
    return ix,good

def read_data():
    source=ROOT/'data/M1/xauusd_m1.parquet'
    audit=json.loads((OLD/'data_audit.json').read_text(encoding='utf-8'))
    sourcehash=sha(source);assert sourcehash==audit['source_sha256'],'Source changed; train-only scale must be refit'
    scale=json.loads((OLD/'scale.json').read_text(encoding='utf-8'))
    d=pd.read_parquet(source)
    ts=d.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
    p=d[['high','open','close','low']].to_numpy(np.float64)
    assert np.all(np.diff(ts)>0)
    ix,good=eligible(ts,p)
    b=[int(pd.Timestamp(v,tz='UTC').timestamp()) for v in ['2025-01-01','2026-01-01','2026-09-01']]
    full=dict(train=ix[ts[ix+1]<b[0]],val=ix[(ts[ix-SUPPORT]>=b[0])&(ts[ix+1]<b[1])],
        test=ix[(ts[ix-SUPPORT]>=b[1])&(ts[ix+1]<b[2])],tail=ix[ts[ix-SUPPORT]>=b[2]])
    rng=np.random.default_rng(20261008)
    splits={k:np.sort(rng.choice(v,min(n,len(v)),replace=False)) for k,v,n in [('train',full['train'],100000),('val',full['val'],12000)]}
    splits.update(test=full['test'][::5],tail=full['tail'][::5])
    info=dict(source_sha256=sourcehash,rows=len(d),start=str(d.timestamp.min()),end=str(d.timestamp.max()),
        invalid_rows=int((~good).sum()),eligible={k:len(v) for k,v in full.items()},samples={k:len(v) for k,v in splits.items()},
        exposure='2026 including September is research-exposed; no untouched-test claim')
    return np.log(p),ts,splits,scale,info

class AngularHistory:
    def __init__(self,logp,s0,H,device=DEVICE):
        self.s0=s0;self.H=H;self.device=device
        angles=np.zeros((len(logp),36),np.float32)
        for j,k in enumerate(LAGS):
            angles[k:,j*4:(j+1)*4]=(2/np.pi)*np.arctan((logp[k:]-logp[:-k])/(s0*k**H))
        self.a=torch.as_tensor(angles,device=device)
        self.offset=torch.arange(-CONTEXT+1,1,device=device)
    def x(self,ix):
        ix=torch.as_tensor(ix,device=self.device,dtype=torch.long)
        return self.a[ix[:,None]+self.offset].transpose(1,2).contiguous()
    def y(self,ix):
        ix=torch.as_tensor(ix,device=self.device,dtype=torch.long)
        return self.a[ix+1,:4]

class CausalConv(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.left=2*dilation
        self.conv=nn.Conv1d(channels,channels,3,dilation=dilation)
    def forward(self,x):return self.conv(F.pad(x,(self.left,0)))

class Block(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.c1=CausalConv(channels,dilation);self.c2=CausalConv(channels,dilation)
        self.n1=nn.LayerNorm(channels);self.n2=nn.LayerNorm(channels);self.drop=nn.Dropout(.1)
    def forward(self,x):
        z=self.drop(F.gelu(self.n1(self.c1(x).transpose(1,2)).transpose(1,2)))
        z=self.drop(F.gelu(self.n2(self.c2(z).transpose(1,2)).transpose(1,2)))
        return x+z

class AngularTCN(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Conv1d(36,32,1)
        self.blocks=nn.Sequential(*[Block(32,d) for d in [1,2,4,8,16,32]])
        self.head=nn.Conv1d(32,4,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def sequence(self,x):return (1-1e-6)*torch.tanh(self.head(self.blocks(self.projection(x))))
    def forward(self,x):return self.sequence(x)[:,:,-1]

def model(seed,state=None):
    torch.manual_seed(seed);np.random.seed(seed)
    m=AngularTCN().to(DEVICE)
    if state is not None:m.load_state_dict(state)
    return m

@torch.no_grad()
def predictions(m,data,ix):
    m.eval();pred=[]
    for start in range(0,len(ix),512):pred.append(m(data.x(ix[start:start+512])).cpu().numpy())
    return np.concatenate(pred)

@torch.no_grad()
def val_score(m,data,ix):
    m.eval();total=np.zeros(4);n=0
    for start in range(0,len(ix),512):
        j=ix[start:start+512];err=m(data.x(j))-data.y(j)
        total+=err.square().sum(0).cpu().numpy();n+=len(j)
    return total/n

def fit(data,splits,seed):
    folder=OUT/'models'/str(seed);folder.mkdir(parents=True,exist_ok=True)
    m=model(seed);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=2,min_lr=1e-5)
    best=float('inf');best_epoch=0;rows=[];started=time.time()
    for ep in range(1,25):
        order=np.random.default_rng(seed*1000+ep).permutation(splits['train'])
        m.train();loss_sum=0
        for start in range(0,len(order),256):
            ix=order[start:start+256];opt.zero_grad(set_to_none=True)
            pred=m(data.x(ix));loss=(pred-data.y(ix)).square().mean()
            assert torch.isfinite(loss),'Nonfinite training loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
            loss_sum+=loss.item()*len(ix)
        per=val_score(m,data,splits['val']);score=float(per.mean());scheduler.step(score)
        row=dict(epoch=ep,train_mse=loss_sum/len(order),val_mse=score,lr=opt.param_groups[0]['lr'],seconds=time.time()-started,
            **{f'val_{ch}':float(v) for ch,v in zip(CHANNELS,per)})
        rows.append(row)
        if score<best:
            best=score;best_epoch=ep
            torch.save(dict(state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},seed=seed,best_epoch=ep,val_mse=best,
                architecture='causal_tcn_32_6blocks_2conv',context=CONTEXT,lags=LAGS.tolist(),channels=CHANNELS,s0=data.s0,H=data.H),folder/'best.pt')
        pd.DataFrame(rows).to_csv(folder/'curve.csv',index=False)
        LOGGER.info('seed=%s epoch=%s train=%.6f val=%.6f best_epoch=%s lr=%.6g elapsed=%.1fs',seed,ep,row['train_mse'],score,best_epoch,row['lr'],row['seconds'])
        if ep>=8 and ep-best_epoch>=5:break
    return dict(seed=seed,best_epoch=best_epoch,epochs=ep,val_mse=best)

def baseline_angles(logp,ix,s0,constants):
    prior=logp[ix];close=prior[:,2:3]
    return dict(zero_angles=np.zeros((len(ix),4)),train_mean=np.tile(constants['mean_angle'],(len(ix),1)),
        repeat_angle=encode_next(logp[ix],logp[ix-1],s0),
        flat_at_close=encode_next(np.broadcast_to(close,prior.shape),prior,s0),
        train_template=encode_next(close+constants['median_offsets'][None,:],prior,s0))

def metrics(y,p,prior,actual,s0,ts,name,split):
    decoded=decode_log(p,prior,s0);errbps=(decoded-actual)*10000
    rows=[];periods={'ALL':np.ones(len(y),bool)}
    months=pd.to_datetime(ts,unit='s',utc=True).strftime('%Y-%m').to_numpy()
    periods.update({k:months==k for k in np.unique(months)})
    for period,mask in periods.items():
        yy=y[mask];pp=p[mask];err=pp-yy;eb=errbps[mask]
        for j,ch in enumerate(CHANNELS):
            v=yy[:,j];vv=pp[:,j];mse=float(np.mean(err[:,j]**2));var=float(np.var(v));nonzero=v!=0
            rows.append(dict(split=split,model=name,period=period,channel=ch,n=int(mask.sum()),angle_mse=mse,
                angle_mae_degrees=float(np.mean(abs(err[:,j]))*90),angle_rmse_degrees=float(np.sqrt(mse)*90),
                angle_r2=1-mse/var if var>0 else None,angle_sign_accuracy=float(np.mean(np.sign(v[nonzero])==np.sign(vv[nonzero]))),
                price_mse_bps2=float(np.mean(eb[:,j]**2)),price_mae_bps=float(np.mean(abs(eb[:,j]))),
                price_mae_dollars=float(np.mean(abs(np.exp(decoded[mask,j])-np.exp(actual[mask,j]))))))
        valid=(decoded[mask,0]>=np.max(decoded[mask],axis=1))&(decoded[mask,3]<=np.min(decoded[mask],axis=1))
        body=actual[mask,2]-actual[mask,1];pbody=decoded[mask,2]-decoded[mask,1];nz=body!=0
        ret=actual[mask,2]-prior[mask,2];pret=decoded[mask,2]-prior[mask,2];nr=ret!=0
        upper_actual=actual[mask,0]-np.maximum(actual[mask,1],actual[mask,2])
        upper_pred=decoded[mask,0]-np.maximum(decoded[mask,1],decoded[mask,2])
        lower_actual=np.minimum(actual[mask,1],actual[mask,2])-actual[mask,3]
        lower_pred=np.minimum(decoded[mask,1],decoded[mask,2])-decoded[mask,3]
        shape=dict(split=split,model=name,period=period,n=int(mask.sum()),invalid_ohlc_fraction=float(1-valid.mean()),
            body_mae_bps=float(np.mean(abs(pbody-body))*10000),
            range_mae_bps=float(np.mean(abs((decoded[mask,0]-decoded[mask,3])-(actual[mask,0]-actual[mask,3])))*10000),
            upper_wick_mae_bps=float(np.mean(abs(upper_pred-upper_actual))*10000),
            lower_wick_mae_bps=float(np.mean(abs(lower_pred-lower_actual))*10000),
            body_direction_accuracy=float(np.mean(np.sign(body[nz])==np.sign(pbody[nz]))),
            close_return_direction_accuracy=float(np.mean(np.sign(ret[nr])==np.sign(pret[nr]))),
            near_boundary_fraction=float(np.mean(abs(pp)>.99)))
        yield rows[-4:],shape

def bootstrap(y,preds,prior,actual,s0,ts,split):
    days,inv=np.unique(ts//86400,return_inverse=True);count=np.bincount(inv)
    rng=np.random.default_rng(1234);draw=rng.integers(0,len(days),(2000,len(days)))
    def losses(p):
        a=(p-y)**2;e=(decode_log(p,prior,s0)-actual)**2*1e8
        return dict(angle_all=a.mean(1),angle_HCL=a[:,[0,2,3]].mean(1),angle_Close=a[:,2],price_all=e.mean(1),price_Close=e[:,2])
    candidate=losses(preds['ensemble']);rows=[]
    for name,p in preds.items():
        if name.startswith('seed') or name=='ensemble':continue
        base=losses(p)
        for metric in candidate:
            delta=candidate[metric]-base[metric];daily=np.bincount(inv,weights=delta)
            boot=daily[draw].sum(1)/count[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(split=split,baseline=name,metric=metric,days=len(days),delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),
                improvement_pct=float((1-candidate[metric].mean()/base[metric].mean())*100)))
    return rows

def generate_report(logp,ts,splits,scale,meta,rows,shapes,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=pd.DataFrame(rows);sh=pd.DataFrame(shapes);ci=pd.DataFrame(comparisons)
    m.to_csv(OUT/'metrics.csv',index=False);sh.to_csv(OUT/'candle_metrics.csv',index=False);ci.to_csv(OUT/'comparisons.csv',index=False)
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,ax=plt.subplots(figsize=(9,4))
    for seed in SEEDS:
        curve=pd.read_csv(OUT/'models'/str(seed)/'curve.csv')
        ax.plot(curve.epoch,curve.val_mse,label=f'validation seed{seed}')
        ax.plot(curve.epoch,curve.train_mse,ls='--',alpha=.5,label=f'train seed{seed}')
    ax.set(xlabel='Epoch',ylabel='Angular MSE',title='Pure angle objective; best checkpoint selected on 2025');ax.legend(ncol=2)
    fig.tight_layout();fig.savefig(plots/'learning.png',dpi=160);plt.close(fig)
    bundle=np.load(OUT/'predictions/test.npz');ix=bundle['origins'];y=bundle['target'];p=bundle['ensemble']
    rng=np.random.default_rng(7);take=rng.choice(len(ix),min(6000,len(ix)),replace=False)
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for j,ax in enumerate(axes.flat):
        ax.hexbin(y[take,j]*90,p[take,j]*90,gridsize=45,mincnt=1,cmap='viridis',extent=(-90,90,-90,90))
        ax.plot([-90,90],[-90,90],'r--',lw=1);ax.set(title=CHANNELS[j],xlabel='Actual angle (degrees)',ylabel='Predicted angle (degrees)',xlim=(-90,90),ylim=(-90,90))
    fig.suptitle('Next-candle angles: held-out research evaluation');fig.tight_layout();fig.savefig(plots/'angles.png',dpi=150);plt.close(fig)
    # Deterministic quantile examples, selected independently of prediction error.
    chosen=np.unique(np.quantile(np.arange(len(ix)),[.1,.3,.5,.7,.9]).astype(int))
    fig,axes=plt.subplots(1,len(chosen),figsize=(16,4.5))
    for ax,n in zip(axes,chosen):
        origin=ix[n];anchor=logp[origin,2]
        candles=np.vstack([logp[origin],logp[origin+1],decode_log(p[n],logp[origin],scale['S0'])])
        candles=(candles-anchor)*10000
        for pos,c in enumerate(candles):
            color=['#778899','#1d8c69','#cc573e'][pos];hi,op,cl,lo=c
            ax.vlines(pos,lo,hi,color=color,lw=2);ax.add_patch(plt.Rectangle((pos-.2,min(op,cl)),.4,max(abs(cl-op),.03),facecolor=color,alpha=.65))
        ax.set_xticks([0,1,2],['Last','Actual','TCN']);ax.set_xlim(-.5,2.5)
        ax.set(title=str(pd.to_datetime(ts[origin],unit='s',utc=True))[:16],ylabel='Log-price from last Close (bps)')
    fig.suptitle('Next candle: fixed evaluation quantiles, not cherry-picked successes');fig.tight_layout();fig.savefig(plots/'candles.png',dpi=150);plt.close(fig)
    errors=np.mean((decode_log(p,logp[ix],scale['S0'])-logp[ix+1])**2,axis=1)
    worst=np.argsort(errors)[-10:][::-1]
    pd.DataFrame(dict(origin=ix[worst],timestamp=pd.to_datetime(ts[ix[worst]],unit='s',utc=True).astype(str),price_mse_bps2=errors[worst]*1e8)).to_csv(OUT/'worst_examples.csv',index=False)
    table=m[(m.split=='test')&(m.period=='ALL')&~m.model.str.startswith('seed')]
    shape=sh[(sh.split=='test')&(sh.period=='ALL')&~sh.model.str.startswith('seed')]
    compares=ci[ci.split=='test']
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((plots/name).read_bytes()).decode()+'">'
    def tbl(df):return df.to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0)
    body=f'''<h1>Углы → TCN → четыре угла → следующая свеча</h1>
<p>Модель получает только 128 × 36 угловых признаков, предсказывает H,O,C,L следующей минуты относительно последней свечи.
Последняя OHLC-свеча используется только вне сети при обратном преобразовании. Обучение MSE непосредственно по четырём углам.</p>
<p>Архитектура: причинная TCN, 32 канала, 6 residual-блоков, две свёртки kernel3 в каждом, dilations1..32, receptive field253.
Три seed; остановка/выбор весов только по validation2025. Полные обученные веса и команда прогноза сохранены.</p>
<p>Исходных свечей: {meta['rows']:,}; окна {html.escape(str(meta['samples']))}. Историческая поддержка одного примера: 183 свечи.
Train2016–2024, validation2025, исследовательская оценка январь–август2026; сентябрь отдельно. Весь 2026 уже использовался ранее.</p>
<h2>Ошибка каждого угла и восстановленной цены</h2><p>Угловая MAE дана в градусах; angle_r2 сравнивает ошибку с дисперсией истинного угла.
R² не является процентом угаданных свечей. Цена оценивается дополнительно: малая ошибка угла может сильно увеличиваться после tan.</p>{tbl(table)}
<h2>Свеча целиком</h2><p>Порядок OHLC не исправлялся сортировкой: доля некорректных свечей показана честно. Ошибки тела/теней/диапазона — в bps логарифмической цены.
Open оценивается отдельно: часто он близок к последнему Close, что не доказывает предсказуемость нового Close.</p>{tbl(shape)}
<h2>Парные сравнения ансамбля</h2><p>Отрицательная delta — меньшая ошибка TCN; положительный improvement — улучшение.
95% интервалы: 2000 bootstrap-повторов по UTC-дням, диагностические, без поправки на множественные сравнения.</p>{tbl(compares)}
<h2>Истинные и предсказанные углы</h2>{image('angles.png')}
<h2>Примеры следующей свечи</h2>{image('candles.png')}
<h2>Кривые обучения</h2>{image('learning.png')}
<h2>Что означают базовые прогнозы</h2><ul><li>zero_angles: повторить прошлую свечу целиком.</li>
<li>train_mean: средние целевые углы train, оптимальный постоянный прогноз для обучающего MSE.</li>
<li>repeat_angle: продолжить предыдущие четыре угла.</li><li>flat_at_close: все четыре цены равны последнему Close.</li>
<li>train_template: медианные train-смещения OHLC от предыдущего Close.</li></ul>
<p>Механические ценовые baselines используют известную опорную свечу; её цены не подаются в TCN.
Для zero_angles все углы нулевые: его нулевая sign_accuracy означает воздержание, а не случайную классификацию.</p>
<h2>Ограничения</h2><p>Небольшая TCN, фиксированная выборка 100000 обучающих окон, одна настройка шкалы и архитектуры.
Сравнение с иной сеткой лагов и иной архитектурой не проводилось. Никакой дистилляции. Прибыльность, комиссии и спред не оценивались.
Модель выдаёт точечную оценку, не распределение вероятностей. Результаты 2026 являются исследовательскими.</p>
<h2>Воспроизведение</h2><pre>python experiments/angular_tcn/run.py train
python experiments/angular_tcn/run.py predict</pre>
<p>predict использует последний доступный исторический бар в локальном parquet, а не текущую рыночную цену.
Артефакты: models/, predictions/, metrics.csv, candle_metrics.csv, comparisons.csv, split_indices.npz, manifest.json,
verification.json, latest_historical_prediction.json, worst_examples.csv. Угловой порядок: H,O,C,L.</p>'''
    (OUT/'results.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Angular TCN</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:32px auto;padding:0 24px;color:#1d2935}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}img{max-width:100%}pre{background:#eef2f5;padding:16px}</style>'+body+'</html>',encoding='utf-8')

def verify_outputs(logp,ts,splits,s0):
    frame=pd.read_csv(OUT/'metrics.csv');checks=0;worst=0
    for split in ['test','tail']:
        z=np.load(OUT/'predictions'/f'{split}.npz');ix=z['origins']
        assert np.array_equal(ix,splits[split])
        y=encode_next(logp[ix+1],logp[ix],s0)
        assert np.max(abs(y-z['target']))<1e-6
        for name in z.files:
            if name in ['origins','target','timestamp']:continue
            p=z[name];assert p.shape==y.shape and np.isfinite(p).all()
            for j,ch in enumerate(CHANNELS):
                calc=np.mean((p[:,j]-y[:,j])**2)
                saved=frame[(frame.split==split)&(frame.model==name)&(frame.channel==ch)&(frame.period=='ALL')].angle_mse.item()
                diff=abs(calc-saved);worst=max(worst,diff);assert diff<1e-8;checks+=1
    result=dict(status='passed',independent_prediction_metric_checks=checks,max_angular_mse_difference=worst,
        source_target_formula_rechecked=True,finite_predictions=True)
    write_json(OUT/'verification.json',result);return result

def predict_latest():
    logp,ts,_,scale,info=read_data();ix=np.array([len(logp)-1])
    assert np.all(np.diff(ts[-SUPPORT-1:])==60),'Latest history crosses a missing-minute boundary'
    data=AngularHistory(logp,scale['S0'],scale['H']);allp=[]
    for seed in SEEDS:
        ck=torch.load(OUT/'models'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
        allp.append(predictions(model(seed,ck['state']),data,ix))
    a=np.mean(allp,axis=0)[0];p=np.exp(decode_log(a,logp[-1],scale['S0']))
    result=dict(kind='historical_next_bar_forecast_not_live',last_bar_utc=str(pd.to_datetime(ts[-1],unit='s',utc=True)),
        forecast_bar_utc=str(pd.to_datetime(ts[-1]+60,unit='s',utc=True)),angles_normalized=dict(zip(CHANNELS,a.tolist())),
        angles_degrees=dict(zip(CHANNELS,(a*90).tolist())),predicted_ohlc=dict(zip(CHANNELS,p.tolist())),
        valid_ohlc=bool(p[0]>=p.max() and p[3]<=p.min()))
    write_json(OUT/'latest_historical_prediction.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

def train():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'run.log',encoding='utf-8')])
    logp,ts,splits,scale,info=read_data()
    np.savez_compressed(OUT/'split_indices.npz',**splits);write_json(OUT/'data_audit.json',info);write_json(OUT/'scale.json',scale)
    manifest=dict(status='running',started_utc=str(pd.Timestamp.now(tz='UTC')),script_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('protocol.md')),
        source_sha256=info['source_sha256'],torch=torch.__version__,numpy=np.__version__,python=sys.version,device=DEVICE,
        parameters=sum(p.numel() for p in AngularTCN().parameters()),seeds=SEEDS)
    write_json(OUT/'manifest.json',manifest);LOGGER.info('DATA %s',info);LOGGER.info('MODEL %s',manifest)
    data=AngularHistory(logp,scale['S0'],scale['H'])
    # Train-only constant baselines, fixed before held-out evaluation.
    trainix=splits['train'];constants=dict(mean_angle=encode_next(logp[trainix+1],logp[trainix],scale['S0']).mean(0),
        median_offsets=np.median(logp[trainix+1]-logp[trainix,2:3],axis=0))
    write_json(OUT/'baseline_constants.json',{k:v.tolist() for k,v in constants.items()})
    selection=[fit(data,splits,seed) for seed in SEEDS];write_json(OUT/'training_selection.json',selection)
    rows=[];shapes=[];comparisons=[]
    for split in ['test','tail']:
        ix=splits[split];y=encode_next(logp[ix+1],logp[ix],scale['S0'])
        preds={}
        for seed in SEEDS:
            ck=torch.load(OUT/'models'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
            preds[f'seed{seed}']=predictions(model(seed,ck['state']),data,ix)
        preds['ensemble']=np.mean(list(preds.values()),axis=0)
        preds.update(baseline_angles(logp,ix,scale['S0'],constants))
        (OUT/'predictions').mkdir(exist_ok=True)
        np.savez_compressed(OUT/'predictions'/f'{split}.npz',origins=ix,timestamp=ts[ix],target=y,**preds)
        for name,p in preds.items():
            for rr,ss in metrics(y,p,logp[ix],logp[ix+1],scale['S0'],ts[ix],name,split):rows+=rr;shapes.append(ss)
            LOGGER.info('EVAL %s %s angular_mse=%.6f',split,name,np.mean((p-y)**2))
        comparisons+=bootstrap(y,preds,logp[ix],logp[ix+1],scale['S0'],ts[ix],split)
    generate_report(logp,ts,splits,scale,info,rows,shapes,comparisons)
    verification=verify_outputs(logp,ts,splits,scale['S0'])
    del data
    torch.cuda.empty_cache()
    predict_latest()
    manifest.update(status='complete',elapsed_seconds=time.time()-start,finished_utc=str(pd.Timestamp.now(tz='UTC')),
        metrics_sha256=sha(OUT/'metrics.csv'),verification=verification)
    write_json(OUT/'manifest.json',manifest)
    LOGGER.info('COMPLETE %.1f minutes: %s',(time.time()-start)/60,OUT)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','predict'],default='train',nargs='?');args=parser.parse_args()
    if args.mode=='train':train()
    else:predict_latest()
