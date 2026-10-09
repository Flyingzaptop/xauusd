"""Five fixed held-out examples:30 actual candles on either side, truth vs infill."""
import argparse,base64,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter,MaxNLocator
import run as infill

def candle(ax,x,row,color,alpha=1.,width=.62,outline=False):
    hi,op,cl,lo=map(float,row)
    ax.vlines(x,lo,hi,color=color,linewidth=1.15,alpha=alpha,zorder=3)
    if abs(op-cl)<1e-9:
        ax.hlines(op,x-width/2,x+width/2,color=color,linewidth=1.5,alpha=alpha,zorder=4)
    else:
        ax.add_patch(Rectangle((x-width/2,min(op,cl)),width,abs(cl-op),
            facecolor='white' if outline else color,edgecolor=color,linewidth=1.,alpha=alpha,zorder=4))

def bounds(*arrays):
    values=np.concatenate([a.reshape(-1) for a in arrays]);lo=float(values.min());hi=float(values.max())
    pad=max((hi-lo)*.13,.12)
    return lo-pad,hi+pad

def panel_style(ax):
    ax.set_facecolor('#fcfcfd');ax.grid(axis='y',color='#e1e5ec',linewidth=.6,zorder=0)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:,.2f}'))
    ax.yaxis.set_major_locator(MaxNLocator(6));ax.set_ylabel('Цена, USD')

def draw(path,n,times,actual,predicted,gap_actual,gap_predicted,metrics,epoch):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    fig=plt.figure(figsize=(19,12.2),facecolor='white')
    grid=fig.add_gridspec(3,2,height_ratios=[1.1,1.1,1.2],hspace=.44,wspace=.16,
        left=.075,right=.975,top=.89,bottom=.08)
    top=fig.add_subplot(grid[0,:]);bottom=fig.add_subplot(grid[1,:],sharex=top,sharey=top)
    detail_true=fig.add_subplot(grid[2,0]);detail_pred=fig.add_subplot(grid[2,1],sharex=detail_true,sharey=detail_true)
    all_limits=bounds(actual,predicted)
    for ax,values,title in [(top,actual,'ФАКТ — настоящие свечи внутри пропуска'),
                            (bottom,predicted,'МОДЕЛЬ — восстановленные свечи внутри пропуска')]:
        panel_style(ax);ax.set_title(title,loc='left',fontweight='bold',fontsize=13)
        ax.axvspan(29.5,29.5+n,color='#f3d998',alpha=.38,zorder=0)
        ax.axvline(29.5,color='#b99030',ls='--',lw=1.1);ax.axvline(29.5+n,color='#b99030',ls='--',lw=1.1)
        for j,row in enumerate(values):
            hidden=30<=j<30+n;up=row[2]>=row[1]
            color=('#188875' if up else '#d64e5a') if hidden else '#7d8b9b'
            candle(ax,j,row,color,alpha=1 if hidden else .72,outline=(up and not hidden))
        ax.set_ylim(*all_limits);ax.set_xlim(-1,len(actual))
        ticks=list(np.unique([0,30,30+n-1,len(actual)-1]))
        minimum_spacing=max(3,int(np.ceil(len(actual)/20)))
        for candidate in np.linspace(0,len(actual)-1,8).astype(int):
            if min(abs(candidate-j) for j in ticks)>=minimum_spacing:ticks.append(int(candidate))
        ticks=np.asarray(sorted(ticks))
        ax.set_xticks(ticks,[times.iloc[j].strftime('%H:%M') for j in ticks],fontsize=10)
        ax.set_xlabel('Время UTC  ·  серые свечи — наблюдаемый контекст  ·  жёлтая область — скрытый участок',fontsize=10)
    detail_limits=bounds(gap_actual,gap_predicted)
    for ax,values,title in [(detail_true,gap_actual,'УВЕЛИЧЕНИЕ: ФАКТ'),(detail_pred,gap_predicted,'УВЕЛИЧЕНИЕ: МОДЕЛЬ')]:
        panel_style(ax);ax.set_title(title,loc='left',fontweight='bold',fontsize=13)
        for j,row in enumerate(values):
            candle(ax,j,row,'#188875' if row[2]>=row[1] else '#d64e5a',width=.58)
            invalid=row[0]<max(row) or row[3]>min(row)
            if invalid:ax.plot(j,max(row),marker='x',color='#712794',ms=9,mew=2,zorder=5)
        ax.set_ylim(*detail_limits);ax.set_xlim(-max(1.1,(4-n)/2),n-1+max(1.1,(4-n)/2))
        ticks=np.unique(np.linspace(0,n-1,min(10,n)).astype(int))
        ax.set_xticks(ticks,[str(j+1) for j in ticks]);ax.set_xlabel('Номер свечи внутри пропуска')
    t0=times.iloc[30];t1=times.iloc[30+n-1]
    interval=t0.strftime('%d.%m.%Y %H:%M')+(('–'+t1.strftime('%H:%M')) if n>1 else '')+' UTC'
    fig.suptitle(f'XAUUSD M1  |  Пропуск: {n} '+('свеча' if n==1 else 'свечей')+f'  |  {interval}',fontsize=20,fontweight='bold',y=.97)
    fig.text(.5,.928,'30 настоящих свечей слева + скрытый участок + 30 настоящих свечей справа. Парные панели имеют одинаковую шкалу.',ha='center',fontsize=12,color='#435061')
    footer=(f'Этот пример: MAE H/C/L = {metrics["mae_hcl_usd"]:.2f} USD; MAE Close = {metrics["mae_close_usd"]:.2f} USD; '
            f'нарушений OHLC = {metrics["invalid_ohlc"]}/{n}.  Checkpoint {epoch}; вход сети: 512 + 512 шагов.')
    fig.text(.5,.027,footer,ha='center',fontsize=11,color='#435061')
    fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)

def main(epoch):
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    checkpoint=infill.OUT/'checkpoints'/f'epoch_{epoch:02d}.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['trained_cap']>=40 and ck['code_sha256']==infill.ref.sha(Path(__file__).with_name('run.py'))
    audit=json.loads((infill.OUT/'data_audit.json').read_text(encoding='utf-8'))
    source=infill.ROOT/'data/M1/xauusd_m1.parquet'
    assert infill.ref.sha(source)==audit['source_sha256']
    df=pd.read_parquet(source);times=pd.to_datetime(df.timestamp,utc=True)
    raw=df[['high','open','close','low']].to_numpy(float)
    test=np.load(infill.OUT/'partition_segments.npz')['test']
    candidates=np.concatenate([np.arange(a+infill.LEFT+55-1,b-infill.RIGHT-infill.MAX_GAP) for a,b in test if b-infill.RIGHT-infill.MAX_GAP>a+infill.LEFT+55-1])
    quantiles=[.1,.3,.5,.7,.9];lengths=[1,5,10,20,40]
    anchors=candidates[np.floor((len(candidates)-1)*np.array(quantiles)).astype(int)]
    output=infill.OUT/f'examples_checkpoint_{epoch:02d}';output.mkdir(exist_ok=True)
    selection=dict(checkpoint=str(checkpoint),checkpoint_sha256=infill.ref.sha(checkpoint),
        rule='Fixed10/30/50/70/90% quantiles of final-test starts supporting40 missing candles; chosen before inference, not by model error.',
        anchors=anchors.tolist(),gap_lengths=lengths,quantiles=quantiles,source_sha256=audit['source_sha256'])
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    net=infill.Infiller().to(infill.DEVICE);net.load_state_dict(ck['model']);net.eval();records=[]
    global_data=infill.MaskedData(np.log(raw),ck['scale']['S0'],ck['scale']['H']);max_input_difference=0.
    for anchor,n,q in zip(anchors,lengths,quantiles):
        anchor=int(anchor);start=anchor-(infill.LEFT+54);end=anchor+n+infill.RIGHT+1
        window=raw[start:end].copy();local_anchor=infill.LEFT+54
        # Actual hidden candles are erased before computing ANY model feature.
        window[local_anchor+1:local_anchor+n+1]=window[local_anchor]
        lp=np.log(window);data=infill.MaskedData(lp,ck['scale']['S0'],ck['scale']['H'])
        local_input=data.x([local_anchor],[n]);global_input=global_data.x([anchor],[n])
        difference=float((local_input-global_input).abs().max().item());max_input_difference=max(max_input_difference,difference)
        assert torch.allclose(local_input,global_input,atol=1e-7,rtol=1e-7),'Local hidden-data removal changed an admissible input feature'
        p=infill.predict(net,data,np.array([local_anchor]),np.array([n]))[:,:n]
        gap_predicted=np.exp(infill.decode(p,lp[local_anchor:local_anchor+1],ck['scale']['S0'])[0])
        gap_actual=raw[anchor+1:anchor+n+1]
        display_actual=raw[anchor-29:anchor+n+31].copy();display_prediction=display_actual.copy()
        display_prediction[30:30+n]=gap_predicted
        display_times=times.iloc[anchor-29:anchor+n+31].reset_index(drop=True)
        assert len(display_times)==60+n
        assert np.array_equal(display_actual[:30],display_prediction[:30])
        assert np.array_equal(display_actual[30+n:],display_prediction[30+n:])
        assert np.all(np.diff(display_times.to_numpy(dtype='datetime64[s]').astype(np.int64))==60)
        assert np.isfinite(gap_predicted).all()
        invalid=(gap_predicted[:,0]<gap_predicted.max(1))|(gap_predicted[:,3]>gap_predicted.min(1))
        record=dict(gap=n,anchor=anchor,quantile=q,first_hidden_utc=str(times.iloc[anchor+1]),last_hidden_utc=str(times.iloc[anchor+n]),
            mae_hcl_usd=float(np.mean(abs(gap_predicted[:,infill.COLS]-gap_actual[:,infill.COLS]))),
            mae_close_usd=float(np.mean(abs(gap_predicted[:,2]-gap_actual[:,2]))),
            invalid_ohlc=int(invalid.sum()),display_left=30,display_right=30,model_left=512,model_right=512)
        file=output/f'gap_{n:02d}.png';draw(file,n,display_times,display_actual,display_prediction,gap_actual,gap_predicted,record,epoch)
        record['image']=str(file);records.append(record)
        np.savez_compressed(output/f'gap_{n:02d}_data.npz',timestamps=display_times.to_numpy(dtype='datetime64[s]').astype(np.int64),
            actual=display_actual,predicted=display_prediction,angles=p[0],hidden_start_index=30,hidden_length=n)
        print(json.dumps(record,ensure_ascii=False))
    (output/'examples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(records).to_csv(output/'examples.csv',index=False)
    gallery='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пять примеров восстановления</title><style>body{font:17px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}img{width:100%}figure{margin:35px 0}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>Пять примеров восстановления свечей</h1>'
    gallery+=f'<p>Checkpoint {epoch}, этап1–40. На изображениях30 свечей слева и30 справа; модель получает512+512. Внутри пропуска цены восстановлены только из предсказанных углов и последней видимой свечи.</p>'
    gallery+='<p>Показаны фиксированные10/30/50/70/90-процентные позиции допустимых стартов отложенного периода. Это пять иллюстраций, не агрегированная оценка качества. Цена на осях — USD.</p>'
    for record in records:
        encoded=base64.b64encode(Path(record['image']).read_bytes()).decode()
        gallery+=f'<figure><h2>Пропуск {record["gap"]}</h2><img src="data:image/png;base64,{encoded}" alt="Пропуск {record["gap"]} свечей"><figcaption>{record["first_hidden_utc"]}; MAE Close ${record["mae_close_usd"]:.2f}</figcaption></figure>'
    (output/'index.html').write_text(gallery+'</html>',encoding='utf-8')
    infill.dump(output/'verification.json',dict(status='passed',images=5,display_context_each_side=30,checkpoint_epoch=epoch,
        source_and_model_hashes_verified=True,hidden_data_erased_before_feature_generation=True,
        actual_context_identical_in_both_panels=True,no_ohlc_projection=True,example_selection_precedes_inference=True,
        blinded_local_inputs_match_original_global_masked_inputs=True,max_input_difference=max_input_difference,
        renderer_sha256=infill.ref.sha(Path(__file__))))
    print('GALLERY',output/'index.html')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=int,default=12);args=parser.parse_args();main(args.epoch)
