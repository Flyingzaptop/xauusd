"""Produce a self-contained comparison report and fixed candle examples."""
import sys,json,base64
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
from train import predict_model

def make_report():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));verification=json.loads((OUT/'verification.json').read_text())
    assert manifest['status'] in ['evaluation_complete','complete'] and verification['status']=='passed'
    diagnostic=json.loads((OUT/'diagnostic/results.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    metrics=pd.read_csv(OUT/'metrics.csv');shapes=pd.read_csv(OUT/'shape_metrics.csv');multiscale=pd.read_csv(OUT/'multiscale_metrics.csv');comparison=pd.read_csv(OUT/'comparisons.csv')
    primary=metrics[metrics.group=='ALL'].groupby('model')[['angle_mse','price_mse_bps2','price_mae_bps']].mean().reset_index()
    primary=primary.merge(shapes[shapes.group=='ALL'][['model','close_total_variation_ratio','body_mae_bps','range_mae_bps','invalid_fraction']],on='model')
    primary.to_csv(OUT/'summary.csv',index=False)
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for kind in ['A','B']:
        c=pd.read_csv(OUT/'diagnostic'/f'{kind}_curve.csv')
        axes[0].plot(c.step,c.angle_rmse_degrees,label=kind);axes[1].plot(c.step,c.price_mae_bps,label=kind)
    axes[0].set(xlabel='Optimizer update',ylabel='Angle RMSE (degrees)',yscale='log');axes[1].set(xlabel='Optimizer update',ylabel='HCL price MAE (bps)',yscale='log')
    for ax in axes:ax.legend()
    fig.suptitle('Eight known TRAIN windows: memorization, not validation');fig.tight_layout();fig.savefig(plots/'diagnostic.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for kind in ['A','B','C']:
        c=pd.read_csv(OUT/'models'/kind/'curve.csv')
        z=json.loads((OUT/'models'/kind/'validation_epoch_0.json').read_text())
        initial=float(np.mean([r['ratio_to_linear_bridge'] for r in z]))
        ax.plot([0]+c.epoch.tolist(),[initial]+c.validation_ratio_to_bridge.tolist(),'o-',label=kind)
    ax.axhline(1,color='black',ls='--',label='Linear interpolation')
    ax.set(xlabel='Full fine-tuning epoch',ylabel='Validation price MSE / bridge MSE',title='Validation checkpoint selection; below1 is better than interpolation')
    ax.legend();fig.tight_layout();fig.savefig(plots/'validation.png',dpi=160);plt.close(fig)
    selected=min(selection,key=lambda r:r['validation_ratio_to_bridge']);winner=selected['kind']
    lp,ts,pieces,va,_,info=old.read_data();ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    data=old.MaskedData(lp,ck['scale']['S0'],ck['scale']['H'])
    fixed=json.loads((ROOT/'reports/angular_infill_512_20261008/examples_checkpoint_12/selection.json').read_text())
    anchors=np.asarray(fixed['anchors']);lengths=np.asarray(fixed['gap_lengths'])
    candidates={}
    for kind in ['A','B','C']:
        net=build(kind,ck);saved=torch.load(OUT/'models'/kind/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(saved['model'])
        candidates[kind]=predict_model(net,data,lp,ts,anchors,lengths,kind)[1];del net
    originals=[]
    for n in lengths:
        z=np.load(ROOT/f'reports/angular_infill_512_20261008/examples_checkpoint_12/gap_{n:02d}_data.npz')
        originals.append(z['predicted'][30:30+n])
    paths=[]
    for i,(anchor,n) in enumerate(zip(anchors,lengths)):
        actual=np.exp(lp[anchor+1:anchor+n+1]);center=lp[anchor,2]
        rows=[('Факт',actual),('Прежняя модель',originals[i])]+[(f'{k}: '+{'A':'угловой loss','B':'геометрия','C':'остаток от интерполяции'}[k],np.exp(candidates[k][i,:n]+center)) for k in ['A','B','C']]
        fig,axes=plt.subplots(1,5,figsize=(19,4.5),sharey=True)
        limits=np.concatenate([p.reshape(-1) for _,p in rows]);pad=max(np.ptp(limits)*.12,.1)
        for ax,(name,p) in zip(axes,rows):
            for j,(hi,op,cl,lo) in enumerate(p):
                color='#158a75' if cl>=op else '#d94d5b';ax.vlines(j,lo,hi,color=color,lw=1)
                if abs(cl-op)<1e-9:ax.hlines(op,j-.3,j+.3,color=color,lw=1.2)
                else:ax.add_patch(plt.Rectangle((j-.3,min(op,cl)),.6,abs(cl-op),facecolor=color,edgecolor=color))
            ax.set_title(name,fontsize=10);ax.set_xlim(-1,n);ax.set_ylim(limits.min()-pad,limits.max()+pad);ax.set_xlabel('Свеча в пропуске')
        axes[0].set_ylabel('Цена, USD');fig.suptitle(f'Тот же пример: {n} свечей, {pd.to_datetime(ts[anchor+1],unit="s",utc=True):%d.%m.%Y %H:%M} UTC')
        fig.tight_layout();file=plots/f'comparison_gap_{n:02d}.png';fig.savefig(file,dpi=150);plt.close(fig);paths.append(file)
    def table(frame):return frame.to_html(index=False,float_format=lambda x:f'{x:.6g}',border=0)
    def image(path):return '<img src="data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'">'
    linear=comparison[(comparison.model==winner)&(comparison.baseline=='linear_bridge')&(comparison.metric=='price_HCL')].iloc[0]
    strong=comparison[(comparison.model==winner)&(comparison.baseline=='right_open_bridge')&(comparison.metric=='price_HCL')].iloc[0]
    if linear.ci_high<0 and strong.ci_high<0:
        verdict=f'Выбранный по validation вариант {winner} уменьшил ценовую MSE H/C/L относительно обоих интерполяционных ориентиров на финальной исследовательской оценке. Форму и Close оцениваем отдельно.'
    elif linear.ci_high<0:
        verdict=f'Вариант {winner} обошёл обычную линейную интерполяцию, но преимущество над мостом с правым Open не подтверждено.'
    else:
        verdict=f'Для выбранного варианта {winner} убедительного выигрыша по ценовой MSE H/C/L над линейной интерполяцией на этой оценке не получено.'
    fallback_note='<p>Выбрана эпоха0 варианта C: это сама линейная интерполяция. Обученная поправка не улучшила критерий validation.</p>' if winner=='C' and selected['best_epoch']==0 else ''
    body=f'''<h1>Восстановление: способность сети, loss и интерполяция</h1><p><b>{verdict}</b></p>
<h2>Диагностика</h2><p>Та же сеть на восьми известных TRAIN-окнах с обычным угловым MSE достигла RMSE угла {diagnostic['results'][0]['final']['angle_rmse_degrees']:.4f}°,
MAE цены {diagnostic['results'][0]['final']['price_mae_bps']:.5f} bps и отношения вариации Close {diagnostic['results'][0]['final']['close_total_variation_ratio']:.5f}.
Это демонстрация запоминания, не качества на новых данных. Более сложный loss не гарантировал лучшего запоминания.
При принятом Open=предыдущий предсказанный Close фактические разрывы O/C также могут оставлять ненулевую ошибку формы.</p>{image(plots/'diagnostic.png')}
<h2>Контролируемое сравнение</h2><p>A — исходный loss; B — тот же угловой выход с ошибкой декодированных цен, формы и M3/M5/M10;
C — обучаемая поправка к линейному log-price мосту между видимыми границами. Нулевая поправка равна интерполяции.
Контекст512+512, LR1e-4, dropout25%, WD0.01. Каждая модель прошла4 полные эпохи по всем {info['eligible_hidden_targets']['train']:,} допустимым скрытым целям.
Это ограниченное дообучение сохранённого epoch12, не обучение с нуля. Диагностические веса не использовались.</p>
    <p>Checkpoint каждого варианта выбран только по validation; победитель по этому критерию: {winner}, эпоха {selected['best_epoch']}.
    Оценка использует ранее исследованные данные. Критерий выбора — ценовая MSE относительно линейного моста, не визуальная резкость.</p>{fallback_note}{table(pd.DataFrame(selection))}
    {image(plots/'validation.png')}
    <h2>Все основные метрики</h2><p>MSE цены в bps²; MAE в bps. Total variation — сумма абсолютных шагов Close относительно факта.
Близость вариации к1 без совпадения движения не означает правильного восстановления.</p>{table(primary)}
<h2>Парные интервалы</h2><p>Отрицательная delta — меньшая ошибка. 95% интервалы по UTC-дням,2000 повторов, без поправки на множественные сравнения.</p>{table(comparison)}
<h2>M3/M5/M10</h2><p>Полные UTC-бары, целиком скрытые внутри пропуска. OHLC агрегированы правильно: первый Open, максимум High,
минимум Low, последний Close. Это оценка агрегатов восстановленных M1 и вспомогательная задача обучения, не отдельная модель прогноза M5.</p>{table(multiscale)}
<h2>Те же пять примеров</h2><p>Даты и длины взяты из предыдущей галереи, не отбирались по результатам нового обучения.</p>'''
    for path in paths:body+=image(path)
    body+=f'''<h2>Проверки</h2><p>6 тестов passed; независимо перепроверено {verification['independent_metric_checks']} метрик,
включая UTC-агрегацию через отдельную реализацию на pandas. Проверены полные эпохи, выбор весов и исходные индексы.</p>
<p>Случайные колебания не добавлялись. Генеративная модель распределения требует отдельной оценки и не реализована в этом сравнении.</p>
<pre>python experiments/angular_reconstruction_v2/pipeline.py</pre>
<p>Код: experiments/angular_reconstruction_v2/. Артефакты: diagnostic/, models/, predictions/, metrics.csv,
shape_metrics.csv, multiscale_metrics.csv, comparisons.csv, verification.json.</p>'''
    html='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Reconstruction objectives</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 24px}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap}img{max-width:100%;margin:20px 0}pre{background:#eef2f5;padding:15px}</style>'+body+'</html>'
    (OUT/'results.html').write_text(html,encoding='utf-8')
    manifest['report_sha256']=old.ref.sha(OUT/'results.html');manifest['verification']=verification
    manifest['status']='complete';manifest['phase']='complete'
    old.dump(OUT/'manifest.json',manifest);print(verdict);print(primary.to_string(index=False));print('REPORT',OUT/'results.html')

if __name__=='__main__':make_report()
