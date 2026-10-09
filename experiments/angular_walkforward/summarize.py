"""Compact delivery summary, learning diagnostics and verification links."""
import base64,json,sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,OUT,ref

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ver=json.loads((OUT/'independent_verification.json').read_text(encoding='utf-8'));assert ver['status']=='passed'
    info=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'metrics.csv');shape=pd.read_csv(OUT/'candle_metrics.csv');comparison=pd.read_csv(OUT/'comparisons.csv')
    selected=metrics[(metrics.period=='ALL')&(metrics.model=='TCN')&(metrics.channel.isin(['H','C','L']))].copy()
    baseline=metrics[(metrics.period=='ALL')&(metrics.model=='train_template')].set_index('channel')
    selected['angle_improvement_vs_template_pct']=[(1-r.angle_mse/baseline.loc[r.channel,'angle_mse'])*100 for r in selected.itertuples()]
    selected['price_improvement_vs_template_pct']=[(1-r.price_mse_bps2/baseline.loc[r.channel,'price_mse_bps2'])*100 for r in selected.itertuples()]
    simple=selected[['channel','angle_mae_degrees','angle_r2','angle_improvement_vs_template_pct','price_mae_bps','price_improvement_vs_template_pct']]
    simple.to_csv(OUT/'delivery_summary.csv',index=False)
    candle=shape[(shape.period=='ALL')&(shape.model=='TCN')].iloc[0]
    curve=pd.read_csv(OUT/'training_curve.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    monthly=metrics[(metrics.period!='ALL')&(metrics.model=='TCN')&(metrics.channel.isin(['H','C','L']))]
    for ch in ['H','C','L']:
        s=monthly[monthly.channel==ch];axes[0].plot(s.period,s.angle_r2,'o-',label=ch)
    axes[0].axhline(0,color='black',lw=.8);axes[0].set(title='Reserved last year: angular R2',ylabel='R2 (not hit rate)');axes[0].tick_params(axis='x',rotation=65);axes[0].legend()
    axes[1].plot(curve.epoch,curve.learning_rate,'o-');axes[1].set(title='Actual learning rate',xlabel='Full chronological epoch',ylabel='LR',yscale='log')
    fig.tight_layout();fig.savefig(OUT/'monthly_and_lr.png',dpi=160);plt.close(fig)
    image=base64.b64encode((OUT/'monthly_and_lr.png').read_bytes()).decode()
    predictions=np.load(OUT/'holdout_predictions.npz');origins=predictions['origins']
    target=predictions['target'];estimate=predictions['TCN']
    chosen=np.random.default_rng(17).choice(len(origins),min(10000,len(origins)),replace=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,col,label in zip(axes,[0,2,3],['High','Close','Low']):
        ax.hexbin(target[chosen,col]*90,estimate[chosen,col]*90,gridsize=45,mincnt=1,cmap='viridis',extent=(-90,90,-90,90))
        ax.plot([-90,90],[-90,90],'r--',lw=1)
        ax.set(title=label,xlabel='Actual angle (degrees)',ylabel='Predicted angle (degrees)',xlim=(-90,90),ylim=(-90,90))
    fig.tight_layout();fig.savefig(OUT/'angles.png',dpi=160);plt.close(fig)
    source=pd.read_parquet(ROOT/'data/M1/xauusd_m1.parquet');lp=np.log(source[['high','open','close','low']].to_numpy(float))
    scale=json.loads((OUT/'scale.json').read_text(encoding='utf-8'))
    examples=np.quantile(np.arange(len(origins)),[.1,.3,.5,.7,.9]).astype(int)
    fig,axes=plt.subplots(1,5,figsize=(16,4.5))
    for ax,index in zip(axes,examples):
        origin=int(origins[index]);values=np.vstack([lp[origin],lp[origin+1],ref.decode_log(estimate[index],lp[origin],scale['S0'])])
        values=(values-lp[origin,2])*10000
        for pos,(high,op,close_,low) in enumerate(values):
            color=['#778899','#20896c','#cb533a'][pos];ax.vlines(pos,low,high,color=color,lw=2)
            ax.add_patch(plt.Rectangle((pos-.2,min(op,close_)),.4,max(abs(close_-op),.025),color=color,alpha=.65))
        ax.set_xticks([0,1,2],['Last','Actual','TCN']);ax.set_xlim(-.5,2.5)
        ax.set(title=str(source.timestamp.iloc[origin])[:16],ylabel='Log-price from prior Close (bps)')
    fig.suptitle('Next candle: fixed evaluation quantiles');fig.tight_layout();fig.savefig(OUT/'candles.png',dpi=150);plt.close(fig)
    angle_image=base64.b64encode((OUT/'angles.png').read_bytes()).decode()
    candle_image=base64.b64encode((OUT/'candles.png').read_bytes()).decode()
    def table(df):return df.to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0)
    close=comparison[(comparison.baseline=='flat_at_close')&(comparison.metric=='price_Close')].iloc[0]
    close_description=('На этом периоде ценовая MSE Close ниже нулевого изменения с отрицательной верхней границей диагностического интервала.' if close.ci_high<0 else
        'Убедительного выигрыша Close по ценовой MSE над нулевым изменением в этом прогоне не получено.')
    section=f'''<!-- DELIVERY_START --><section style="padding:18px;background:#edf3f8;border-radius:10px;margin:24px 0">
<h2>Завершённый полный прогон</h2><p><b>{close_description}</b></p>
<p>Все {info['train_origins']:,} допустимых точек train пройдены в каждой из30 эпох; всего {ver['total_target_presentations']:,} предъявлений,
фактически {ver['actual_optimizer_steps']:,} обновлений AdamW. Покрытие каждого года во всех30 эпохах независимо сверено.
Сеть91 267 параметров, width64, dropout25%, начальный LR1e-4, финальный LR{curve.learning_rate.iloc[-1]:.6g}.</p>
<p>Последний год — {info['holdout_origins']:,} прогнозов без прореживания. Выбран последний checkpoint, не лучший по этому году.
Все метрики ниже рассчитаны на новом периоде оценки и с новой train-шкалой; напрямую сравнивать их с прежним пилотом без выравнивания условий нельзя.</p>
{table(simple)}
<p>Направление изменения Close: {candle.close_return_direction_accuracy*100:.2f}%; направление тела: {candle.body_direction_accuracy*100:.2f}%.
Нарушений порядка OHLC: {candle.invalid_ohlc_fraction*100:.2f}%. Open взят как предыдущий Close и не входит в обучающий loss.</p>
<p>Ошибка цены Close относительно нулевого изменения: улучшение {close.improvement_pct:.5f}%.
Разность MSE {close.delta:.6g}, диагностический95% интервал [{close.ci_low:.6g}; {close.ci_high:.6g}].</p>
<p>5 тестов пройдены; отдельно пересчитано {ver['independent_metric_checks']} метрик. Проверены хеши,
все исходные индексы, счётчик optimizer и воспроизведение предсказаний сохранённой моделью.</p>
<img src="data:image/png;base64,{image}" style="max-width:100%">
<h3>Предсказанные углы и следующая свеча</h3>
<img src="data:image/png;base64,{angle_image}" style="max-width:100%">
<img src="data:image/png;base64,{candle_image}" style="max-width:100%">
<pre>python experiments/angular_walkforward/predict.py --input data/M1/xauusd_m1.parquet</pre>
<p>Модель: checkpoints/epoch_30.pt. Прогноз: forecast.json. Возобновление: run.py train --resume.
Детальная инструкция: experiments/angular_walkforward/README.md.</p></section><!-- DELIVERY_END -->'''
    path=OUT/'results.html';doc=path.read_text(encoding='utf-8')
    if '<!-- DELIVERY_START -->' in doc:
        a=doc.index('<!-- DELIVERY_START -->');b=doc.index('<!-- DELIVERY_END -->')+len('<!-- DELIVERY_END -->');doc=doc[:a]+doc[b:]
    path.write_text(doc.replace('</h1>','</h1>'+section,1),encoding='utf-8')
    manifest['independent_verification']=ver;manifest['delivery_report_sha256']=ref.sha(path)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(simple.to_string(index=False));print(comparison.to_string(index=False));print('REPORT',path)

if __name__=='__main__':main()
