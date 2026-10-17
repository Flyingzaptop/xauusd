"""Independent CPU verification, forecast diagnostics, and report generation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];Q=np.array([.1,.5,.9])
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def pinball(p,y):
    e=y[...,None]-p;return np.maximum(Q*e,(Q-1)*e)
def metrics(p,y,scale):
    loss=pinball(p,y);err=np.abs(p[...,1]-y)*scale[:,None,None];covered=(y>=p[...,0])&(y<=p[...,2]);width=(p[...,2]-p[...,0])*scale[:,None,None]
    return dict(pinball_normalized=float(loss.mean()),pinball_dollars=float((loss*scale[:,None,None,None]).mean()),mean_price_mae_dollars=float(err[:,:,0].mean()),first60_mean_price_mae_dollars=float(err[:,:4,0].mean()),last_block_mean_price_mae_dollars=float(err[:,-1,0].mean()),mean_price_mae_by_block=err[:,:,0].mean(0).tolist(),mean_price_80pct_coverage=float(covered[:,:,0].mean()),all_targets_80pct_coverage=float(covered.mean()),mean_price_interval_width_dollars=float(width[:,:,0].mean()),extrema_mae_dollars=float(err[:,:,1:].mean()))
def paired_day_ci(a,b,days):
    unique=np.unique(days);sums=np.array([(a[days==d]-b[days==d]).sum() for d in unique]);counts=np.array([(days==d).sum() for d in unique]);rng=np.random.default_rng(177);ix=rng.integers(0,len(unique),(10000,len(unique)));boot=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean_difference=float(np.mean(a-b)),interval95=np.quantile(boot,[.025,.975]).tolist(),days=len(unique),method='paired bootstrap of whole calendar days, 10000 resamples; descriptive with few days')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reports/multiscale_curriculum_20261010');args=ap.parse_args();out=args.out;r=out/'remote';run=read(r/'run.json');z=np.load(out/'dataset.npz');meta=read(out/'dataset.json')
    assert run['status']=='complete' and run['selection_frozen_before_test'];assert read(r/'worker_result.json')['returncode']==0
    assert run['data_sha256']==hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()
    for name in ['train.py','model.py','data.py','objective.py']:assert run['source_sha256'][name]==hashlib.sha256((ROOT/'experiments/multiscale_curriculum'/name).read_bytes()).hexdigest()
    assert len(run['trials'])==8;curves={}
    for name,t in run['trials'].items():
        rows=read(r/f'{name}_curve.json');curves[name]=rows;assert len(rows)==t['epochs'];assert all(row['examples_seen']==len(z['train_origins']) for row in rows)
        for stage in t['stages']:
            rr=[v for v in rows if v['stage']==stage['stage']];assert len(rr)==stage['epochs']
            best=min(rr,key=lambda v:v['validation']['primary']);assert best['epoch']==stage['best_global_epoch'];assert np.isclose(best['validation']['primary'],stage['best_validation'])
        v=np.load(r/f'{name}_validation.npz');assert np.allclose(pinball(v['prediction'],z['val_final']).mean(),t['best_validation'],rtol=1e-5)
    for seed in [17,29]:
        shas={run['trials'][f'{k}_{m}_{seed}']['initial_trunk_sha256'] for k in ['static','hyper'] for m in ['curriculum','direct']};assert len(shas)==1
        for kind in ['static','hyper']:assert run['trials'][f'{kind}_direct_{seed}']['epoch_cap']==run['trials'][f'{kind}_curriculum_{seed}']['epochs']
    predictions={};results={}
    for family,v in run['families'].items():
        a=np.load(r/f'{family}_test.npz');p=a['prediction'];y=z['test_final'];assert np.array_equal(a['target'],y)
        assert np.isfinite(p).all() and np.all(p[...,0]<=p[...,1]+1e-6) and np.all(p[...,1]<=p[...,2]+1e-6)
        assert np.all(p[:,:,1,:]<=p[:,:,0,:]+1e-6) and np.all(p[:,:,0,:]<=p[:,:,2,:]+1e-6)
        members=[np.load(r/f'{family}_{seed}_test.npz')['prediction'] for seed in [17,29]];assert np.array_equal(np.mean(members,axis=0),p)
        assert np.isclose(pinball(p,y).mean(),v['test_pinball'],rtol=1e-6)
        if not v['diagnostic']:
            vp=np.load(r/f'{family}_validation.npz')['prediction'];vv=np.mean([np.load(r/f'{family}_{seed}_validation.npz')['prediction'] for seed in [17,29]],axis=0);assert np.array_equal(vp,vv)
            assert np.isclose(pinball(vp,z['val_final']).mean(),v['validation_pinball'],rtol=1e-6)
        predictions[family]=p;results[family]=dict(validation_pinball=v['validation_pinball'],**metrics(p,y,z['test_scale']))
    selected=min([k for k,v in run['families'].items() if not v['diagnostic']],key=lambda k:run['families'][k]['validation_pinball']);assert selected==run['selected_family']
    template=np.moveaxis(np.quantile(z['train_final'],Q,axis=0),0,-1).astype('float32');baseline=np.broadcast_to(template,z['test_final'].shape+(3,));saved=np.load(r/'baseline_test.npz')['prediction'];assert np.array_equal(saved,baseline)
    results['train_template']=dict(validation_pinball=float(pinball(template,z['val_final']).mean()),**metrics(baseline,z['test_final'],z['test_scale']));predictions['train_template']=baseline
    persistence=np.abs(z['test_final'][:,:,0])*z['test_scale'][:,None]
    results['persistence']=dict(mean_price_mae_dollars=float(persistence.mean()),first60_mean_price_mae_dollars=float(persistence[:,:4].mean()),last_block_mean_price_mae_dollars=float(persistence[:,-1].mean()),mean_price_mae_by_block=persistence.mean(0).tolist())
    errors={k:(np.abs(p[:,:,0,1]-z['test_final'][:,:,0])*z['test_scale'][:,None]).mean(1) for k,p in predictions.items()};days=(int(z['start_minute'])+z['test_origins'])//1440
    comparisons={f'{selected}_vs_template':paired_day_ci(errors[selected],errors['train_template'],days),f'{selected}_vs_persistence':paired_day_ci(errors[selected],persistence.mean(1),days)}
    for kind in ['static','hyper']:comparisons[f'{kind}_curriculum_minus_direct']=paired_day_ci(errors[kind+'_curriculum'],errors[kind+'_direct'],days)
    probabilistic_comparison=paired_day_ci(pinball(predictions[selected],z['test_final']).mean((1,2,3)),pinball(baseline,z['test_final']).mean((1,2,3)),days)
    verification=dict(status='passed',trials=8,total_epochs=sum(t['epochs'] for t in run['trials'].values()),full_train_examples_per_epoch=len(z['train_origins']),paired_initial_trunks=True,stage_selections_recomputed=True,source_data_hashes=True,ensemble_metrics_recomputed=True,quantile_and_range_ordering=True,baseline_recomputed=True)
    write(out/'verification.json',verification);write(out/'summary.json',dict(selected_family=selected,metrics=results,comparisons=comparisons,probabilistic_comparison=probabilistic_comparison,trials=run['trials'],seconds=run['seconds']))
    colors={'static_curriculum':'#3766ad','static_direct':'#82a2cf','hyper_curriculum':'#cc691d','hyper_direct':'#dfab76','train_template':'#555555','persistence':'#999999'}
    mainkeys=['static_curriculum','static_direct','hyper_curriculum','hyper_direct','train_template']
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    axs[0].bar(np.arange(5),[results[k]['pinball_normalized'] for k in mainkeys],color=[colors[k] for k in mainkeys]);axs[0].set_xticks(np.arange(5),mainkeys,rotation=22,ha='right');axs[0].set_ylabel('Test normalized pinball loss (lower is better)')
    for k in mainkeys+['persistence']:axs[1].plot(np.arange(1,9)*15,results[k]['mean_price_mae_by_block'],label=k,color=colors[k],linestyle='--' if k in ['train_template','persistence'] else '-')
    axs[1].set_xlabel('End of 15-minute forecast block (minutes ahead)');axs[1].set_ylabel('Mean-price MAE ($)');axs[1].legend(fontsize=8)
    fig.suptitle(f'Validation-selected family: {selected}');fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');sample_ids=np.linspace(0,len(z['test_origins'])-1,6,dtype=int);p=predictions[selected];xs=np.arange(8)*15+7.5
    for ax,j in zip(axs.flat,sample_ids):
        scale=z['test_scale'][j];actual=z['test_final'][j]*scale;forecast=p[j]*scale
        ax.fill_between(xs,forecast[:,0,0],forecast[:,0,2],color='#3766ad',alpha=.18,label='Predicted q10–q90 of block mean')
        ax.plot(xs,actual[:,0],color='black',label='Actual block mean');ax.plot(xs,forecast[:,0,1],color='#3766ad',label='Predicted median of block mean')
        ax.plot(xs,actual[:,1],':',color='gray',alpha=.7);ax.plot(xs,actual[:,2],':',color='gray',alpha=.7,label='Actual min/max')
        ax.plot(xs,forecast[:,1,1],'--',color='#cc691d',alpha=.7);ax.plot(xs,forecast[:,2,1],'--',color='#cc691d',alpha=.7,label='Predicted median min/max')
        ax.axhline(0,color='gray',lw=.5);ax.set_title(str(np.datetime64(int(z['start_minute']+z['test_origins'][j]),'m')));ax.set_xlabel('Minutes ahead (block center)');ax.set_ylabel('Change from current Close ($)')
    axs.flat[0].legend(fontsize=7);fig.suptitle('Six examples spaced evenly through test; marginal intervals, not whole-path confidence bands');fig.savefig(out/'forecasts.png',dpi=170);plt.close(fig)
    examples=[]
    for j in sample_ids:
        scale=z['test_scale'][j];anchor=z['test_anchor'][j]
        examples.append(dict(origin=str(np.datetime64(int(z['start_minute']+z['test_origins'][j]),'m')),current_close=float(anchor),block_end_minutes=list(range(15,121,15)),statistics=['mean_Close','min_Low','max_High'],quantiles=Q.tolist(),prediction_dollars=(p[j]*scale+anchor).tolist(),actual_dollars=(z['test_final'][j]*scale+anchor).tolist()))
    write(out/'forecast_examples.json',dict(model=selected,examples=examples))
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(kind,mode) in zip(axs.flat,[(k,m) for k in ['static','hyper'] for m in ['curriculum','direct']]):
        for seed in [17,29]:
            rr=[v for v in curves[f'{kind}_{mode}_{seed}'] if v['stage']==4];xx=[v['epoch'] for v in rr];style='-' if seed==17 else '--'
            ax.plot(xx,[v['train_probe']['primary'] for v in rr],style,color='#3766ad',label=f'train probe, seed {seed}');ax.plot(xx,[v['validation']['primary'] for v in rr],style,color='#cc691d',label=f'validation, seed {seed}')
        ax.axhline(results['train_template']['validation_pinball'],color='gray',linestyle=':',label='template validation');ax.set_title(kind+' / '+mode);ax.set_xlabel('Global epoch');ax.set_ylabel('Stage-4 normalized pinball');ax.legend(fontsize=7)
    fig.savefig(out/'learning.png',dpi=160);plt.close(fig)
    lines=['# Многомасштабная TCN: curriculum и гиперсеть','',f"По validation выбрана **{selected}**. Обучение/оценка скриптом: {run['seconds']/60:.1f} мин. GPU: {run['gpu']}.",'',
           'Цель: восемь последовательных 15-минутных блоков на 120 минут вперёд. Для каждого: среднее M1 Close, минимум Low, максимум High; квантили 10%, 50%, 90%. Центральный ответ — условная медиана соответствующей статистики. Это прогноз общей картины, а не точных M1-свечей.', '',
           'Контекст: 6 часов M1, 2 суток M5, 2 недели H1. Завершённые временные интервалы, пропуски помечены. Три независимые TCN-ветки по шесть остаточных блоков, dropout 25%. Гиперсеть генерирует поправку ранга 4 к одному слою. Размеры: '+str(run['checks']['parameters'])+'.','',
           f"Train: {meta['train']['examples']} окон января, полный хронологический проход за эпоху. Validation: {meta['val']['examples']} окон 1–7 февраля. Test: {meta['test']['examples']} окон 15–29 февраля. Цели не пересекают границы периодов. Validation/test с шагом 15 минут; пересекающиеся горизонты зависимы.",'',
           'Curriculum: описание прошлого → активность будущих 60/120 минут → четыре 30-минутные средние цены → восемь 15-минутных блоков. Direct: только последняя задача. Две инициализации на вариант; прогнозы усреднены. Direct имеет тот же верхний бюджет эпох, что фактически использовал парный curriculum, но может закончить раньше из-за плато. Перенос лучших промежуточных весов также является частью curriculum; это не изолированный тест только порядка задач.','',
           '| Вариант | Validation pinball | Test pinball | MAE средней цены, $ | Покрытие 80% интервала средней цены | Ширина интервала, $ |','|---|---:|---:|---:|---:|---:|']
    for k,m in results.items():
        if k=='persistence':continue
        lines.append(f"| {k} | {m['validation_pinball']:.4f} | {m['pinball_normalized']:.4f} | {m['mean_price_mae_dollars']:.3f} | {100*m['mean_price_80pct_coverage']:.1f}% | {m['mean_price_interval_width_dollars']:.3f} |")
    lines+=['',f"Прогноз «цена останется текущей»: MAE средней цены **${results['persistence']['mean_price_mae_dollars']:.3f}**. train_template — квантили будущих нормированных траекторий, вычисленные только на TRAIN и масштабированные доступной прошлой волатильностью. Он учитывает типичный размах и служит более сильной вероятностной базой, чем нулевая линия.",'',
           'Гиперсеть *_fixed: выбранные веса сохранены; условие генератора заменено средним TRAIN-контекстом. Это диагностическая проверка без переобучения, исключённая из выбора семейства.','',
           '![Сравнение](comparison.png)','','![Прогнозы](forecasts.png)','','![Кривые обучения](learning.png)','','| Запуск | Эпохи по этапам | Выбранная финальная эпоха |','|---|---|---:|']
    for name,t in run['trials'].items():lines.append(f"| {name} | "+', '.join(f"{s['stage']}: {s['epochs']} ({s['stop']})" for s in t['stages'])+f" | {t['stages'][-1]['best_global_epoch']} |")
    stage_baseline={}
    for stage,key in [(1,'past'),(2,'activity'),(3,'coarse'),(4,'final')]:
        if stage==1:
            e=np.abs(z['val_past']-np.median(z['train_past'],axis=0));value=float(np.where(e<1,.5*e**2,e-.5).mean())
        else:
            template_stage=np.moveaxis(np.quantile(z['train_'+key],Q,axis=0),0,-1);value=float(pinball(template_stage,z['val_'+key]).mean())
        stage_baseline[stage]=value
    lines+=['','| Этап | Константный TRAIN-шаблон: validation loss | Static curriculum: средний лучший loss двух запусков | Hyper curriculum: средний лучший loss двух запусков |','|---|---:|---:|---:|']
    for stage in [1,2,3,4]:
        scores={kind:np.mean([next(v['best_validation'] for v in run['trials'][f'{kind}_curriculum_{seed}']['stages'] if v['stage']==stage) for seed in [17,29]]) for kind in ['static','hyper']}
        lines.append(f"| {stage} | {stage_baseline[stage]:.4f} | {scores['static']:.4f} | {scores['hyper']:.4f} |")
    lines+=['','Значения loss разных этапов напрямую несопоставимы. Эта таблица показывает качество относительно собственного простого шаблона каждого этапа на validation, использованном для выбора весов. Пользу curriculum для конечной задачи оцениваем по test-сравнению с direct.','']
    lines+=['','Разница MAE, $: отрицательное значение означает преимущество первого варианта. Интервалы получены парным bootstrap по целым календарным дням; при небольшом числе дней это ориентир, не доказательство переносимости.','', '| Сравнение | Разница | 95% интервал по дням |','|---|---:|---:|']
    for k,v in comparisons.items():lines.append(f"| {k} | {v['mean_difference']:.3f} | [{v['interval95'][0]:.3f}; {v['interval95'][1]:.3f}] |")
    value=probabilistic_comparison
    lines+=['',f"Разница нормированного pinball loss выбранной модели и исторического шаблона: {value['mean_difference']:.5f}; 95% интервал по дням [{value['interval95'][0]:.5f}; {value['interval95'][1]:.5f}]. Интервал также включает ноль.",'',
           'Вывод: описание видимого прошлого освоено заметно лучше константного шаблона; прогноз активности на validation улучшился умеренно. Для будущей центральной траектории выигрыш на test очень мал и не подтверждён проверкой по дням. Поэтапная обычная TCN выбрана на validation, но устойчивое преимущество curriculum и гиперсети не установлено. На равномерно выбранных графиках центральный прогноз чаще остаётся близко к текущей цене со слабым наклоном, тогда как диапазон расширяется. Покрытие интервала около 80% само по себе не означает, что модель угадывает изгибы цены.','']
    lines+=['','Проверены исходные цели и доступность агрегатов (data_verification.json), полные проходы данных, исходные веса общих TCN-веток, выбранные эпохи, порядок квантилей и диапазонов, независимый пересчёт метрик и baseline (verification.json).','',
           'Ограничения: один месяц обучения; один будущий тестовый период; подбор эпох и архитектуры на небольшой validation-выборке. Интервалы маргинальные: они не являются 80%-гарантией всей траектории. Среднее квантилей двух моделей — ансамблевый прогноз, а не точные квантили смеси распределений. Даже успех задачи описания прошлого не доказывает предсказуемость будущего. Результаты не позволяют заключить, что в M1 отсутствует информация.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(selected=selected,metrics=results,verification=verification,comparisons=comparisons),indent=2))
if __name__=='__main__':main()
