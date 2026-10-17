"""Independent future-only audit and stage-two activity forecast report."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import Q,pinball,describe,baseline_predictions
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/transformer_stage1'))
from representation import decode_path

def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def rank(v):
    order=np.argsort(v,kind='stable');sorted_v=v[order];_,starts,counts=np.unique(sorted_v,return_index=True,return_counts=True);r=np.empty(len(v));r[order]=np.repeat(starts+(counts-1)/2,counts);return r

def spearman(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(rank(a),rank(b))[0,1])
def day_ci(a,b,days):
    unique=np.unique(days);sums=np.array([(a[days==d]-b[days==d]).sum() for d in unique]);counts=np.array([(days==d).sum() for d in unique]);rng=np.random.default_rng(184);ix=rng.integers(0,len(unique),(10000,len(unique)));boot=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean_difference=float(np.mean(a-b)),interval95=np.quantile(boot,[.025,.975]).tolist(),days=len(unique))
def old_mae(p,z):
    path=decode_path(p,z['val_anchor_log'],float(z['s15']));e=np.abs(np.exp(path)-np.exp(z['val_target_log']));return float(e.mean()),float(e[:,:,2].mean())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reports/transformer_stage2_20261012');a=ap.parse_args();out=a.out;r=out/'remote';run=read(r/'run.json');z=np.load(out/'dataset.npz');meta=read(out/'dataset.json')
    assert run['status']=='complete' and run['stage']==2 and run['no_stage1_extension'] and run['later_stages_disabled'] and run['selection_frozen_before_test'];assert read(r/'worker_result.json')['returncode']==0
    assert run['data_sha256']==hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()
    for name in ['train.py','model.py','data.py','metrics.py']:assert run['source_sha256'][name]==hashlib.sha256((ROOT/'experiments/transformer_stage2'/name).read_bytes()).hexdigest()
    for name,source in [('stage1_model.py','model.py'),('representation.py','representation.py')]:assert run['source_sha256'][name]==hashlib.sha256((ROOT/'experiments/transformer_stage1'/source).read_bytes()).hexdigest()
    assert len(run['trials'])==4;curves={};first=np.load(ROOT/'reports/transformer_stage1_20261012/dataset.npz');indices=np.searchsorted(first['val_origins'],z['val_origins']);assert np.array_equal(first['val_origins'][indices],z['val_origins'])
    for name,t in run['trials'].items():
        rows=read(r/f'{name}_curve.json');curves[name]=rows;assert len(rows)==t['epochs'];assert all(row['examples_seen']==len(z['train_origins']) for row in rows)
        assert all(row['phase']==('head_only' if row['epoch']<=3 else 'joint') for row in rows);best=min(rows,key=lambda v:v['validation_pinball']);assert best['epoch']==t['best_epoch']
        p=np.load(r/f'{name}_validation.npz')['prediction'];assert np.isclose(pinball(p,z['val_y']).mean(),t['best_validation_pinball'],rtol=1e-5)
        checkpoint=ROOT/f'reports/transformer_stage1_20261012/remote/{name}_best.pt';assert t['transfer']['all_legacy_weights_exact'];assert t['transfer']['checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        before=np.load(r/f'{name}_past_before.npz')['angles'];reference=np.load(ROOT/f'reports/transformer_stage1_20261012/remote/{name}_validation.npz')['angles'][indices]
        # Batch shape changes can change bf16 rounding, so compare decoded metric
        # and angles within a small tolerance, not bitwise equality.
        assert np.allclose(before,reference,atol=.008,rtol=.01)
        for phase in ['before','after']:
            past=np.load(r/f'{name}_past_{phase}.npz')['angles'];mae,close=old_mae(past,z);assert np.isclose(mae,t['stage1_'+phase]['path_mae_dollars']);assert np.isclose(close,t['stage1_'+phase]['close_mae_dollars'])
    results={};predictions={}
    for family,v in run['families'].items():
        p=np.load(r/f'{family}_test.npz')['prediction'];members=[np.load(r/f'{family}_{seed}_test.npz')['prediction'] for seed in [17,29]];assert np.array_equal(p,np.mean(members,axis=0))
        assert np.isfinite(p).all() and (p>0).all() and (np.diff(p,axis=-1)>=0).all();m=describe(p,z,'test')
        for k in ['pinball_normalized','mae_dollars','coverage80']:assert np.allclose(m[k],v['test'][k],rtol=1e-6)
        names=[f'{family}_{seed}' for seed in [17,29]] if family!='hyper_fixed' else [f'hyper_{seed}_fixed' for seed in [17,29]];vp=np.mean([np.load(r/f'{name}_validation.npz')['prediction'] for name in names],axis=0)
        assert np.isclose(pinball(vp,z['val_y']).mean(),v['validation']['pinball_normalized'],rtol=1e-6)
        predictions[family]=p;results[family]=dict(validation_pinball=v['validation']['pinball_normalized'],**m)
    for name,p in baseline_predictions(z,'test').items():
        assert np.array_equal(p,np.load(r/f'{name}_test.npz')['prediction']);m=describe(p,z,'test');vp=baseline_predictions(z,'val')[name]
        assert np.isclose(pinball(vp,z['val_y']).mean(),run['baselines'][name]['validation']['pinball_normalized'],rtol=1e-6)
        predictions[name]=p;results[name]=dict(validation_pinball=run['baselines'][name]['validation']['pinball_normalized'],**m)
    selected=min(['static','hyper'],key=lambda k:results[k]['validation_pinball']);baseline=min(run['baselines'],key=lambda k:results[k]['validation_pinball']);assert selected==run['selected_family'] and baseline==run['selected_baseline']
    # Baselines may inspect shape, but must never inspect validation/test targets.
    copied={k:z[k] for k in z.files};copied['test_y']=np.full_like(z['test_y'],999.)
    for name,p in baseline_predictions(copied,'test').items():assert np.array_equal(p,predictions[name])
    days=(int(z['start_minute'])+z['test_origins'])//1440;errors={k:pinball(p,z['test_y']).mean((1,2,3)) for k,p in predictions.items()};comparisons={}
    for k in ['train_distribution','past_activity',('static' if selected=='hyper' else 'hyper')]:comparisons[f'{selected}_minus_{k}']=day_ci(errors[selected],errors[k],days)
    for k,p in predictions.items():
        results[k]['rank_correlations']=[[spearman(p[:,h,m,1],z['test_y'][:,h,m]) for m in range(2)] for h in range(2)]
        results[k]['quantile_cdf']=(z['test_y'][...,None]<=p).mean(0).tolist()
    # Exploratory context-alignment check; never influences family selection.
    rng=np.random.default_rng(789);groups=[np.flatnonzero(days==day) for day in np.unique(days)];shuffled=[]
    for _ in range(2000):
        ix=np.arange(len(days))
        for g in groups:ix[g]=rng.permutation(g)
        shuffled.append(float(pinball(predictions[selected][ix],z['test_y']).mean()))
    aligned=float(errors[selected].mean());shuffle_mean=float(np.mean(shuffled))
    alignment=dict(actual_pinball=aligned,within_day_shuffled_mean=shuffle_mean,relative_advantage=1-aligned/shuffle_mean,permutations=2000,seed=789,note='Exploratory descriptive ablation after inspecting forecasts; dependent windows are not exchangeable, no formal permutation p-value claimed.')
    verification=dict(status='passed',stage=2,trials=4,total_epochs=sum(t['epochs'] for t in run['trials'].values()),train_examples_per_epoch=len(z['train_origins']),exact_pretrained_checkpoints=True,legacy_weights_verified_in_warmup=True,source_data_hashes=True,future_metrics_recomputed=True,baseline_future_targets_not_used=True,stage1_retention_recomputed=True,quantiles_positive_ordered=True,no_stage3=True)
    write(out/'verification.json',verification);write(out/'summary.json',dict(selected_family=selected,selected_baseline=baseline,metrics=results,comparisons=comparisons,context_alignment=alignment,trials=run['trials'],seconds=run['seconds']))
    keys=['static','hyper','train_distribution','past_activity'];colors={'static':'#3266ad','hyper':'#cb7724','train_distribution':'#666666','past_activity':'#398657'}
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,rows in curves.items():
        kind=name.rsplit('_',1)[0];style='-' if name.endswith('17') else '--';xs=[v['epoch'] for v in rows]
        axs[0].plot(xs,[v['train_probe_pinball'] for v in rows],style,color=colors[kind],label=name);axs[1].plot(xs,[v['validation_pinball'] for v in rows],style,color=colors[kind],label=name)
    for ax,title in zip(axs,['Train: fixed probe','Validation']):ax.set_title(title);ax.set_xlabel('Full chronological epoch');ax.set_ylabel('Normalized future pinball');ax.axvline(3.5,color='#aaaaaa',ls=':',label='unfreeze backbone');ax.legend(fontsize=7)
    for k in ['train_distribution','past_activity']:axs[1].axhline(results[k]['validation_pinball'],color=colors[k],ls=':',alpha=.8)
    fig.suptitle('STAGE 2 ONLY — FUTURE activity from pretrained Transformer');fig.savefig(out/'learning.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels=['60 min range','60 min total Close variation','120 min range','120 min total Close variation']
    for ax,(h,m),label in zip(axs.flat,[(0,0),(0,1),(1,0),(1,1)],labels):
        ax.bar(range(4),[results[k]['mae_dollars'][h][m] for k in keys],color=[colors[k] for k in keys]);ax.set_xticks(range(4),keys,rotation=18);ax.set_title(label);ax.set_ylabel('Median forecast MAE ($)')
    fig.suptitle('Future magnitudes, not direction or net price displacement');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    p=predictions[selected];factor=z['target_scale'][None,:,:,None]*z['test_anchor_price'][:,None,None,None];dollar=p*factor;truth=z['test_relative']*z['test_anchor_price'][:,None,None];dates=np.array([str(np.datetime64(int(z['start_minute']+i),'m')) for i in z['test_origins']]);sample=np.linspace(0,len(dates)-1,6,dtype=int)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(h,m),label in zip(axs.flat,[(0,0),(0,1),(1,0),(1,1)],labels):
        y=dollar[sample,h,m];ax.vlines(range(6),y[:,0],y[:,2],color=colors[selected],lw=5,alpha=.45,label='Predicted q10–q90');ax.scatter(range(6),y[:,1],color=colors[selected],label='Predicted median');ax.scatter(range(6),truth[sample,h,m],color='black',marker='x',s=55,label='Actual future');ax.set_xticks(range(6),[v[5:] for v in dates[sample]],rotation=30,ha='right');ax.set_ylabel('$');ax.set_title(label);ax.legend(fontsize=8)
    fig.suptitle('Six origins spaced evenly through test — target strictly AFTER the origin');fig.savefig(out/'forecast_examples.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');names=list(run['trials']);xx=np.arange(4)
    ax.bar(xx-.18,[run['trials'][n]['stage1_before']['path_mae_dollars'] for n in names],.36,label='Before stage2',color='#3266ad');ax.bar(xx+.18,[run['trials'][n]['stage1_after']['path_mae_dollars'] for n in names],.36,label='After stage2',color='#cb7724');ax.set_xticks(xx,names,rotation=10);ax.set_ylabel('Stage1 OHLC reconstruction MAE ($)');ax.set_title('Unchanged old readout applied before/after fine-tuning');ax.legend();fig.savefig(out/'retention.png',dpi=170);plt.close(fig)
    write(out/'forecast_examples.json',dict(model=selected,targets=['range','sum_abs_Close_changes'],horizons=[60,120],examples=[dict(origin=dates[i],forecast_dollars=dollar[i].tolist(),actual_dollars=truth[i].tolist(),quantiles=Q.tolist()) for i in sample]))
    lines=['# Transformer — только этап 2: будущая активность','',f"Выбрана по validation модель **{selected}**, простой ориентир **{baseline}**. Обучение/оценка скриптом: {run['seconds']/60:.2f} мин. Первый этап не продлевался; третий не запускался.",'',
           'Цели действительно будущие: диапазон max High−min Low и сумма абсолютных минутных изменений Close на следующих 60 и 120 минутах. Сумма колебаний — длина пути, не направленное изменение цены. Модель не выдаёт LONG/SHORT или будущие свечи. Квантили10/50/90 описывают маргинальную неопределённость величин.','',
           'Входы побитово совпадают с этапом1: угловые H/O/C/L-векторы и доступность данных. Абсолютные уровни, новые признаки волатильности и будущие свечи во входе отсутствуют. Цели относительны к доступному текущему Close и нормированы фиксированными медианами TRAIN. Доллары рассчитываются после модели для отчёта.','',
           f"Train: {meta['train']['examples']} окон января, полный хронологический проход за эпоху. Validation: {meta['val']['examples']} окон 1–7 февраля. Test: {meta['test']['examples']} окон 15–29 февраля. Все будущие120 минут реально наблюдаются без пропусков; цели не пересекают границы выборок. Рыночные периоды исследовательски использованы ранее, не новый независимый торговый тест.",'',
           'Перенесены все прежние веса из best-checkpoint каждого запуска этапа1. Добавлены только выходы активности. Старый readout заморожен; первые3 эпохи заморожены все прежние веса, затем они обновляются с LR в10 раз меньше новых выходов. Dropout25%. Loss — обычный pinball для четырёх будущих величин и трёх квантилей. Оптимизатор для новой задачи сброшен, веса сохранены.','',
           '| Вариант | Validation pinball | Test pinball | MAE размаха60, $ | MAE размаха120, $ | MAE колебаний60, $ | MAE колебаний120, $ |','|---|---:|---:|---:|---:|---:|---:|']
    for k,m in results.items():
        e=m['mae_dollars'];lines.append(f"| {k} | {m['validation_pinball']:.4f} | {m['pinball_normalized']:.4f} | {e[0][0]:.3f} | {e[1][0]:.3f} | {e[0][1]:.3f} | {e[1][1]:.3f} |")
    lines+=['','train_distribution — безусловные квантили TRAIN. past_activity — квантили отношения будущей активности к прошлому Close-прокси, подогнанные только на TRAIN. Сам прокси восстановлен из входных Close-векторов в произвольных координатах; абсолютные цены и будущие ответы ему не нужны. Лучший baseline выбран на validation до теста. hyper_fixed — диагностическая фиксация условия гиперсети средним TRAIN-контекстом, без переобучения и без участия в выборе модели.','',
           '| Вариант | Покрытие80%: range60 / range120 | Покрытие80%: variation60 / variation120 |','|---|---:|---:|']
    for k,m in results.items():c=m['coverage80'];lines.append(f"| {k} | {100*c[0][0]:.1f}% / {100*c[1][0]:.1f}% | {100*c[0][1]:.1f}% / {100*c[1][1]:.1f}% |")
    lines+=['','Калибровка выбранной модели: для суммы колебаний на60/120мин ниже q10 оказалось '+f"{100*results[selected]['quantile_cdf'][0][1][0]:.1f}% / {100*results[selected]['quantile_cdf'][1][1][0]:.1f}%"+' ответов вместо ожидаемых10%. Это показывает смещение нижней границы вверх и завышение низкой активности, а не доказывает, что проблема только в ширине интервала. Никакая калибровка по test не применялась.','']
    lines+=['','[Интерактивный просмотр всех 554 будущих прогнозов](review.html) — модели, простые ориентиры, выбор окна и худшего случая.','','![Кривые обучения](learning.png)','','![Сравнение](comparison.png)','','![Будущие цели](forecast_examples.png)','','![Сохранение первого этапа](retention.png)','','| Запуск | Эпохи / выбранная | MAE прежнего readout до → после, $ |','|---|---:|---:|']
    for name,t in run['trials'].items():lines.append(f"| {name} | {t['epochs']} / {t['best_epoch']} | {t['stage1_before']['path_mae_dollars']:.3f} → {t['stage1_after']['path_mae_dollars']:.3f} |")
    lines+=['','Ошибка первого этапа проверяется на тех же validation-окнах через прежний фиксированный выход. Его ухудшение показывает потерю совместимости с прежним декодером; оно не доказывает полное исчезновение информации из новых признаков. Loss первого этапа в обучение не добавлялся.','',
           '| Парная разница pinball: выбранная модель минус… | Средняя разница | 95% интервал по дням |','|---|---:|---:|']
    for k,v in comparisons.items():lines.append(f"| {k} | {v['mean_difference']:.4f} | [{v['interval95'][0]:.4f}; {v['interval95'][1]:.4f}] |")
    lines+=['',f"Дополнительная описательная проверка контекста: при2000 случайных перестановках прогнозов внутри каждого календарного дня средний pinball вырос с {alignment['actual_pinball']:.4f} до {alignment['within_day_shuffled_mean']:.4f}. Правильная привязка к окну даёт {100*alignment['relative_advantage']:.1f}% выигрыша относительно такого перемешивания. Проверка добавлена после просмотра прогнозов, не участвовала в выборе весов или семейства; перестановочный p-value не заявляется из-за зависимости перекрывающихся окон.",'',
           'Оценка этапа2: на этом периоде получен полезный прогноз будущей активности. Обычный Transformer оказался лучше варианта с гиперсетью по выбранному loss. Интервалы суммарной активности плохо откалиброваны, и точность старого фиксированного выхода реконструкции заметно ухудшилась. Успех нового прогноза не означает предсказание направления или формы будущих свечей.','']
    gain=1-results[selected]['pinball_normalized']/results[baseline]['pinball_normalized'];lines+=['',f"Разница выбранной модели относительно выбранного простого ориентира на test: {100*gain:.1f}% уменьшения pinball (отрицательное значение означает ухудшение). Bootstrap по целым календарным дням описательный; перекрывающиеся окна зависимы. Подбор на небольшой validation и один период не доказывают переносимость результата.",'',
           'Проверки: исходные будущие цели пересчитаны, все будущие цены искусственно изменены — входы не изменились, target изменился; незавершённые H1 исключены. Проверены перенос весов, неизменность в warmup, исходные хэши, полные эпохи, метрики, baseline без будущих labels, порядок квантилей и показатели прежнего навыка. См. data_verification.json и verification.json.','',
           'Дальнейшие этапы не выполнялись. Результат подготовлен для оценки пользователем. Этот запуск не проверял, поможет ли более долгое обучение первого этапа; использованы именно уже сохранённые веса.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(selected_family=selected,selected_baseline=baseline,metrics=results,comparisons=comparisons,verification=verification),indent=2))
if __name__=='__main__':main()
