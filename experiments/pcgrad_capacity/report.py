"""CPU-only report of adaptive capacity tests; selection was frozen before test."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/pcgrad_capacity_20261009';REMOTE=OUT/'remote'
REFERENCE=ROOT/'reports/pcgrad_mse_month_20261009/remote/run.json'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def bootstrap(a,b,days):
    unique,inv=np.unique(days,return_inverse=True);counts=np.bincount(inv)
    aa=np.bincount(inv,weights=a);bb=np.bincount(inv,weights=b)
    rng=np.random.default_rng(732);draws=rng.integers(len(unique),size=(3000,len(unique)))
    means_a=aa[draws].sum(1)/counts[draws].sum(1);means_b=bb[draws].sum(1)/counts[draws].sum(1)
    return dict(adam=float(a.mean()),pcgrad=float(b.mean()),pcgrad_gain_pct=float(100*(a.mean()-b.mean())/a.mean()),
                ci95_gain_pct=np.quantile(100*(means_a-means_b)/means_a,[.025,.975]).tolist(),days=len(unique))


def candle(ax,z,color,offset):
    for k,(h,o,c,l) in enumerate(z):
        x=k+offset;ax.plot([x,x],[l,h],color=color,lw=1)
        ax.add_patch(plt.Rectangle((x-.09,min(o,c)),.18,max(abs(c-o),.012),edgecolor=color,
                                  facecolor=color if c<o else 'white',lw=1))


def main():
    run=json.loads((REMOTE/'run.json').read_text());ref=json.loads(REFERENCE.read_text())
    assert run['status']=='complete'
    assert run['decisions_frozen_before_test'] or (run.get('all_segments_frozen_before_own_test') and run['selection_used_test'] is False)
    for segment in run.get('segments',[]):
        raw=json.loads((REMOTE/segment['report']).read_text())
        assert raw['source_sha256']==hashlib.sha256((ROOT/'experiments/pcgrad_capacity'/segment['source']).read_bytes()).hexdigest()
    assert run['source_sha256']==hashlib.sha256((ROOT/'experiments/pcgrad_capacity/train.py').read_bytes()).hexdigest()
    assert run['dataset_sha256']==ref['dataset_sha256']
    widths=[int(w) for w in run['stages']];curves={}
    table=[]
    for mode in ['adam','pcgrad']:
        names=[f'{mode}_{s}' for s in [17,29]]
        table.append(dict(width=128,params=617504,mode=mode,best_epochs=[ref['trials'][k]['best_epoch'] for k in names],
                          validation=float(np.mean([ref['trials'][k]['best_validation'] for k in names])),
                          test=float(np.mean([ref['metrics'][k]['normalized_mse'] for k in names])),
                          close_mse=float(np.mean([ref['metrics'][k]['close_mse'] for k in names])),reference=True))
    for w in widths:
        for seed in [17,29]:
            assert run['trials'][f'w{w}_adam_{seed}']['initial_sha256']==run['trials'][f'w{w}_pcgrad_{seed}']['initial_sha256']
        for mode in ['adam','pcgrad']:
            names=[f'w{w}_{mode}_{s}' for s in [17,29]]
            for name in names:
                rows=json.loads((REMOTE/f'{name}_curve.json').read_text());curves[name]=rows
                assert len(rows)==13 and all(r.get('examples_seen',0)==27716 for r in rows[1:])
                assert [r['lr'] for r in rows[1:]]==run['settings']['lr']
                case=np.load(REMOTE/f'{name}_cases.npz')
                assert np.allclose(((case['prediction']-case['target'])**2).mean((1,2)),case['normalized_mse'],atol=1e-6,rtol=3e-6)
            table.append(dict(width=w,params=run['stages'][str(w)]['parameters'],mode=mode,
                              best_epochs=[run['trials'][k]['best_epoch'] for k in names],
                              validation=float(np.mean([run['trials'][k]['best_validation'] for k in names])),
                              test=float(np.mean([run['metrics'][k]['normalized_mse'] for k in names])),
                              close_mse=float(np.mean([run['metrics'][k]['close_mse'] for k in names])),
                              last_test=float(np.mean([run['metrics'][k+'_last']['normalized_mse'] for k in names])),
                              tv_ratio=float(np.mean([run['metrics'][k]['tv_ratio'] for k in names])),
                              final_train=float(np.mean([curves[k][-1]['train_fixed']['normalized_mse'] for k in names])),
                              final_validation=float(np.mean([curves[k][-1]['validation']['normalized_mse'] for k in names])),
                              late_gain_pct=run['stages'][str(w)]['decision']['arms'][mode]['late_improvement_pct'],reference=False))
    (OUT/'summary.json').write_text(json.dumps(dict(rows=table,stop_reason=run['stop_reason'],selected_width=run['selected_width']),indent=2))
    colors={w:plt.colormaps['viridis'](i/max(1,len(widths)-1)) for i,w in enumerate(widths)}
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for col,mode in enumerate(['adam','pcgrad']):
        for w in widths:
            c=[curves[f'w{w}_{mode}_{s}'] for s in [17,29]]
            for row,kind in enumerate(['train_fixed','validation']):
                ys=np.array([[v[kind]['normalized_mse'] for v in rows] for rows in c])
                axes[row,col].plot(range(13),ys.mean(0),color=colors[w],label=f"{run['stages'][str(w)]['parameters']:,}")
                axes[row,col].fill_between(range(13),ys.min(0),ys.max(0),color=colors[w],alpha=.08)
        base=next(r['validation'] for r in table if r['reference'] and r['mode']==mode)
        axes[1,col].axhline(base,color='#555555',ls='--',lw=1,label='Лучший исходный 618 тыс.')
        axes[0,col].set_title(f'{mode}: train, фиксированные веса');axes[1,col].set_title(f'{mode}: validation')
        for ax in axes[:,col]:ax.set(xlabel='Полная эпоха',ylabel='Нормированный OHLC MSE');ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Последовательное уменьшение TCN · среднее двух seed\nПолупрозрачная область — разброс между двумя запусками',fontsize=15)
    fig.savefig(OUT/'capacity_curves.png',dpi=170);fig.savefig(OUT/'capacity_curves.pdf');plt.close(fig)
    small=[w for w in widths if w<=20]
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    palette=plt.colormaps['tab10']
    for col,mode in enumerate(['adam','pcgrad']):
        for i,w in enumerate(small):
            for row,kind in enumerate(['train_fixed','validation']):
                ys=np.array([[v[kind]['normalized_mse'] for v in curves[f'w{w}_{mode}_{s}'][1:]] for s in [17,29]])
                ax=axes[row,col];ax.plot(range(1,13),ys.mean(0),color=palette(i),label=f"{run['stages'][str(w)]['parameters']:,}")
                ax.fill_between(range(1,13),ys.min(0),ys.max(0),color=palette(i),alpha=.10)
        for row,title in enumerate(['train','validation']):
            ax=axes[row,col];ax.set(title=f'{mode}: {title}',xlabel='Полная эпоха',ylabel='Нормированный OHLC MSE')
            ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Меньшие модели: видно замедление обучения и плато',fontsize=15)
    fig.savefig(OUT/'small_capacity_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for mode,color in [('adam','#2866b1'),('pcgrad','#d57716')]:
        t=sorted([r for r in table if r['mode']==mode],key=lambda r:r['params'])
        for ax,key,title in zip(axes,['best_epochs','validation','test'],['Эпоха лучшего validation','Лучший validation MSE','Test MSE выбранных checkpoints']):
            ys=[np.mean(r[key]) if key=='best_epochs' else r[key] for r in t]
            ax.plot([r['params'] for r in t],ys,marker='o',color=color,label=mode)
            ax.set(xscale='log',xlabel='Число параметров',title=title);ax.grid(alpha=.15);ax.legend()
    fig.suptitle('Поздний минимум не обязательно означает меньшую ошибку',fontsize=15)
    fig.savefig(OUT/'capacity_summary.png',dpi=170);plt.close(fig)
    # Same chronological examples as the previous pilot; widths chosen only by validation.
    chosen={mode:int(run['selected_width'][mode]) for mode in ['adam','pcgrad']}
    a=np.load(REMOTE/f"w{chosen['adam']}_adam_17_cases.npz");b=np.load(REMOTE/f"w{chosen['pcgrad']}_pcgrad_17_cases.npz")
    assert np.array_equal(a['start'],b['start']) and np.array_equal(a['target'],b['target'])
    fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
    ix=np.linspace(0,len(a['start'])-1,8,dtype=int)[1:-1]
    for ax,i in zip(axes.flat,ix):
        scale=a['scale'][i]*1e4
        candle(ax,a['target'][i]*scale,'#30343b',-.23);candle(ax,a['prediction'][i]*scale,'#2866b1',0);candle(ax,b['prediction'][i]*scale,'#d57716',.23)
        ax.set(title=str(np.datetime64(int(a['timestamp'][i]),'s'))+' UTC',xlabel='Скрытая минута',ylabel='Б.п. от левого Close');ax.grid(alpha=.15)
    handles=[plt.Line2D([],[],color=c,lw=3,label=l) for c,l in [('#30343b','Реальные свечи'),('#2866b1',f"Adam, width {chosen['adam']}"),('#d57716',f"PCGrad, width {chosen['pcgrad']}")]]
    fig.legend(handles=handles,loc='outside lower center',ncol=3)
    fig.suptitle('Размер каждой версии выбран по validation · свечи на test, seed 17',fontsize=16)
    fig.savefig(OUT/'candles.png',dpi=170);plt.close(fig)
    selected_comparison={}
    for key in ['normalized_mse','hcl_mse','close_mse']:
        av=np.mean([np.load(REMOTE/f"w{chosen['adam']}_adam_{s}_cases.npz")[key] for s in [17,29]],0)
        bv=np.mean([np.load(REMOTE/f"w{chosen['pcgrad']}_pcgrad_{s}_cases.npz")[key] for s in [17,29]],0)
        selected_comparison[key]=bootstrap(av,bv,a['timestamp']//86400)
    (OUT/'selected_comparison.json').write_text(json.dumps(selected_comparison,indent=2))
    # Preserve matched-width comparisons; selected architectures can differ.
    paired={}
    for w in widths:
        av=np.mean([np.load(REMOTE/f'w{w}_adam_{s}_cases.npz')['normalized_mse'] for s in [17,29]],0)
        bv=np.mean([np.load(REMOTE/f'w{w}_pcgrad_{s}_cases.npz')['normalized_mse'] for s in [17,29]],0)
        paired[str(w)]=bootstrap(av,bv,a['timestamp']//86400)
    (OUT/'matched_width_comparison.json').write_text(json.dumps(paired,indent=2))
    diagnostics={name:{key:float(np.mean([r['train'][key] for r in rows[1:]]))
                       for key in ['conflict_fraction','plain_harm_fraction','projected_harm_fraction','adam_harm_fraction','clipped']}
                 for name,rows in curves.items()}
    (OUT/'gradient_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    lines=['# Последовательное уменьшение TCN','',f"Проверено уменьшенных размеров: {len(widths)}; обучений: {len(run['trials'])}. Время worker: {run['seconds']/60:.1f} мин.",
           '','Месяц train — январь 2024, 27 716 окон за эпоху. 1–7 февраля validation, 8–14 февраля test. Контекст 512 слева и справа; восстанавливаем восемь минут.',
           'Две версии: MSE + Adam и MSE + Adam + групповой PCGrad. Два seed, по 12 полных эпох, одинаковый порядок данных и LR. Меняется только ширина encoder/decoder; глубина, рецептивное поле и отображение в OHLC сохраняются.',
           'Train и validation у новых моделей измерены при одних и тех же фиксированных весах после эпохи. Train online дополнительно сохранён в JSON.',
           '', '## Как читать результат','',
           'Уменьшение действительно отодвинуло лучший validation: исходная сеть — эпохи 1–2, 38 912 параметров — 3–4, 18 752 — 5–9, 8 924 — последняя, двенадцатая эпоха во всех четырёх запусках. Это подтверждает изменение динамики при уменьшении размера.',
           'На одинаковом бюджете 12 эпох лучший средний validation у обеих версий даёт размер 38 912. При 8 924 параметрах обучение ещё не достигло минимума; худший результат пока не доказывает недостаточную ёмкость. Нужен более длинный прогон, чтобы отделить медленное обучение от ограничения модели.',
           'При 4 064 параметрах и ниже улучшение в этих 12 эпохах небольшое, качество хуже. Продолжать уменьшение этой реализации до width 1 нельзя без изменения LayerNorm.',
           'У выбранного размера 38 912 test OHLC MSE: Adam 3.59703, PCGrad 3.59688. Выигрыш PCGrad около 0.004%; парный bootstrap по шести датам допускает оба знака разницы. Убедительного выигрыша по общей ошибке нет. Предсказания всё ещё сильно сглаживают свечи; изменение размера это не исправило.',
           '', '## Результаты','', '| Параметров | Метод | Лучшие эпохи, два seed | Validation MSE | Test MSE | Последний train | Последний validation |',
           '|---:|---|---|---:|---:|---:|---:|']
    for r in table:
        ft='—' if r['reference'] else f"{r['final_train']:.5f}";fv='—' if r['reference'] else f"{r['final_validation']:.5f}"
        lines.append(f"| {r['params']:,} | {r['mode']} | {r['best_epochs']} | {r['validation']:.5f} | {r['test']:.5f} | {ft} | {fv} |")
    lines+=['','618 тыс. — сохранённые агрегаты предыдущего теста; исходные поминутные предсказания и кривые утрачены при сбросе среды. Его обучающий код и агрегаты восстановлены с Colab.',
            '', '## Правило остановки','',
            'На каждом размере проверяется: у обеих версий средний лучший validation эпох 4–12 хотя бы на 0.2% ниже лучшего эпох 1–3; train между этими выбранными эпохами падает у обоих seed; поздний validation не хуже исходной модели более чем на 0.2%. Иначе число параметров снова уменьшается примерно вдвое.',
            'Это правило отличает более длительное полезное обучение от простого замедления. Самый малый допустимый width 2: при width 1 LayerNorm уничтожает различия единственного канала.',
            '',f"Причина остановки: `{run['stop_reason']}`.",
            f"Лучший размер среди уменьшенных по среднему validation: Adam width{chosen['adam']}, PCGrad width{chosen['pcgrad']}.",
            'Первый этап остановился на 18 752 параметрах. После прямого указания пользователя продолжить заранее зафиксирован второй этап: 8 924 → 4 064 → 2 012 → 1 004 → 608 параметров, без досрочной остановки.',
            'Внутри каждого этапа решения о размерах зафиксированы до его test. Первый test уже был вычислен до расширения эксперимента; это повторно используемая исследовательская выборка. Выбор размеров и checkpoints использует только validation. Все лучшие и последние checkpoints оценены и сохранены.',
            '', '## Сравнение с интерполяцией без обучения','',
            f"Линейный мост между известными границами: test OHLC MSE {run['metrics']['bridge']['normalized_mse']:.5f}, Close MSE {run['metrics']['bridge']['close_mse']:.5f} б.п.².",
            'Снижение общей ошибки OHLC само по себе не означает восстановления траектории закрытий. Close MSE и отношение вариации предсказания к реальной сохранены отдельно в summary.json.',
            '', '## Проверки и ограничения','',
            f"Проверки реализации: `{json.dumps(run['checks'])}`.",
            'Проверены SHA данных/кода, равенство начальных весов в каждой паре, 12 полных эпох по 27 716 окон, расписание LR; test MSE независимо пересчитан из предсказаний.',
            'Всего шесть дат test и два seed; это исследовательский пилот. Данные ранее использовались в исследовании. Снижение train и рост validation согласуются с переобучением, но не доказывают, что сеть просто запоминает примеры или что информации в данных нет.',
            'PCGrad применяется к четырём группам по 32 примера и не исключает все конфликты. Задача использует правый контекст и не является причинным прогнозом будущего.',
            '', '![Сводка размеров](capacity_summary.png)','![Кривые train/validation](capacity_curves.png)','![Малые модели](small_capacity_curves.png)','![Свечи](candles.png)',
            '', 'Артефакты: remote/ — модели, Adam state, кривые, предсказания; summary.json — все размеры; selected_comparison.json и matched_width_comparison.json — парный bootstrap по дням. Выбранные по validation размеры могут различаться; сравнение при одинаковом размере находится во втором файле. Интервалы не скорректированы за множество сравнений.',
            'gradient_diagnostics.json показывает конфликты групп и долю групп, которым шаг мешает в линейном приближении. projected_harm_fraction — до преобразования Adam; adam_harm_fraction — после фактического изменения весов, на периодической подвыборке шагов. Эти величины не гарантируют знак фактического изменения loss после конечного шага.',
            'Для обычного Adam projected_harm_fraction вычисляется для сравнения, но проекция не применяется к обновлению весов.',
            '', 'Воспроизведение на CUDA: `python experiments/pcgrad_capacity/reproduce.py --data reports/pcgrad_mse_month_20261009/dataset.npz --out /path/to/new-output`. Требуются PyTorch с CUDA и NumPy; matplotlib нужен только для отчёта.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=table,stop_reason=run['stop_reason'],selected_width=chosen),indent=2))

if __name__=='__main__':main()
