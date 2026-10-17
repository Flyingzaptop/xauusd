"""Independent precision and coverage report; validation-selected operating point."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from policy import apply_policy,fit_policy,metrics,route
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/direction_precision_20261010';REMOTE=OUT/'remote'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
LABELS={0:'SHORT',2:'LONG'}


def pct(x):return '—' if x is None else f'{100*x:.1f}%'


def main():
    run=json.loads((REMOTE/'run.json').read_text());assert run['status']=='complete' and run['selection_frozen_before_test']
    assert run['source_sha256']==hashlib.sha256(Path(__file__).with_name('train.py').read_bytes()).hexdigest()
    assert run['dataset_sha256']==hashlib.sha256((ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz').read_bytes()).hexdigest()
    curves={}
    for seed in [17,29]:
        hashes={v['initial_sha256'] for k,v in run['trials'].items() if k.endswith('_'+str(seed))};assert len(hashes)==1
    for name,trial in run['trials'].items():
        rows=json.loads((REMOTE/f'{name}_curve.json').read_text());curves[name]=rows
        assert len(rows)==12 and all(r['examples_seen']==27716 for r in rows)
        assert trial['best_epoch']==int(np.argmax([r['selection_policy']['selection_score'] for r in rows]))+1
    assert run['selected_family']==max(run['families'],key=lambda k:run['families'][k]['policy']['selection_score'])
    summary=[];ceilings={}
    for name,entry in run['families'].items():
        z=np.load(REMOTE/f'{name}_test.npz');cal=np.load(REMOTE/f'{name}_calibration.npz')
        policy=fit_policy(cal['p'],cal['target'],cal['timestamp']//86400)
        assert policy==entry['policy']
        pred=apply_policy(z['p'],policy);assert np.array_equal(pred,z['prediction'])
        assert np.array_equal(z['target'],np.where(z['delta_cents']>200,2,np.where(z['delta_cents'] < -200,0,1)))
        recalculated=metrics(pred,z['target']);assert recalculated==entry['test']
        ceilings[name]={}
        routed=route(cal['p']);days=cal['timestamp']//86400
        for c in [0,2]:
            candidates=[]
            for threshold in np.unique(cal['p'][routed==c,c]):
                mask=(routed==c)&(cal['p'][:,c]>=threshold);count=int(mask.sum())
                if count>=5 and len(np.unique(days[mask]))>=2:
                    tp=int(((cal['target']==c)&mask).sum());candidates.append((tp/count,tp,count,float(threshold)))
            ceilings[name][str(c)]=dict(max_raw_precision=max(candidates) if candidates else None,selected=entry['policy']['rules'][str(c)])
        summary.append(dict(family=name,selected=name==run['selected_family'],epochs=[run['trials'][f'{name}_{s}']['best_epoch'] for s in [17,29]],
                            calibration=entry['calibration'],test=recalculated,policy=policy))
    (OUT/'summary.json').write_text(json.dumps(dict(selected_family=run['selected_family'],rows=summary,seconds=run['seconds']),indent=2))
    (OUT/'calibration_precision_ceiling.json').write_text(json.dumps(ceilings,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,objective in enumerate(['ce','fp9']):
        for mode,color in [('adam','#2675b7'),('pcgrad','#cf781d')]:
            c=[curves[f'{objective}_{mode}_{s}'] for s in [17,29]]
            for row,key in enumerate(['train_cost','selection_score']):
                ys=np.array([[v['train_cost'] if row==0 else v['selection_policy']['selection_score'] for v in rows] for rows in c]);ax=axes[row,col]
                ax.plot(range(1,13),ys.mean(0),color=color,label=mode);ax.fill_between(range(1,13),ys.min(0),ys.max(0),color=color,alpha=.12)
                ax.set(title=f'{objective}: '+('train loss' if row==0 else 'Оценка precision на первой части validation'),xlabel='Эпоха');ax.grid(alpha=.15);ax.legend()
    fig.suptitle('Среднее двух seed · train loss разных objectives имеет разный масштаб')
    fig.savefig(OUT/'training.png',dpi=160);plt.close(fig)
    chosen=run['selected_family'];cal=np.load(REMOTE/f'{chosen}_calibration.npz');entry=run['families'][chosen]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,c in zip(axes,[0,2]):
        ix=np.flatnonzero(route(cal['p'])==c);ix=ix[np.argsort(-cal['p'][ix,c],kind='stable')]
        if len(ix):
            ns=np.arange(1,len(ix)+1);precision=np.cumsum(cal['target'][ix]==c)/ns
            ends=np.r_[cal['p'][ix[:-1],c]!=cal['p'][ix[1:],c],True]
            ax.plot(ns[ends],precision[ends],label='Calibration: precision при разном числе сигналов')
        rule=entry['policy']['rules'][str(c)]
        if rule['enabled']:ax.scatter([rule['signals']],[rule['precision']],s=75,color='#cc741d',label='Зафиксированный порог')
        ax.set(title=LABELS[c],xlabel='Количество сигналов на calibration',ylabel='Precision',ylim=(-.02,1.02));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'Выбор порогов: {chosen} · test не участвует')
    fig.savefig(OUT/'thresholds.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    labels=[v['family'] for v in summary]
    for ax,c in zip(axes,[0,2]):
        vals=[v['test']['per_direction'][str(c)] for v in summary]
        bars=ax.bar(labels,[0 if v['precision'] is None else v['precision'] for v in vals],color=['#2675b7' if v['selected'] else '#98b7cd' for v in summary])
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f"{v['tp']}/{v['signals']}",ha='center')
        ax.set(title=LABELS[c],ylim=(0,1.1),ylabel='Test precision');ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Правильные / все направленные сигналы · 0/0 означает отсутствие оценки')
    fig.savefig(OUT/'precision_counts.png',dpi=160);plt.close(fig)
    lines=['# Обучение с приоритетом precision','',
           'Таргет подтверждён пользователем: Close через 8 минут минус текущий Close; LONG >+$2, SHORT <−$2, остальные NONE. Только прошлые 512 свечей, TCN 605 315 параметров, январь train.',
           f"Выполнено 8 обучений по 12 полных эпох. GPU worker: {run['seconds']/60:.1f} мин. Два seed в каждом сочетании loss и оптимизатора; итог каждого сочетания — среднее scores двух моделей.",
           '', '## Что изменено','',
           'Убраны прежние обратные частотам веса классов. Контроль ce — обычная cross-entropy. fp9 добавляет штраф 9 × −log(1−p) за каждый ложный SHORT/LONG. Это дополнительный штраф, а не обещание определённого precision.',
           'Лучшая эпоха выбирается по precision на первых 265 окнах validation. На следующих 428 окнах выбираются отдельные пороги SHORT/LONG; нужно хотя бы 5 сигналов на двух датах. Если оценка не проходит порог, ответ NONE.',
           'Максимизируется нижняя оценка Wilson для precision, усреднённая по двум направлениям. Это защита от выбора единственного случайного попадания; после перебора и при зависимых данных она не является гарантированным доверительным пределом.',
           '',f"**По validation заранее выбран вариант `{chosen}`.** Все таблицы test рассчитаны после фиксации модели и порогов.",
           'Высокого precision добиться не удалось. У выбранной модели на test SHORT: 9 правильных из 109 сигналов (8.3%); LONG: 0 из 34 (0%). В сумме 9 из 143 (6.3%). Усиленный штраф fp9 не улучшил этот результат.',
           'Дополнительно перебраны допустимые пороги только на calibration: у выбранной модели максимальный сырой precision равен 5.5% для SHORT и 11.8% для LONG. Эти пороги совпали с выбранными. Более высокий precision не был скрыт нашим критерием Wilson при заданном минимуме 5 сигналов на двух датах.',
           '', '## Результаты: только направленные сигналы','',
           '| Вариант | Precision calibration | Сигналов calibration | Precision test | Правильно / все test | Recall test |',
           '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        cm=row['calibration'];m=row['test'];mark=' **выбран**' if row['selected'] else ''
        lines.append(f"| {row['family']}{mark} | {pct(cm['precision'])} | {cm['signals']} | {pct(m['precision'])} | {m['tp']} / {m['signals']} | {pct(m['recall'])} |")
    lines+=['','| Вариант | Направление | Правильно | Ложные | Precision | Recall |','|---|---|---:|---:|---:|---:|']
    for row in summary:
        for c in [0,2]:
            m=row['test']['per_direction'][str(c)]
            lines.append(f"| {row['family']} | {LABELS[c]} | {m['tp']} | {m['fp']} | {pct(m['precision'])} | {pct(m['recall'])} |")
    lines+=['','Нулевая выдача означает неопределённый precision, а не 100%. Нельзя выбирать другой вариант по этой таблице test и считать его независимым результатом.',
            'В calibration только 7 SHORT и 7 LONG. В test 690 окон, 18 SHORT и 12 LONG, шесть дат. При нескольких сигналах проценты сильно нестабильны; количество правильных и ложных сигналов важнее красивой округлённой цифры.',
            'Test уже использовался в предыдущих исследованиях. Это пилот на конкретном месяце; он не доказывает отсутствие информации в M1.',
            '', '## Проверки','',f"Проверки градиентов и правил: `{json.dumps(run['checks'])}`.",
            'Проверены хэши данных и кода, одинаковая инициализация, 12 полных эпох, выбор эпох и порогов без test. Все TP/FP, precision и recall пересчитаны по сохранённым predictions и labels. Веса и per-example scores сохранены.',
            '', '![Precision и число сигналов](precision_counts.png)','![Обучение](training.png)','![Пороги](thresholds.png)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(selected_family=chosen,rows=summary),indent=2))

if __name__=='__main__':main()
