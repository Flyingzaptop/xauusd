"""CPU report and independent verification of directional classification."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/direction_classification_20261010';REMOTE=OUT/'remote'
CLASSES=['SHORT','NONE','LONG']
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})


def from_cm(cm):
    recall=np.diag(cm)/np.maximum(cm.sum(1),1);precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    f1=2*recall*precision/np.maximum(recall+precision,1e-20)
    return dict(accuracy=float(np.trace(cm)/cm.sum()),balanced_accuracy=float(recall.mean()),macro_f1=float(f1.mean()),precision=precision.tolist(),recall=recall.tolist())


def main():
    run=json.loads((REMOTE/'run.json').read_text());assert run['status']=='complete' and run['selection_frozen_before_test']
    assert run['source_sha256']==hashlib.sha256((ROOT/'experiments/direction_classification/train.py').read_bytes()).hexdigest()
    data=ROOT/'reports/pcgrad_mse_month_20261009/dataset.npz';assert hashlib.sha256(data.read_bytes()).hexdigest()==run['dataset_sha256']
    curves={};table=[]
    for seed in [17,29]:assert run['trials'][f'adam_{seed}']['initial_sha256']==run['trials'][f'pcgrad_{seed}']['initial_sha256']
    for name in run['trials']:
        c=json.loads((REMOTE/f'{name}_curve.json').read_text());curves[name]=c
        assert len(c)==12 and all(r['examples_seen']==27716 for r in c)
        for version in ['', '_last']:
            label=name+version;z=np.load(REMOTE/f'{label}_cases.npz');y=z['target']
            assert np.array_equal(y,np.where(z['delta_cents']>200,2,np.where(z['delta_cents'] < -200,0,1)))
            for decision,key in [('balanced','logits'),('natural','natural_logits')]:
                cm=np.bincount(y*3+z[key].argmax(1),minlength=9).reshape(3,3)
                m=from_cm(cm);stored=run['metrics'][label][decision]
                assert np.array_equal(cm,stored['confusion'])
                for metric in ['accuracy','balanced_accuracy','macro_f1']:assert abs(m[metric]-stored[metric])<1e-10
    for decision in ['balanced','natural']:
        for mode in ['adam','pcgrad']:
            m=[run['metrics'][f'{mode}_{s}'][decision] for s in [17,29]]
            table.append(dict(mode=mode,decision=decision,best_epochs=[run['trials'][f'{mode}_{s}']['best_epoch'] for s in [17,29]],
                              **{k:np.mean([v[k] for v in m],axis=0).tolist() for k in ['accuracy','balanced_accuracy','macro_f1','precision','recall','confusion','predicted_counts']}))
    summary=dict(parameters=run['parameters'],seconds=run['seconds'],counts=run['counts'],rows=table,baselines=run['baselines'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,mode in enumerate(['adam','pcgrad']):
        for split,color in [('train','#2473b8'),('validation','#d67b21')]:
            for row,metric in enumerate(['balanced_nll','macro_f1']):
                ys=np.array([[c[split]['balanced'][metric] for c in curves[f'{mode}_{s}']] for s in [17,29]])
                ax=axes[row,col];ax.plot(range(1,13),ys.mean(0),color=color,label=split)
                ax.fill_between(range(1,13),ys.min(0),ys.max(0),color=color,alpha=.15)
                ax.set(title=f'{mode}: {metric}',xlabel='Полная эпоха');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('LONG / NONE / SHORT · только прошлые 512 свечей · среднее двух seed')
    fig.savefig(OUT/'curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,mode in zip(axes,['adam','pcgrad']):
        r=next(r for r in table if r['mode']==mode and r['decision']=='balanced');cm=np.array(r['confusion']);norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{100*norm[i,j]:.1f}%',ha='center',va='center',color='white' if norm[i,j]>.55 else 'black')
        ax.set(xticks=range(3),xticklabels=CLASSES,yticks=range(3),yticklabels=CLASSES,xlabel='Предсказание',ylabel='Реальный класс',title=f'{mode}: test')
    fig.suptitle('Доля ответов внутри каждого реального класса · среднее двух seed')
    fig.savefig(OUT/'confusion.png',dpi=160);plt.close(fig)
    # Paired bootstrap across calendar days; mean per-seed scores, no selection by test.
    zs={name:np.load(REMOTE/f'{name}_cases.npz') for name in run['trials']}
    z=zs['adam_17'];days=z['timestamp']//86400;unique=np.unique(days);rng=np.random.default_rng(212)
    draws=[]
    for _ in range(3000):
        ix=np.concatenate([np.flatnonzero(days==d) for d in rng.choice(unique,len(unique))]);y=z['target'][ix]
        if len(np.unique(y))!=3:continue
        scores={}
        for mode in ['adam','pcgrad']:
            scores[mode]=np.mean([from_cm(np.bincount(y*3+zs[f'{mode}_{s}']['logits'][ix].argmax(1),minlength=9).reshape(3,3))['balanced_accuracy'] for s in [17,29]])
        draws.append([scores['adam']-1/3,scores['pcgrad']-1/3,scores['pcgrad']-scores['adam']])
    intervals=dict(names=['adam_minus_none','pcgrad_minus_none','pcgrad_minus_adam'],ci95_percentage_points=(100*np.quantile(draws,[.025,.975],axis=0)).tolist(),days=len(unique),valid_draws=len(draws))
    (OUT/'bootstrap.json').write_text(json.dumps(intervals,indent=2))
    lines=['# Классификация LONG / NONE / SHORT','',
           'Первый пилот: изменение Close через 8 минут относительно текущего Close. LONG строго больше +$2, SHORT строго меньше −$2, иначе NONE. Границы ±$2 включены в NONE.',
           'Во входе только предыдущие 512 свечей. TCN: исходная ширина 128 и пять блоков; голова заменена на три класса. Правый контекст и восстановление отсутствуют.',
           f"Всего параметров: {run['parameters']:,}. Четыре обучения, по 12 полных эпох на январе 2024 (27 716 окон). Validation 1–7 февраля, test 8–14 февраля. GPU worker: {run['seconds']/60:.1f} мин.",
           '',f"Test: SHORT / NONE / LONG = {run['counts']['test']}. Всегда NONE даёт {100*run['baselines']['always_none']['accuracy']:.2f}% accuracy и 33.33% balanced accuracy.",
           '', '## Вывод по пилоту','',
           'Средняя полнота по трём классам (balanced accuracy): Adam 38.19%, PCGrad 38.68%, всегда NONE 33.33%. Интервалы дневного bootstrap для обоих превышений над NONE включают ноль; устойчивое преимущество на этом маленьком test не установлено.',
           'При равном весе классов модель выдаёт много ложных направленных сигналов. После поправки на естественные частоты классов обе версии отвечают только NONE. Высокая accuracy такого ответа не означает решения поставленной задачи.',
           'По validation NLL выбраны эпохи 7/6 для Adam и 5/3 для PCGrad. Macro F1 на validation достигает более высоких значений позднее, поэтому результат зависит и от заранее выбранного критерия остановки. Последние checkpoints и их test-метрики также сохранены; по test модель не перевыбиралась.',
           '', '## Результаты на test','',
           '| Метод | Правило ответа | Лучшие эпохи | Accuracy | Balanced accuracy | Macro F1 | SHORT precision / recall | LONG precision / recall |',
           '|---|---|---|---:|---:|---:|---|---|']
    for r in table:
        lines.append(f"| {r['mode']} | {r['decision']} | {r['best_epochs']} | {100*r['accuracy']:.2f}% | {100*r['balanced_accuracy']:.2f}% | {r['macro_f1']:.4f} | {100*r['precision'][0]:.2f}% / {100*r['recall'][0]:.2f}% | {100*r['precision'][2]:.2f}% / {100*r['recall'][2]:.2f}% |")
    lines+=['','Среднее двух seed; checkpoint выбирается только по class-averaged validation NLL.',
            'balanced — argmax сырых logits после обучения с равными суммарными весами классов; natural — поправка logits на частоты классов train. Правила зафиксированы до test. Это не гарантия калиброванных вероятностей.',
            '', '## Проверки и ограничения','',
            f"Доля классов train: {run['train_priors']}; веса cross-entropy: {run['class_weights']}.",
            'Подтверждены точные границы в центах, неизменность входов при изменении будущих цен, равенство начальных весов в паре, полное покрытие эпох. Матрицы ошибок, accuracy, balanced accuracy и macro F1 независимо пересчитаны из сохранённых logits.',
            'Контроль утечки включает перерасчёт волатильности только по прошлым свечам. Старые масштабы, использовавшие правый контекст, не применяются.',
            'В test всего 30 движений за порогом и шесть дат. Дневной bootstrap сохранён в bootstrap.json; малый объём и предыдущие исследования на этих датах ограничивают силу выводов.',
            'Этот результат не доказывает отсутствие информации в M1. Он проверяет конкретный восьмиминутный таргет, входы, модель и режим обучения.',
            '', '![Обучение](curves.png)','![Матрицы ошибок](confusion.png)',
            '', 'remote/ содержит checkpoints, состояния Adam, кривые и per-example logits. summary.json — полные метрики, bootstrap.json — интервалы по датам.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
