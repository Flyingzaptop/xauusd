from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;r=json.loads((P/'results.json').read_text());a=np.load(P/'randomization_runs.npz');labels={'constant_LONG_probability':'Constant LONG probability','global_shuffle':'Shuffle all minutes','within_day_shuffle':'Shuffle within each day','within_day_circular_shift':'Shift within each day'}
fig,ax=plt.subplots(1,2,figsize=(13,4.5),layout='constrained');n=len(a['actual_daily']);x=np.arange(n)
ax[0].plot(x,np.cumsum(a['actual_daily']),color='black',lw=2,label='V2 actual entries')
for k,v in labels.items():
 eq=np.cumsum(a[k+'_daily'],axis=1);line=ax[0].plot(x,eq.mean(0),label=v,lw=1)[0]
 if k=='within_day_shuffle':
  q=np.quantile(eq,[.025,.975],axis=0);ax[0].fill_between(x,*q,color=line.get_color(),alpha=.15,label='95% random-path range')
ax[0].set(xlabel='Calendar day',ylabel='Closed net P&L, $ / ounce',title='Actual vs randomized LONG timing');ax[0].legend(fontsize=7)
for j,(k,label) in enumerate(labels.items()):
 c=r['controls'][k];lo,hi=c['profit_95pct_range'];m=c['mean_profit'];ax[1].errorbar(m,j,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4)
ax[1].axvline(r['actual']['profit'],color='black',label='Actual +$231.67');ax[1].set_yticks(range(4),list(labels.values()),fontsize=8);ax[1].set(xlabel='Total net P&L, $ / ounce',title='Mean and central 95% of 2,000 runs');ax[1].legend(fontsize=8)
fig.savefig(P/'entry_timing_comparison.png',dpi=160)
lines=['# V2: проверка выбора LONG-входов','', 'Все 720 исходных сделок и их P&L воспроизведены точно. Симуляции CPU, GPU обучения не использовалась.','',f"Фактический результат: ${r['actual']['profit']:.2f}; {r['actual']['trades']} сделок. Всегда LONG: ${r['always_long']['profit']:.2f}; {r['always_long']['trades']} сделок.",'','| Контроль | Средняя прибыль | 95% диапазон симуляций | Доля не хуже ученика |','|---|---:|---:|---:|']
for k,label in labels.items():
 c=r['controls'][k];lo,hi=c['profit_95pct_range'];lines.append(f"| {label} | ${c['mean_profit']:.2f} | ${lo:.2f} … ${hi:.2f} | {100*c['fraction_random_at_least_actual']:.1f}% |")
lines+=['','На этом тесте прибыльность выбора момента не подтверждена: фактическая прибыль находится внутри распределения случайных входов. Это не доказывает отсутствие сигнала; текущих результатов недостаточно, чтобы отличить его от случайности.','', 'Во всех контролях: только LONG, одна позиция, прежний выход 120 минут/граница UTC, прежние цены исполнения, $0.25 издержек. Внутридневное перемешивание сохраняет число LONG-сигналов каждого дня; сдвиг сохраняет их внутридневную последовательность с циклическим переносом. Ни один контроль не выбирается по будущему P&L. Сохранено число сигналов, а не точное число исполненных сделок: блокировка открытой позицией различает их.','', 'Недельный bootstrap разницы фактической прибыли и среднего контроля дан в results.json; интервалы, включающие ноль, не подтверждают устойчивость преимущества. Один уже просмотренный тест, один ученик, приближённый учёт зависимости недель. Это диагностический анализ, а не новая независимая проверка.','', 'Полные поминутные вероятности не сохранены; проверены реализованные действия, а не причинная зависимость вероятностей от контекста.']
(P/'REPORT_RU.md').write_text('\n'.join(lines))
