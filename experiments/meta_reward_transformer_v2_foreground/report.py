from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/workspace/xauusd/reports/meta_reward_transformer_v2_foreground_20261015');R=P/'remote/results'
curves=json.loads((R/'learning_curves.json').read_text()) if (R/'learning_curves.json').exists() else []
plt.rcParams.update({'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':.2})
if curves:
 x=[r['cycle'] for r in curves];dev=[r for r in curves if 'development' in r]
 fig,a=plt.subplots(3,1,figsize=(12,10),sharex=True)
 a[0].plot(x,[r['teacher_policy_train_loss'] for r in curves]);a[0].set_ylabel('Policy train loss');a[0].set_title('REINFORCE surrogate: не ошибка предсказания и не сравнивается напрямую с validation objective')
 a[1].plot(x,[r['teacher_external_reward'] for r in curves],alpha=.65);a[1].axhline(0,color='black',lw=.7);a[1].set_ylabel('Внешняя награда ΔU')
 a[2].plot([r['cycle'] for r in dev],[-r['development']['utility_per_day'] for r in dev],'o-');a[2].set_ylabel('Validation objective −U');a[2].set_xlabel('Внешний цикл');a[2].set_title('Фиксированные непрерывные 28 дней; меньше −U лучше');fig.tight_layout();fig.savefig(P/'teacher_diagnostics.png',dpi=160);plt.close(fig)
 fig,a=plt.subplots(4,1,figsize=(12,12),sharex=True)
 for si in range(3):
  d=[r for r in dev if r['student']==si];xx=[r['cycle'] for r in d];a[0].plot(xx,[r['development']['net_total'] for r in d],'o-',label=f'Ученик {si+1}');a[1].plot(xx,[r['development']['trades'] for r in d],'o-');q=[r for r in curves if r['student']==si];a[2].plot([r['cycle'] for r in q],[r['student_loss_components']['entropy'] for r in q]);a[3].plot([r['cycle'] for r in q],[r['student_loss_components']['entropy_alpha'] for r in q])
 a[0].set_ylabel('Net $ за 28 дней');a[0].legend();a[1].set_ylabel('Сделки за 28 дней');a[2].set_ylabel('Энтропия');a[2].axhline(.5*np.log(3),ls='--',color='black');a[3].set_ylabel('Коэффициент alpha');a[3].set_xlabel('Внешний цикл');fig.tight_layout();fig.savefig(P/'actor_behavior.png',dpi=160);plt.close(fig)
fresh=json.loads((R/'fresh_results.json').read_text()) if (R/'fresh_results.json').exists() else {}
lines=['# Повторный запуск v2','', 'Единица позиции — одна унция XAUUSD; издержки сценарные $0.25/сделку. Абсолютный PnL не является процентной доходностью счёта.','', 'Сравнение: одинаковый seed/число обновлений; test 84 последовательных календарных дня. Сохранение исследования не гарантирует торгового преимущества.','']
if fresh:
 fig,ax=plt.subplots(figsize=(12,5))
 lines+=['| Модель | Net $ | Сделок | Просадка дневной equity, $ | Эффективных прибыльных сделок |','|---|---:|---:|---:|---:|']
 series={}
 for name,v in fresh.items():
  m=v['metrics']['test'];daily=np.asarray(m['daily_net']);series[name]=daily;ax.plot(np.array(m['daily_dates'],dtype='datetime64[D]'),np.cumsum(daily),label=name)
  lines.append(f"| {name} | {m['net_total']:.2f} | {m['trades']} | {m['max_daily_drawdown']:.2f} | {m['effective_profitable_trades']:.1f} |")
  lines+=['',f"{name}: недельный PnL: "+', '.join(f'{v:.2f}' for v in m['weekly_net']),f"Средний PnL/день при издержках 0/.1/.25/.5/1: "+', '.join(f"{(m['net_total']+(.25-c)*m['trades'])/84:.3f}" for c in [0,.1,.25,.5,1]),'']
 ax.legend();ax.set_ylabel('Накопленный net PnL, $');ax.set_title('Итоговый непрерывный тест: 12 недель');fig.tight_layout();fig.savefig(P/'test_equity.png',dpi=160);plt.close(fig)
 rng=np.random.default_rng(777)
 if 'learned_teacher' in series:
  for baseline in ['direct_money','untrained_teacher']:
   if baseline in series:
    delta=(series['learned_teacher']-series[baseline]).reshape(12,7).sum(1);draw=rng.choice(delta,(10000,12),replace=True).mean(1)/7;lo,hi=np.quantile(draw,[.025,.975]);lines.append(f'Learned − {baseline}: средняя разница $/день {delta.mean()/7:.3f}; 95% недельный bootstrap [{lo:.3f}; {hi:.3f}]. Один seed и 12 недель недостаточны для окончательного вывода.')
else:lines+=['Нет завершённого сравнения fresh-учеников. Не выдавать промежуточную meta/dev прибыль за итог теста.']
if (R/'primary_complete.json').exists():lines+=['','Фактическое покрытие/причина остановки:','```json',(R/'primary_complete.json').read_text(),'```']
if (R/'error.txt').exists():lines+=['','Ошибка исполнения:','```',(R/'error.txt').read_text(),'```']
(P/'REPORT_RU.md').write_text('\n'.join(lines));print(P/'REPORT_RU.md')
