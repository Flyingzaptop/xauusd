from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/workspace/xauusd/reports/meta_reward_transformer_v3_20261016');R=P/'remote/results'
curves=json.loads((R/'learning_curves.json').read_text()) if (R/'learning_curves.json').exists() else []
plt.rcParams.update({'font.family':'DejaVu Sans','axes.grid':True,'grid.alpha':.2})
if curves:
 x=[r['cycle'] for r in curves];dev=[r for r in curves if 'development' in r]
 fig,a=plt.subplots(3,1,figsize=(12,10),sharex=True)
 a[0].plot(x,[r['teacher_policy_train_loss'] for r in curves]);a[0].set_ylabel('Policy train loss');a[0].set_title('REINFORCE surrogate: не ошибка предсказания и не сравнивается напрямую с validation objective')
 a[1].plot(x,[r['teacher_external_reward'] for r in curves],alpha=.65);a[1].axhline(0,color='black',lw=.7);a[1].set_ylabel('Внешняя награда ΔU')
 a[2].plot([r['cycle'] for r in dev],[-r['development']['utility_per_day'] for r in dev],'o-');a[2].set_ylabel('Validation objective −U');a[2].set_xlabel('Внешний цикл');a[2].set_title('Четыре фиксированных окна по28дней; критерий — худший excess; меньше −U лучше');fig.tight_layout();fig.savefig(P/'teacher_diagnostics.png',dpi=160);plt.close(fig)
 fig,a=plt.subplots(4,1,figsize=(12,12),sharex=True)
 for si in range(3):
  d=[r for r in dev if r['student']==si];xx=[r['cycle'] for r in d];a[0].plot(xx,[r['development']['net_total'] for r in d],'o-',label=f'Ученик {si+1}');a[1].plot(xx,[r['development']['trades'] for r in d],'o-');q=[r for r in curves if r['student']==si];a[2].plot([r['cycle'] for r in q],[r['student_loss_components']['entropy'] for r in q]);a[3].plot([r['cycle'] for r in q],[r['student_loss_components']['entropy_alpha'] for r in q])
 a[0].set_ylabel('Net $ за 4×28 дней');a[0].legend();a[1].set_ylabel('Сделки за 4×28 дней');a[2].set_ylabel('Энтропия');a[3].set_ylabel('Коэффициент alpha');a[3].set_xlabel('Внешний цикл');fig.tight_layout();fig.savefig(P/'actor_behavior.png',dpi=160);plt.close(fig)
fresh=json.loads((R/'fresh_results.json').read_text()) if (R/'fresh_results.json').exists() else {}
controls=json.loads((R/'test_static_controls.json').read_text()) if (R/'test_static_controls.json').exists() else {}
lines=['# Итерация v3','', 'Warm-start teacher v2cycle80; направления rewardцентрированы по4096trainточкам. Энтропия управляется фиксированным плавным расписанием.','', 'Test:2025-04-07..2025-06-30,84календарныхдня. Позиция1унция,условныеиздержки0.25$/сделку. Тест новый: абсолютныеPNL нельзя напрямую сравнивать с январём–мартомv2. Дневнаяequity учитывает закрытуюприбыль по дате входа, не внутрисделочный mark-to-market риск.','']
series={}
if fresh:
 fig,ax=plt.subplots(figsize=(13,6));lines+=['| Модель / seed | Net $ | LONG | SHORT | Просадка дневной equity $ |','|---|---:|---:|---:|---:|']
 for key,v in fresh.items():
  m=v['metrics']['test'];series[key]=np.asarray(m['daily_net']);color={'learned_teacher':'tab:blue','previous_teacher':'tab:green','direct_money':'tab:orange'}[v['mode']]
  ax.plot(np.array(m['daily_dates'],dtype='datetime64[D]'),np.cumsum(series[key]),color=color,ls='-' if v['seed']==888 else ':',label=key)
  lines.append(f"| {key} | {m['net_total']:.2f} | {m['long_trades']} | {m['short_trades']} | {m['max_daily_drawdown']:.2f} |")
 for key,m in controls.items():
  lines.append(f"| always_{key} | {m['net_total']:.2f} | {m['long_trades']} | {m['short_trades']} | {m['max_daily_drawdown']:.2f} |")
  if key!='flat':ax.plot(np.array(m['daily_dates'],dtype='datetime64[D]'),np.cumsum(m['daily_net']),ls='--',lw=1,color='black' if key=='long' else 'gray',label='always_'+key)
 ax.legend(ncol=2);ax.set_ylabel('Накопленный net PnL, $');ax.set_title('Новый finaltest:12непрерывных недель,2seed');fig.tight_layout();fig.savefig(P/'test_equity.png',dpi=160);plt.close(fig)
 for key,v in fresh.items():
  m=v['metrics']['test'];lines+=['',key+': недельный netPnL '+', '.join(f'{x:.2f}' for x in m['weekly_net']), 'Средний $/день при cost0/.1/.25/.5/1: '+', '.join(f"{(m['net_total']+(.25-c)*m['trades'])/84:.3f}" for c in [0,.1,.25,.5,1])]
 rng=np.random.default_rng(777)
 for baseline in ['previous_teacher','direct_money']:
  pairs=[(series[f'learned_teacher_seed{s}']-series[f'{baseline}_seed{s}']).reshape(12,7).sum(1) for s in [888,889] if f'learned_teacher_seed{s}' in series and f'{baseline}_seed{s}' in series]
  if pairs:
   delta=np.stack(pairs);si=rng.integers(0,len(pairs),(10000,len(pairs)));wi=rng.integers(0,12,(10000,12));boot=delta[si[:,:,None],wi[:,None,:]].mean((1,2))/7;lo,hi=np.quantile(boot,[.025,.975]);lines+=['',f'Новый − {baseline}: средняя разница{delta.mean()/7:.3f}$/день, paired seed/week bootstrap95%[{lo:.3f};{hi:.3f}], seeds={len(pairs)}. Дваseed и12недель дают ограниченную уверенность.']
else:lines+=['Нет завершённого freshсравнения. Не делать вывод о торговом преимуществе из meta/dev.']
for name in ['primary_complete.json','initial_reward_calibration.json','frozen_reward_calibration.json']:
 if (R/name).exists():lines+=['',name, '```json',(R/name).read_text(),'```']
if (R/'error.txt').exists():lines+=['','Ошибка исполнения:','```',(R/'error.txt').read_text(),'```']
(P/'REPORT_RU.md').write_text('\n'.join(lines));print(P/'REPORT_RU.md')
