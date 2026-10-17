from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/workspace/xauusd/reports/meta_reward_transformer_20261015')
r=json.loads((P/'learning_curves_live.json').read_text()); x=np.array([v['cycle'] for v in r]); dev=[v for v in r if 'development' in v]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.22})
def finish(fig,name):
 fig.savefig(P/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
fig,a=plt.subplots(3,1,figsize=(12,10),sharex=True)
g=np.array([v['teacher_external_reward'] for v in r]);a[0].plot(x,g,alpha=.35,label='Награда учителя: прирост PnL ученика, $/день')
a[0].plot(x[7:],np.convolve(g,np.ones(8)/8,mode='valid'),lw=2,label='Среднее за 8 циклов');a[0].axhline(0,color='black',lw=.7);a[0].legend();a[0].set_ylabel('Δ PnL, $/день')
a[1].plot([v['cycle'] for v in dev],[v['development']['mean_net_dollars_per_day'] for v in dev],'o-');a[1].axhline(0,color='black',lw=.7);a[1].set_ylabel('PnL ученика, $/день');a[1].set_title('Проверка: те же 12 выбранных дней 2024 года; это не validation loss учителя')
a[2].plot(x,[v['teacher_gradient_norm'] for v in r],color='tab:purple');a[2].set_ylabel('Норма градиента\nдо clipping');a[2].set_xlabel('Внешний RL-цикл (не эпоха)')
fig.suptitle('Учитель-критик: доступная диагностика\nTrain loss и validation loss учителя НЕ сохранены',fontsize=14);fig.tight_layout(rect=(0,0,1,.94));finish(fig,'teacher_diagnostics')
fig,a=plt.subplots(3,1,figsize=(12,10),sharex=True)
for si in range(3):
 d=[v for v in dev if v['student']==si];xx=[v['cycle'] for v in d]
 a[0].plot(xx,[v['development']['mean_net_dollars_per_day'] for v in d],'o-',label=f'Ученик {si+1}')
 a[1].plot(xx,[v['development']['trades'] for v in d],'o-',label=f'Ученик {si+1}')
 q=[v for v in r if v['student']==si];a[2].plot([v['cycle'] for v in q],[v['student_entropy'] for v in q],label=f'Ученик {si+1}')
a[0].axhline(0,color='black',lw=.7);a[0].set_ylabel('Net PnL, $/день');a[0].legend(ncol=3);a[0].set_title('Одинаковые 12 проверочных дней; стоимость $0.25 на сделку за унцию')
a[1].set_ylabel('Сделки за ВСЕ 12 дней');a[1].annotate('147 сделок',xy=(1,147),xytext=(8,130),arrowprops={'arrowstyle':'->'});a[1].annotate('3 сделки',xy=(96,3),xytext=(79,40),arrowprops={'arrowstyle':'->'})
a[2].set_ylabel('Энтропия действий');a[2].set_xlabel('Внешний RL-цикл (три ученика обучаются по очереди)');a[2].set_title('Снижение к нулю: политика почти всегда выбирает одно действие; по логам — WAIT')
fig.suptitle('Актёр: прибыль, активность и поведение по ходу обучения',fontsize=14);fig.tight_layout(rect=(0,0,1,.95));finish(fig,'actor_behavior')
with (P/'actor_validation_history.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['cycle','student','validation_days','net_usd_per_day','net_usd_total','trades_total','trades_per_day'])
 for v in dev:
  d=v['development'];w.writerow([v['cycle'],v['student'],d['days'],d['mean_net_dollars_per_day'],d['mean_net_dollars_per_day']*d['days'],d['trades'],d['trades']/d['days']])
print(json.dumps({'cycles':len(r),'validation_points':len(dev),'first':dev[0]['development'],'best':max(dev,key=lambda v:v['development']['mean_net_dollars_per_day'])['development'],'last':dev[-1]['development']},indent=2))
