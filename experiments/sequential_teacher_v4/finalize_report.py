from pathlib import Path
import json,hashlib,time,subprocess,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gates import source_hashes,assess
from provenance import COMMON,fingerprint,current
P=Path('/workspace/xauusd/reports/sequential_teacher_v4_20261016')
s=json.loads((P/'synthetic_gates.json').read_text());m=json.loads((P/'meta_gate.json').read_text())
check=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent),'-p','test_*.py','-v'],capture_output=True,text=True);(P/'tests.log').write_text(check.stdout+check.stderr)
mechanics={'code_hashes':fingerprint(COMMON+['real_data.py','meta_gradient.py','test_environment.py','test_learning.py','test_meta_gradient.py']),'complete':True,'passed':check.returncode==0,'tests':16,'command':'python -m unittest discover -s experiments/sequential_teacher_v4 -p test_*.py -v','coverage':['execution transition and blocking','no zero-duration or overnight extension','missing exit cannot alter past legal actions','mark-to-market drawdown','duration-aware discount/GAE','actual actor and teacher gradients','RNG independence','real-panel observation excludes current bar']}
(P/'mechanics.json').write_text(json.dumps(mechanics,indent=2))
manifest={'created_epoch':time.time(),'source_hashes':source_hashes(),'untouched_test_protocol':None}
for name,file in [('mechanics','mechanics.json'),('synthetic','synthetic_gates.json'),('meta','meta_gate.json')]:
    data=(P/file).read_bytes();x=json.loads(data);manifest[name]={'passed':x.get('passed') is True and current(x),'file':file,'sha256':hashlib.sha256(data).hexdigest()}
variants={task:json.loads((P/('meta_gradient_'+task)/'meta_gradient_gate.json').read_text()) for task in ['direction','wait','cross_market']}
mg={'code_hashes':variants['direction']['code_hashes'],'complete':True,'passed':all(v['passed'] and current(v) for v in variants.values()),'method':'exact expected-return meta-gradient, a different learner from PPO','variants':variants}
(P/'meta_gradient_suite.json').write_text(json.dumps(mg,indent=2));data=(P/'meta_gradient_suite.json').read_bytes();manifest['meta_gradient']={'passed':mg['passed'],'file':'meta_gradient_suite.json','sha256':hashlib.sha256(data).hexdigest()}
manifest['real_pilot']={'passed':False,'reason':'Meta-teacher transfer gate failed; no market training or new GPU allocation'}
(P/'gate_manifest.json').write_text(json.dumps(manifest,indent=2));gate=assess(P);(P/'launch_decision.json').write_text(json.dumps(gate,indent=2))
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for i,task in enumerate(['direction','wait','cross_market']):
    y=[r['oracle_ratio'] for r in s['results'] if r['task']==task];axs[0].scatter([i]*len(y),y)
axs[0].set_xticks(range(3),['Known direction','Wait for better entry','Cross-instrument signal']);axs[0].set(ylabel='P&L / hindsight oracle P&L',ylim=(0,1.05),title='Sequential student: unseen synthetic episodes')
for i,(name,v) in enumerate(m['results'].items()):
    axs[1].scatter([i]*len(v),v,color='tab:blue');axs[1].plot([i-.2,i+.2],[np.mean(v)]*2,color='black')
axs[1].set_xticks(range(len(m['results'])),list(m['results']),rotation=20);axs[1].axhline(0,color='grey',lw=.5);axs[1].set(ylabel='Net P&L per synthetic episode',title='Teacher transfer: three fresh student seeds')
fig.savefig(P/'gates.png',dpi=160)
rows=[]
for task in ['direction','wait','cross_market','null']:
    rs=[r for r in s['results'] if r['task']==task];rows.append(f"| {task} | {sum(r['passed'] for r in rs)}/{len(rs)} | {min(r['mean_net'] for r in rs):.3f} … {max(r['mean_net'] for r in rs):.3f} |")
lines=['# V4: реализованные изменения и результаты проверок','','GPU не запускалась. Исторические v2/v3 сохранены отдельно. Реализация: experiments/sequential_teacher_v4.','','## Готово','','- Единая последовательная среда train/eval, fixed exit, position blocking, заранее объявленные сессии, явные неоценимые выходы, mark-to-market equity.','- Малый Transformer/PPO с временными переходами и GAE.','- Контролируемый outer REINFORCE учителя, отдельные RNG, credit выбранной стороны, реальное повторное обучение reset-учеников для отбора.','- Диагностика scores/future/past, risk sensitivity; causal real-panel builder для 222 parquet; guarded real-pilot runner.','- Проверки и SHA256-реестр результатов; большой запуск блокируется при непройденных gates.','','## Проверки','','16 механических/градиентных/каузальных тестов прошли. Полный synthetic suite: 4 задачи × 3 seeds.','', '| Задача | Успешные seed | Средний P&L по seed, диапазон |','|---|---:|---:|',*rows,'','Известное направление и совместный сигнал дают 99.5–99.9% hindsight optimum на новых простых эпизодах. Ожидание лучшего входа: 97.0–99.2%. Это специально простые положительные контроли, не прибыль на рынке.','', 'Первый запуск с необмасштабированной денежной value-loss не прошёл wait gate; его результаты сохранены в initial_unscaled_gates. Исправлены масштаб reward ($10 за единицу обучения, общий фиксированный масштаб) и бюджет smoke-test (100→250 updates). Поэтому улучшение нельзя целиком приписать только изменению масштаба. Отчётный P&L не нормирован.','','## Учитель не прошёл допуск','','| Награда | Средняя прибыль нового ученика |','|---|---:|']
for name,v in m['results'].items():lines.append(f'| {name} | {np.mean(v):.4f} |')
lines+=['', '24 outer cycles × 16 inner updates; это ограниченный CPU-тест. Новый учитель не превосходит денежную награду и постоянного учителя. На маленьком наборе нельзя сделать вывод о невозможности meta-RL, но расширять этот вариант до большого рынка оснований нет.','', '## Что намеренно не запущено / не заявляется завершённым','','- Полная реальная абляция старых v2/v3 с восстановлением 222×120 контекстов: оригинальный 75GB cache удалён вместе с GPU, нужны исходные данные и восстановление выбранных контекстов. Реализован diagnose_legacy.py; подмена настоящих входов синтетикой запрещена.','- Реальное обучение учителя с новым протоколом и сравнение 222 инструментов против золота. Текущий meta.py — синтетический исследовательский harness; для рыночного meta-обучения ещё нужен адаптер episode factory и validation protocol.','- Итоговый выбор risk utility и календаря брокера: зафиксирована денежная база и инструмент проверки чувствительности, универсального коэффициента не выдумано.','- Масштабирование и независимый итоговый тест: заблокированы meta-gate. Реальные модели/прибыль не заявляются.','','## Следующий шаг','','Исследовать outer teacher optimizer на этой дешёвой задаче, где student learning уже проверено: стабильность оценок, влияние будущего, дисперсия внешнего градиента, matched-budget constant/shuffled/money сравнения. Альтернатива для отдельного эксперимента — короткий дифференцируемый meta-gradient; не подменять им score-function RL без явного описания смены алгоритма.','', 'launch_decision.json фиксирует запрет рыночного пилота. Все состояния, маленькие модели, исходные результаты и график сохранены локально.']
lines+=['','## Дополнительная реализация: точный meta-gradient','','Добавлен meta_gradient.py: градиент внешней ожидаемой прибыли проходит через три шага обучения нового ученика. Используется точный policy gradient по конечной среде с фиксированным выходом и динамическим программированием, без меток правильного действия. Это отдельный алгоритм, не PPO и не score-function обновление учителя; перенос его teacher в PPO автоматически не разрешён. Производная проверена конечными разностями.','', '| Задача | Обученный учитель | Прямая денежная награда | Все контроли пройдены |','|---|---:|---:|---|']
for task,v in variants.items():lines.append(f"| {task} | {np.mean(v['results']['learned']):.3f} | {np.mean(v['results']['money']):.3f} | {v['passed']} |")
lines+=['','На направлении и ожидании найдено преимущество в коротком обучении трёх новых учеников. На совместном сигнале двух инструментов — нет. Проверка пройдена не полностью: exact_pilot также заблокирован. Это не рыночные деньги и не оценка полной сходимости после трёх шагов.','', 'Проверка реальных котировок XAUUSD в 9 заранее заданных UTC-окнах 08:00–18:00: пропусков нет; все 90 LONG/SHORT smoke-сделок держались 120 минут и закрывались внутри окна. Прибыльность модели этим не проверялась.']
fig2,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,(task,v) in zip(axes,variants.items()):
    names=['learned','constant','shuffled','money']
    for j,name in enumerate(names):
        values=v['results'][name];ax.scatter([j]*len(values),values);ax.plot([j-.2,j+.2],[np.mean(values)]*2,color='black')
    ax.axhline(0,color='grey',lw=.5);ax.set_xticks(range(4),names,rotation=25);ax.set_title(task+' — '+('PASS' if v['passed'] else 'FAIL'));ax.set_ylabel('Expected synthetic net return')
fig2.savefig(P/'meta_gradient_comparison.png',dpi=160)
(P/'REPORT_RU.md').write_text('\n'.join(lines));print(json.dumps(gate,ensure_ascii=False))
