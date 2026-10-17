from pathlib import Path
import json,numpy as np,pandas as pd,hashlib,zipfile
P=Path('/workspace/xauusd/reports/meta_reward_transformer_20261015');R=P/'remote/results'
lines=['# Transformer-учитель обучаемой награды: первый эксперимент\n']
def add(s=''):lines.append(s)
config=json.loads((R/'run_config.json').read_text()) if (R/'run_config.json').exists() else {}
if (R/'error.txt').exists():add('**Прогон завершился ошибкой. Результат обучения не подтверждён.**\n\n'+(R/'error.txt').read_text()+'\n')
proto=json.loads((R/'dataset_protocol.json').read_text()) if (R/'dataset_protocol.json').exists() else {}
primary=json.loads((R/'primary_complete.json').read_text()) if (R/'primary_complete.json').exists() else {}
add('Учитель и ученики — отдельные иерархические Transformer: 6 временных +6 межрыночных блоков, hidden192,6heads,dropout.25. Учитель получает разрешённые120мин будущего золота, ученик — только120мин прошлого всех222инструментов. Выход учителя — непрерывная Beta-политика обучающих оценок, не вероятность прибыльности. Внешнее обучение учителя — score-function policy gradient от изменения PNL ученика на других датах; это не классический метаградиент LIRPG и не гарантированная оценка физической предсказуемости точки.\n')
if proto:
 add('Матрица: '+str(proto['source']['shape'])+' FP16;48признаков на инструмент +8часовых. Все3,505,124M1моменты2016–2026подготовлены. Разделы: '+str(proto['counts'])+'. 2016–2022 внутреннее обучение,2023meta-feedback,2024checkpointselection,2025–2026ретроспективныйfinal. Исторические годы ранее исследовались, поэтому это не полностью слепая оценка.\n')
if primary:add('Фактическое обучение учителя: '+str(primary)+'. Один комбинированный проход не равен полному проходу каждого из3учеников.\n')
add('Только открытия WAIT/LONG/SHORT. Временный выход фиксирован: до120мин или границаUTCдня; при отсутствии котировки используется следующая доступная, возможны переносы и они учитываются в отчёте. Объём фиксирован, PNL $/унцию. Сценарий расходов $0.25/сделку, bid/ask неизвестны. Mean по всем пригодным дням включает дни без сделок с0. Внешняя цель — прирост доходности; одинаковый ежедневный рост не был отдельной оптимизируемой гарантией.\n')
fr=json.loads((R/'fresh_results.json').read_text()) if (R/'fresh_results.json').exists() else {}
add('| Новый ученик | Примеров обучения | Дней test | Сделок test | Net $/день | Gross $/сделку | Net $/сделку |\n|---|---:|---:|---:|---:|---:|---:|')
for name,res in fr.items():
 t=res['metrics']['test'];g=t.get('gross_mean_per_trade');n=t.get('mean_net_per_trade');add(f"| {name} | {res['train_examples']} | {t['days']} | {t['trades']} | {t['mean_net_dollars_per_day']:+.4f} | {g if g is not None else '—'} | {n if n is not None else '—'} |")
add('\nСвежие ученики имеют одинаковое начальное состояние, одинаковые обучающие точки и бюджет обновлений. Сравнение только на одном новом seed, поэтому перенос между инициализациями ещё ограниченно проверен. Если отдельные arms не завершились, нельзя считать сравнение полным.\n')
daily_cache={}
for name in fr:
 path=R/f'{name}_test_daily.npz'
 if path.exists():
  v=np.load(path);days,pnl=v['day'],v['pnl']
 else:
  idx=np.load(R/f'{name}_test.npz')['rows'];raw=pd.read_parquet('/workspace/xauusd/reports/intermarket_lightgbm_20261015/samples/XAUUSD_41.parquet',columns=['timestamp']);ts=pd.DatetimeIndex(raw.timestamp.sort_values()).as_unit('s').asi8;days=np.unique(ts[idx]//86400);pnl=np.zeros(len(days))
 daily_cache[name]=(days,pnl);equity=np.r_[0,np.cumsum(pnl)];dd=np.maximum.accumulate(equity)-equity;positive=np.maximum(pnl,0);share=float(np.sort(positive)[-5:].sum()/positive.sum()) if positive.sum()>0 else None
 add(f"Стабильность {name}: максимальная просадка накопленного PNL {dd.max():.3f}$; доля пяти самых прибыльных дней в сумме положительных дневных PNL {share}.\n")
 dates=pd.to_datetime(days*86400,unit='s',utc=True);frame=pd.DataFrame({'date_UTC':dates,'net_pnl_dollars':pnl});frame.to_csv(P/f'{name}_test_daily.csv',index=False)
if len(fr)==3:
 days,pnl=daily_cache['learned_teacher'];rng=np.random.default_rng(177)
 for baseline in ['direct_money','untrained_teacher']:
  bd,bp=daily_cache[baseline];assert np.array_equal(days,bd)
  delta=pnl-bp;u,ix=np.unique(days//7,return_inverse=True);sums=np.bincount(ix,weights=delta);counts=np.bincount(ix);samples=rng.integers(0,len(u),size=(1000,len(u)));ci=np.quantile(sums[samples].sum(1)/counts[samples].sum(1),[.025,.975]);add(f"Добавка учителя относительно {baseline}: {delta.mean():+.4f} $/день,95%интервал7-дневными блоками [{ci[0]:+.4f};{ci[1]:+.4f}].\n")
add('Полные метрики, дни, переносы сделок, кривые внешней награды, entropy и градиенты учителя сохранены в JSON/NPZ. Ненулевой градиент или успешная искусственная задача не доказывают полезности учителя на рынке. Оценки учителя не используются как доступные при живой торговле входы ученика.\n')
add('Крупная матрица признаков — runtime-кэш, не включена в компактный архив. Все исходные файлы пользователя сохранены; схема и точный код позволяют воспроизведение. Снимок включает фактические исходники runtime с SHA256, веса/оптимизаторы/RNG и проверки.\n')
if (P/'final_usage.txt').exists():add('Colab после выгрузки результатов остановлен.\n\n'+(P/'final_usage.txt').read_text())
(P/'REPORT_RU.md').write_text('\n'.join(lines));print('Russian report ready')
