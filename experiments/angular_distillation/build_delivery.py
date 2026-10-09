"""Assemble verified results into a compact Russian HTML delivery; no fitting."""
import base64,html,json,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/angular_distillation_20261008'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    verified=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    assert verified['status']=='passed'
    scores=pd.read_csv(OUT/'primary_scores_by_seed.csv')
    metrics=pd.read_csv(OUT/'metrics.csv')
    comp=pd.read_csv(OUT/'paired_comparisons.csv')
    summary=pd.read_csv(OUT/'summary.csv')
    zero=pd.read_csv(OUT/'selected_vs_zero.csv')
    selection=json.loads((OUT/'market_frozen_lambda.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    primary=[];by_horizon=[];monthly=[]
    for rep in ['log','angular']:
        for baseline in ['A','A_long']:
            row=comp[(comp.kind=='market')&(comp.split=='test')&(comp.representation==rep)&(comp.candidate=='C_selected')&(comp.baseline==baseline)&(comp.horizon==0)].iloc[0]
            base=scores[(scores.kind=='market')&(scores.split=='test')&(scores.representation==rep)&(scores.method==baseline)].primary_mse_bps2_per_minute.mean()
            primary.append(dict(representation=rep,comparison=f'C − {baseline}',improvement_pct=row.relative_improvement_pct,
                ci_low_pct=-row.ci_high/base*100,ci_high_pct=-row.ci_low/base*100))
        for h in [1,5,10,20]:
            row=comp[(comp.kind=='market')&(comp.split=='test')&(comp.representation==rep)&(comp.candidate=='C_selected')&(comp.baseline=='A')&(comp.horizon==h)].iloc[0]
            sm=summary[(summary.split=='test')&(summary.representation==rep)&(summary.horizon==h)]
            a=sm[sm.method=='A'].iloc[0];cc=sm[sm.method=='C_selected'].iloc[0];t=sm[sm.method=='T'].iloc[0]
            by_horizon.append(dict(representation=rep,horizon=h,A_mse=a.mse_bps2,C_mse=cc.mse_bps2,
                C_improvement_vs_A_pct=row.relative_improvement_pct,C_skill_vs_zero_pct=cc.skill_vs_zero*100,
                A_direction_pct=a.direction_accuracy*100,C_direction_pct=cc.direction_accuracy*100,
                T_skill_vs_zero_pct=t.skill_vs_zero*100))
        # Primary per-month and per-seed scores recomputed from stored predictions.
        folder=OUT/'predictions/market'/rep/'test';truth=np.load(folder/'truth.npz')
        y=truth['y_bps'].astype(float);months=pd.to_datetime(truth['timestamp'],unit='s',utc=True).strftime('%Y-%m').to_numpy()
        jj=np.array([0,4,9,19]);weights=np.array([1,5,10,20])
        for seed in [17,29,43]:
            a=np.load(folder/f'A_{seed}.npz')['p_bps'].astype(float)
            cc=np.load(folder/f'{selection[rep]["selected"]}_{seed}.npz')['p_bps'].astype(float)
            al=np.mean((a[:,jj,3]-y[:,jj,3])**2/weights,axis=1)
            cl=np.mean((cc[:,jj,3]-y[:,jj,3])**2/weights,axis=1)
            for month in np.unique(months):
                mask=months==month
                monthly.append(dict(representation=rep,seed=seed,month=month,A_loss=al[mask].mean(),C_loss=cl[mask].mean(),
                    improvement_pct=(1-cl[mask].mean()/al[mask].mean())*100))
    p=pd.DataFrame(primary);h=pd.DataFrame(by_horizon);mm=pd.DataFrame(monthly)
    p.to_csv(OUT/'delivery_primary.csv',index=False);h.to_csv(OUT/'delivery_horizons.csv',index=False);mm.to_csv(OUT/'monthly_primary_by_seed.csv',index=False)
    # Two representation comparisons comprise the main family. These 95% intervals
    # remain exploratory; the result text explicitly avoids a confirmatory claim.
    c_a=p[p.comparison=='C − A']
    improvement=bool((c_a.ci_low_pct>0).any())
    verdict=('Угловая дистилляция дала небольшой выигрыш относительно прямой модели, но преимущество над прогнозом «цена не изменится» не подтвердилось. Точность реконструкции teacher в полезный прогнозный сигнал здесь не перенеслась.' if improvement else
        'Устойчивого преимущества дистилляции C над прямым обучением A в этом прогоне не обнаружено.')
    title='XAUUSD: финальные результаты проверки teacher → student'
    def table(df):return df.to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)
    def image(name,caption):
        data=base64.b64encode((OUT/'plots'/name).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    synthetic=metrics[(metrics.kind!='market')&(metrics.period=='ALL')&(metrics.method.isin(['A','T','C_selected','bridge_oracle']))&(metrics.horizon.isin([1,20]))]
    synthetic=synthetic[['kind','method','horizon','skill_vs_zero','direction_accuracy']].copy()
    synthetic['skill_vs_zero']*=100;synthetic['direction_accuracy']*=100
    monthly_mean=mm.groupby(['representation','month'],as_index=False)[['A_loss','C_loss']].mean()
    monthly_mean['improvement_pct']=(1-monthly_mean.C_loss/monthly_mean.A_loss)*100
    stable=[]
    for rep in ['log','angular']:
        s=scores[(scores.kind=='market')&(scores.split=='test')&(scores.representation==rep)]
        a=s[s.method=='A'].set_index('seed').primary_mse_bps2_per_minute
        cc=s[s.method==selection[rep]['selected']].set_index('seed').primary_mse_bps2_per_minute
        months=monthly_mean[monthly_mean.representation==rep]
        stable.append(f'{rep}: выигрыш C у A в {int((cc<a).sum())}/3 seed и {int((months.improvement_pct>0).sum())}/8 месяцев')
    conv=pd.read_csv(OUT/'convergence_diagnostics.csv')
    late=conv[(conv.best_at_last_epoch)&(conv.method=='T')]
    tail=zero[(zero.kind=='market')&(zero.split=='tail')]
    body=f'''<h1>{title}</h1><p class="lead">{verdict}</p>
<div class="facts"><b>3 505 124 свечи M1</b> · 3 seed · 2 представления · 36 сохранённых рыночных вариантов<br>
Train: 2016–2024 · validation: 2025 · исследовательская оценка: январь–август 2026<br>
100 000 обучающих окон · 12 000 validation · 42 893 evaluation · 894 окна короткого сентябрьского хвоста</div>
<p>Это завершённый ограниченный эксперимент на предоставленном локальном наборе данных. Январь–август 2026 ранее использовались в проекте;
поэтому результаты являются исследовательскими, а не независимым подтверждением. В сентябре доступны лишь четыре дня.</p>
<h2>Главное сравнение: помогает ли teacher студенту?</h2>
<p>A — прямое обучение; A_long — вдвое больше обновлений; B — реконструкция и причинное дообучение; C — дистилляция;
T — teacher с правым контекстом. Вес C выбран только по validation, отдельно для представлений: {html.escape(str(selection))}.</p>
<p>Первичный критерий — средняя квадратная ошибка Close на горизонтах 1, 5, 10, 20 минут, делённая на соответствующий горизонт.
Это эквивалентно validation-критерию с точностью до общего постоянного масштаба. Положительный improvement означает пользу C.
Интервалы 95% получены парным bootstrap по дням; несколько сравнений и зависимость между днями ограничивают их интерпретацию.</p>
{table(p)}
<p>{html.escape('; '.join(stable))}. Устойчивость знака выигрыша не заменяет сравнение с нулевым прогнозом и оценку размера эффекта.</p>
{image('primary_comparisons.png','На графике показана разность потерь: отрицательное значение означает меньшую ошибку кандидата.')}
<h2>Качество по горизонтам</h2>
<p>MSE — квадрат базисного пункта логарифмической доходности. C_skill_vs_zero — уменьшение ошибки относительно «цена не изменится».
Результаты — средние потерь трёх seed; ансамбль прогнозов не строился. Direction — доля правильно предсказанных знаков ненулевого изменения.
Нулевой прогноз воздерживается от направления; его direction_accuracy=0 в сырой таблице является техническим соглашением, не показателем качества классификации.</p>
{table(h)}
{image('forecast_skill.png','Преимущество причинных моделей относительно нулевого изменения цены.')}
<h2>Устойчивость</h2>{table(monthly_mean)}
{image('monthly_stability.png','Месячная диагностика на 20-минутном горизонте.')}
<h2>Контроли: интерполяция и прогноз — разные задачи</h2>
<p>На случайном блуждании teacher видит будущее и может хорошо восстановить участок, хотя направление следующего независимого приращения непредсказуемо.
Положительный контроль AR(1) проверяет, что модель способна обнаружить существующую зависимость. Числа skill и direction ниже выражены в процентах.</p>
{table(synthetic)}
<h2>Дополнительная проверка против нулевого прогноза</h2>{table(zero[zero.kind=='market'])}
<p>Эта таблица — дополнительная диагностика после фиксированного обучения. Сентябрьские интервалы основаны на очень малом числе дней.</p>
<h2>Признаки и контроль утечек</h2>
<p>У обеих моделей одинаковая Fibonacci-сетка 1,2,3,5,8,13,21,34,55. Сравнение log/angular изолирует arctan;
оно не проверяет превосходство Fibonacci над другими сетками. 4 привязанных к последнему известному Close OHLC-канала дополняют 36 lag-признаков.
Из-за лагов реальная историческая поддержка достигает 75 свечей. Маскируются все признаки, зависящие от скрытых цен.</p>
<p>Все окна непрерывны по минутам; данные не заполнялись через выходные. Полная поддержка каждого окна, включая будущий контекст учителя,
находится внутри своей части. Масштабы оценены только до 2025 года. Замена скрытых цен не меняет вход teacher;
замена любых будущих цен не меняет вход student. Намеренно небезопасный вариант показал алгебраическую утечку.</p>
{image('scale_law.png','Хороший fit масштаба не является доказательством прогнозируемости.')}
<h2>Бюджет обучения и ограничения</h2>
<p>MLP: 641 872 параметра, AdamW, batch 512, 12 эпох; A_long — 24. Лучший checkpoint выбран по validation.
Полный прогон занял {manifest['elapsed_seconds']/60:.1f} минуты. Использована фиксированная выборка 100 000 окон из 2 535 093 допустимых обучающих окон;
это не исчерпывающее обучение на всех возможных окнах. Учитель обучался на тех же train-окнах, cross-fitting не применялся.</p>
<p>В {len(late)} из 6 teacher-конфигураций лучший validation checkpoint совпал с последней эпохой; это ограничивает выводы о максимуме качества teacher.
Кривые позволяют отличить исчерпание бюджета от ухудшения validation. OHLC-порядок выходов не навязывался, нарушения сохранены в diagnostics.
Использовались только M1, без L2/MBO, комиссий, spread/slippage или оценки прибыльности. Не проверялись другие архитектуры и латентная дистилляция.</p>
{image('validation_curves.png','Кривые обучения всех seed; вертикальная линия — исходный 12-эпоховый бюджет A.')}
<h2>Проверка результата и воспроизведение</h2>
<p>5 тестов прошли. Отдельный скрипт сверил {verified['prediction_metric_checks']} метрик с сохранёнными предсказаниями,
проверил временные границы всех окон, выбор весов и checkpoint по validation, конечность предсказаний и хеши исходных данных/кода/протокола.</p>
<pre>python experiments/angular_distillation/run_experiment.py
python -m pytest experiments/angular_distillation/test_experiment.py -q
python experiments/angular_distillation/verify_and_plot.py
python experiments/angular_distillation/build_delivery.py</pre>
<p>Машиночитаемые файлы рядом с отчётом: metrics.csv, paired_comparisons.csv, delivery_primary.csv, delivery_horizons.csv,
monthly_primary_by_seed.csv, primary_scores_by_seed.csv, selected_vs_zero.csv, verification.json, manifest.json.
Кривые и веса — models/; предсказания — predictions/; подробный лог — run.log.</p>'''
    document='<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#182533;background:#fcfcfd}h1{font-size:32px;line-height:1.2}h2{font-size:23px;margin-top:38px}.lead{font-size:20px;font-weight:600}.facts{padding:18px;background:#eaf0f6;border-radius:10px}table{display:block;overflow:auto;border-collapse:collapse;font-size:12px;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #dbe1e8;text-align:right;white-space:nowrap}th{background:#edf1f5}img{width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:13px;color:#546171}pre{padding:16px;background:#edf1f5;overflow:auto}</style></head><body>'+body+'</body></html>'
    (OUT/'results.html').write_text(document,encoding='utf-8')
    print(verdict);print(p.to_string(index=False));print(h.to_string(index=False));print('REPORT',OUT/'results.html')

if __name__=='__main__':main()
