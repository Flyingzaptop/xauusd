"""Attach a compact verified result summary to the local HTML report."""
import json,sys
import pandas as pd
from run import OUT,sha
from pathlib import Path

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    v=json.loads((OUT/'independent_verification.json').read_text(encoding='utf-8'));assert v['status']=='passed'
    m=pd.read_csv(OUT/'metrics.csv');sh=pd.read_csv(OUT/'candle_metrics.csv');ci=pd.read_csv(OUT/'comparisons.csv')
    rows=m[(m.split=='test')&(m.period=='ALL')&(m.model=='ensemble')]
    simple=rows[['channel','angle_mae_degrees','angle_r2','price_mae_bps']].copy()
    simple.columns=['Канал','MAE угла, градусы','R² угла','MAE цены, bps']
    shape=sh[(sh.split=='test')&(sh.period=='ALL')&(sh.model=='ensemble')].iloc[0]
    baseline=m[(m.split=='test')&(m.period=='ALL')&(m.model=='flat_at_close')&(m.channel=='O')].iloc[0]
    def gain(base,metric):return ci[(ci.split=='test')&(ci.baseline==base)&(ci.metric==metric)].improvement_pct.item()
    text=f'''<!-- VERIFIED_SUMMARY_START --><section style="padding:20px;background:#edf4f8;border-radius:12px;margin:24px 0">
<h2>Результат обученной модели</h2><p><b>TCN частично предсказывает движение High и Low; следующий Close остаётся почти нулевым угловым прогнозом.</b>
Среднее трёх моделей: угловая MSE ниже среднего train-угла на {gain('train_mean','angle_all'):.2f}%,
ниже типичной train-свечи на {gain('train_template','angle_all'):.2f}%. Без Open, по H/C/L, улучшение относительно типичной свечи — {gain('train_template','angle_HCL'):.2f}%.</p>
{simple.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}
<p>Open хорошо восстанавливается относительно постоянного угла, но механический прогноз Open=последний Close точнее:
его MAE угла {baseline.angle_mae_degrees:.2f}°. Поэтому большой R² Open не означает новый рыночный сигнал.</p>
<p>Направление изменения Close: {shape.close_return_direction_accuracy*100:.2f}%; направление тела свечи: {shape.body_direction_accuracy*100:.2f}%.
Некорректный порядок OHLC: {shape.invalid_ohlc_fraction*100:.2f}%. Цены не исправлялись сортировкой.
Общая ценовая MSE относительно типичной свечи изменилась на {-gain('train_template','price_all'):.2f}% в сторону увеличения;
это различие неотличимо от нуля по данному доверительному интервалу. Угловое улучшение не равно улучшению всех ценовых метрик.</p>
<p>6 тестов прошли; 216 метрик перепроверены отдельным скриптом. Воспроизведение с тем же размером пакета совпало точно.
При одиночном применении TF32 наблюдалась разница {v['standalone_tf32_single_batch_max_difference_degrees']:.5f}° из-за GPU-округления;
проверка одинаковых входов и режима IEEE FP32 подтвердила численную причину.</p>
<p>Модель: 39 332 параметра, 3 seed; обучение до2025, выбор на2025, исследовательская оценка на40 532 окнах2026.
Это первый ограниченный эксперимент с данной TCN, а не проверка всех вариантов углового представления.</p>
<pre>python experiments/angular_tcn/predict.py --input data/M1/xauusd_m1.parquet</pre>
<p>По умолчанию прогноз сохраняется в forecast.json рядом с этим отчётом. Для другого набора закрытых свечей передайте свой CSV/Parquet.
Подробная инструкция: experiments/angular_tcn/README.md. Веса: models/17/best.pt, models/29/best.pt, models/43/best.pt.</p>
</section><!-- VERIFIED_SUMMARY_END -->'''
    path=OUT/'results.html';doc=path.read_text(encoding='utf-8')
    if '<!-- VERIFIED_SUMMARY_START -->' in doc:
        first=doc.index('<!-- VERIFIED_SUMMARY_START -->');last=doc.index('<!-- VERIFIED_SUMMARY_END -->')+len('<!-- VERIFIED_SUMMARY_END -->')
        doc=doc[:first]+doc[last:]
    doc=doc.replace('</h1>','</h1>'+text,1);path.write_text(doc,encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    manifest['delivery_report_sha256']=sha(path);manifest['independent_verification']=v
    manifest['inference_script_sha256']=sha(Path(__file__).with_name('predict.py'))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    simple.to_csv(OUT/'summary.csv',index=False)
    print(simple.to_string(index=False));print('REPORT',path)

if __name__=='__main__':main()
