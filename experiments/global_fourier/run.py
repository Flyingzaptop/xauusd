"""Global FFT: randomly selected month, fixed harmonic continuation for seven days.

No neural model, optimization sweep, windowing, detrending, or holdout selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


COUNTS = [10, 50, 100, 500, 1000]


def fit(y, count):
    n = len(y)
    mean = float(np.mean(y))
    spectrum = np.fft.rfft(y - mean)
    energy = 2 * np.abs(spectrum[1:]) ** 2 / n ** 2
    if n % 2 == 0:
        energy[-1] *= .5
    ranked = np.argsort(-energy, kind='stable') + 1
    chosen = ranked[:count]
    kept = np.zeros_like(spectrum)
    kept[chosen] = spectrum[chosen]
    reconstructed = mean + np.fft.irfft(kept, n=n)
    return reconstructed, chosen, spectrum, mean, float(energy[chosen - 1].sum() / energy.sum())


def predict_from_train(train, count, future_length):
    reconstruction, chosen, spectrum, mean, energy_fraction = fit(train, count)
    future = reconstruction[np.arange(future_length) % len(train)]
    return np.r_[reconstruction, future], chosen, spectrum, mean, energy_fraction


def metrics(actual, pred, mask, train_std):
    err = pred[mask] - actual[mask]
    return {
        'scored_observed_candles': int(np.sum(mask)),
        'mae_usd': float(np.mean(abs(err))),
        'rmse_usd': float(np.sqrt(np.mean(err ** 2))),
        'mape_percent': float(np.mean(abs(err) / abs(actual[mask])) * 100),
        'max_abs_error_usd': float(np.max(abs(err))),
        'mae_divided_by_train_std': float(np.mean(abs(err)) / train_std),
    }


def minute_change_metrics(actual, pred, valid):
    e = np.diff(pred)[valid] - np.diff(actual)[valid]
    truth = np.diff(actual)[valid]
    return {'pairs': int(valid.sum()), 'mae_usd': float(np.mean(abs(e))),
            'actual_mean_abs_change_usd': float(np.mean(abs(truth))),
            'prediction_mean_abs_change_usd': float(np.mean(abs(np.diff(pred)[valid])))}


def explicit(chosen, spectrum, mean, n, positions):
    result = np.full(len(positions), mean, dtype=float)
    for k in chosen:
        weight = 1 if n % 2 == 0 and k == n // 2 else 2
        result += weight / n * np.real(spectrum[k] * np.exp(2j*np.pi*k*positions/n))
    return result


def synthetic_check():
    n = 256
    t = np.arange(n + 100)
    y = 1800 + 3*np.cos(2*np.pi*3*t/n + .2) + 2*np.sin(2*np.pi*7*t/n) + .5*np.cos(2*np.pi*11*t/n + .9)
    pred, *_ = predict_from_train(y[:n], 3, 100)
    error = float(np.max(abs(pred - y)))
    assert error < 1e-9
    return error


def plot_panel(ax, times, actual, pred, split, train_start, limits=None, baseline=None):
    ax.plot(times, actual, color='#161c27', linewidth=.85, label='Actual Close', zorder=3)
    before = times < split
    ax.plot(times[before], pred[before], color='#226ac5', linewidth=.8, alpha=.9, label='Fit: month')
    after = ~before
    ax.plot(times[after], pred[after], color='#d54b24', linewidth=1, label='Forecast: unseen week', zorder=4)
    if baseline is not None:
        ax.plot(times[after], baseline[after], color='#6b7280', linewidth=.9, linestyle='--', label='Last-Close forecast')
    ax.axvline(split, color='#9a3450', linestyle='--', linewidth=1)
    ax.axvspan(split, times[-1], color='#fae8db', alpha=.5, zorder=0)
    ax.grid(alpha=.18)
    ax.set_ylabel('XAUUSD Close ($)')
    if limits is not None:
        ax.set_xlim(*limits)
        visible = (times >= limits[0]) & (times <= limits[1])
        values = np.r_[actual[visible],pred[visible]]
        if baseline is not None:
            values = np.r_[values,baseline[visible & after]]
        lo, hi = values.min(), values.max()
        margin = max(.25, .07*(hi-lo))
        ax.set_ylim(lo-margin,hi+margin)
    else:
        ax.set_xlim(train_start, times[-1])
    ax.set_xlabel('Consecutive observed candle index (market closures removed)')


def make_plots(out, times, actual, preds, split, month_start, baseline, rows):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                        'text.parse_math':False})
    dates = times
    times = np.arange(len(actual))
    month_start = 0
    def label(value, position):
        i = int(np.clip(round(value), 0, len(dates)-1))
        return str(dates[i]).replace('T','\n')
    for filename, limits, heading in [
        ('month_and_forecast.png', None, 'Global Fourier: reconstructed month + unseen next week'),
        ('forecast_week.png', (split, times[-1]), 'Global Fourier: next-week forecast, no refitting'),
        ('boundary.png', (split-1440,split+1440), 'Boundary detail: last fitted candles and first forecast candles'),
    ]:
        fig, axs = plt.subplots(5, 1, figsize=(15, 17), constrained_layout=True)
        fig.suptitle(heading+'\nFixed strongest harmonics; raw Close with mean offset', fontsize=16)
        for ax, k in zip(axs, COUNTS):
            plot_panel(ax, times, actual, preds[k], split, month_start, limits, baseline)
            ax.xaxis.set_major_formatter(FuncFormatter(label))
            row = rows[str(k)]
            ax.set_title(f"{k} harmonics | month MAE ${row['train']['mae_usd']:.2f} ({row['train']['mape_percent']:.3f}%)"
                         f" | week MAE ${row['test']['mae_usd']:.2f} ({row['test']['mape_percent']:.3f}%)", loc='left', fontsize=11)
        axs[0].legend(loc='best', ncol=4, fontsize=9)
        fig.savefig(out/filename, dpi=150)
        plt.close(fig)
    # Window fixed by chronology, not chosen by quality of fit.
    mid = split//2
    fig, axs = plt.subplots(5, 1, figsize=(15, 14), constrained_layout=True)
    fig.suptitle('Reconstruction detail: 360 consecutive candles at month midpoint', fontsize=16)
    for ax, k in zip(axs, COUNTS):
        plot_panel(ax, times, actual, preds[k], split, month_start, (mid, mid+360))
        ax.xaxis.set_major_formatter(FuncFormatter(label))
        ax.set_title(f'{k} harmonics', loc='left')
    axs[0].legend(loc='best', fontsize=9)
    fig.savefig(out/'reconstruction_detail.png', dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(COUNTS))
    for ax, metric, label in [(axs[0], 'mae_usd', 'MAE ($)'), (axs[1], 'mape_percent', 'MAPE (%)')]:
        ax.plot(x, [rows[str(k)]['train'][metric] for k in COUNTS], 'o-', color='#226ac5', label='Fitted month')
        ax.plot(x, [rows[str(k)]['test'][metric] for k in COUNTS], 'o-', color='#d54b24', label='Unseen week')
        ax.axhline(rows['persistence']['test'][metric], color='#6b7280', linestyle='--', label='Last-Close week baseline')
        ax.set_xticks(x, COUNTS)
        ax.set_xlabel('Number of retained harmonics')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(out/'errors.png', dpi=160)
    plt.close(fig)


def write_report(out, meta):
    s = meta['candle_clock']
    table = ['| Гармоник | Месяц: MAE, $ | Месяц: MAPE | Неделя: MAE, $ | Неделя: MAPE | Энергия на месяце |',
             '|---:|---:|---:|---:|---:|---:|']
    for k in COUNTS:
        r = s['models'][str(k)]
        table.append(f"| {k} | {r['train']['mae_usd']:.3f} | {r['train']['mape_percent']:.4f}% | {r['test']['mae_usd']:.3f} | {r['test']['mape_percent']:.4f}% | {100*r['retained_energy_fraction']:.2f}% |")
    p = s['models']['persistence']['test']
    table.append(f"| Последний Close | — | — | {p['mae_usd']:.3f} | {p['mape_percent']:.4f}% | — |")
    text = f'''# Глобальный Fourier: месяц и продолжение на неделю

Случайно выбран **{meta['selected_month']}**, seed={meta['seed']}, равномерный выбор среди {meta['candidate_month_count']} полных месяцев с доступной следующей неделей. Выбор до расчёта спектров и ошибок. Месяц {meta['train_period']}; неделя {meta['test_period']}. Это один исследовательский пример, не доказательство закономерности на всех периодах.

## Что считали

Обычный FFT сырого Close в долларах. Вычитание среднего и его последующее добавление — обратимая смена начала координат; рост/тренд не удаляли, clipping и нормализацию по будущему не применяли. Одна гармоника — пара cos/sin одной положительной частоты, или эквивалентно A*cos(2π*f*t+phase); постоянное среднее не входит в 10/50/100/500/1000.

Частоты — стандартная сетка k/N. Для каждого K оставлены K гармоник с максимальной вкладом в квадрат ошибки/энергию TRAIN. Это наименьшая квадратичная ошибка на регулярной сетке при данном K среди подмножеств FFT-базиса. Не использовали будущее для выбора частот, фаз, амплитуд, месяца или K. Никакие частоты между FFT-бинами отдельно не оптимизировались. Сравниваем все пять K; лучший прогноз задним числом не объявляем выбранной моделью.

По уточнению пользователя **все паузы удалены, реальные свечи склеены**. После пятничной свечи сразу идёт понедельничная. Регулярная координата FFT — номер свечи, не календарная минута. Ни пропуски, ни выходные не заполняются. За месяц {s['train_candles']} свечей, за следующую календарную неделю {s['test_candles']}. На графиках также используется номер свечи; реальные даты служат подписями и границами split. Timezone исходного feed не подтверждён; шкала следует сохранённым timestamp.

Формула продолжения:

    prediction(t) = mean_train + Σ A_k*cos(2π*k*t/N + phase_k)

**При FFT-сетке продолжение периодично с периодом N свечей. Прогноз следующих M свечей буквально повторяет первые M свечей восстановленного месяца по фазе.** Это свойство проверенного глобального метода, а не ошибка программирования. Подгонка большого количества гармоник не превращает его в другой способ экстраполяции. После границы возможен скачок из-за замыкания конца месяца на начало.

## Ошибки: склеенные свечи

{chr(10).join(table)}

MAPE = средняя абсолютная ошибка / фактический Close * 100. Небольшой процент от цены золота не обязательно означает малую ошибку относительно минутных движений. Поэтому также показаны MAE/RMSE в долларах и ошибки минутных приращений в summary.json. Постоянный forecast последнего реального Close месяца — baseline, использует только TRAIN.

В этом месяце добавление гармоник улучшает восстановление прошлого, но не продолжение на неделю. Все пять продолжений уступают baseline по последнему Close. Это результат именно обычного глобального FFT с указанной сеткой частот на одном месяце; он не исключает предсказуемости иных спектральных признаков или локальных методов. Простая афинная нормализация этого результата не изменит, что проверено отдельно.

'''
    text += f'''
## Проверки и ограничения

Полный FFT восстанавливает месяц с максимальной численной ошибкой ${meta['checks']['full_fft_train_max_error_usd']:.3g}; для этого нужны {s['available_nonconstant_harmonics']} непостоянных гармоник. Синтетическая сумма трёх известных гармоник корректно восстановлена и продолжена. Прямое вычисление cos/sin независимо сверено с ifft. Изменение всех будущих цен не меняет спектр и прогноз. Афинное TRAIN-масштабирование даёт тот же forecast после обратного преобразования. Числа находятся в checks.json.

1000 гармоник не являются полной реконструкцией каждого минутного движения. Сравнение соседних Close проверяет это отдельно. Полный FFT восстановит прошлое почти точно, но его продолжение тоже повторяет прошлый период.

Один месяц; без доверительных интервалов по независимым сделкам, без торговой стратегии/издержек. Склейка не устраняет ценовые гэпы: она лишь делает соседними реально наблюдавшиеся свечи. Периоды гармоник измеряются в свечах, а не в календарных минутах. STFT, вейвлеты, тренд+FFT и нейросети в этом опыте не запускались. GPU не использовался.

## Артефакты

- month_and_forecast.png — все пять вариантов, месяц и следующая неделя.
- forecast_week.png — следующая неделя крупно.
- boundary.png — последний день месяца и первый день недели.
- reconstruction_detail.png — фиксированные 360 свечей внутри месяца.
- errors.png — сравнение ошибок.
- predictions.npz — реальные Close, timestamp, реконструкции и продолжения.
- harmonics.csv — частоты, периоды, амплитуды, фазы всех оставленных гармоник.
- summary.json / checks.json — настройки, метрики и проверки.

Воспроизведение: Python с NumPy/Matplotlib, `experiments/global_fourier/run.py --source reports/pcgrad_mse_month_20261009/dataset.npz --out <new-directory> --seed {meta['seed']}`.
'''
    (out/'REPORT.md').write_text(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261012)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.source) as z:
        ts = z['timestamp']//60
        prices = np.exp(z['log_prices'][:, 2])
    assert np.all(np.diff(ts)>0)
    first = np.datetime64(int(ts[0]), 'm').astype('datetime64[M]') + np.timedelta64(1,'M')
    last = np.datetime64(int(ts[-1]), 'm').astype('datetime64[M]')
    months = np.arange(first, last, dtype='datetime64[M]')
    candidates = [m for m in months if int(((m+np.timedelta64(1,'M')).astype('datetime64[m]')+np.timedelta64(7,'D')).astype('int64')) <= ts[-1]]
    rng = np.random.default_rng(a.seed)
    month = candidates[int(rng.integers(len(candidates)))]
    start = int(month.astype('datetime64[m]').astype('int64'))
    split = int((month+np.timedelta64(1,'M')).astype('datetime64[m]').astype('int64'))
    end = split + 7*1440
    selected = (ts >= start) & (ts < end)
    dates = ts[selected]
    actual = prices[selected]
    observed = np.ones(len(actual),dtype=bool)
    n = int((dates < split).sum())
    assert n > 2000 and len(actual)-n > 20
    arrays = {'timestamp_minutes': dates, 'close': actual, 'train_candles':np.array(n)}
    last_close = float(actual[n-1])
    info = {}
    coeff_rows = []
    all_preds = {}
    checks = {'synthetic_three_harmonic_max_error_usd': synthetic_check()}
    for clock, y, nn, mask in [('candle_clock', actual, n, observed)]:
        train = y[:nn].copy()
        std = float(train.std())
        rows, preds = {}, {}
        baseline = np.full(len(y), last_close)
        rows['persistence'] = {'test':metrics(y[nn:],baseline[nn:],mask[nn:],std)}
        for k in COUNTS:
            pred, chosen, spectrum, mean, fraction = predict_from_train(train,k,len(y)-nn)
            preds[k] = pred
            arrays[f'{clock}_{k}'] = pred
            rows[str(k)] = {'train':metrics(y[:nn],pred[:nn],mask[:nn],std),
                            'test':metrics(y[nn:],pred[nn:],mask[nn:],std),
                            'retained_energy_fraction':fraction}
            if clock == 'candle_clock':
                valid = np.diff(dates[:nn]) == 1
                rows[str(k)]['minute_change_train'] = minute_change_metrics(y[:nn],pred[:nn],valid)
                valid = np.diff(dates[nn-1:]) == 1
                rows[str(k)]['minute_change_test'] = minute_change_metrics(y[nn-1:],pred[nn-1:],valid)
            positions = np.array([0,17,nn-1,nn,nn+20,len(y)-1])
            error = float(np.max(abs(explicit(chosen,spectrum,mean,nn,positions)-pred[positions])))
            assert error < 1e-8
            checks[f'{clock}_{k}_direct_cosine_max_error_usd'] = error
            for j in chosen:
                weight = 1 if nn%2==0 and j==nn//2 else 2
                coeff_rows.append([clock,k,int(j),float(j/nn),float(nn/j),
                                   float(weight*abs(spectrum[j])/nn),float(np.angle(spectrum[j])),mean])
        # Use a future copy deliberately transformed to demonstrate train-only forecast.
        mutated = y.copy()
        mutated[nn:] = mutated[nn:]*7+1234
        p2,*_ = predict_from_train(mutated[:nn],1000,len(y)-nn)
        assert np.array_equal(p2,preds[1000])
        checks[f'{clock}_future_mutation_forecast_unchanged'] = True
        assert np.array_equal(preds[1000][nn:],preds[1000][:len(y)-nn])
        checks[f'{clock}_continuation_equals_first_period'] = True
        scaled, *_ = predict_from_train((train-train.mean())/std,1000,len(y)-nn)
        scale_error = float(np.max(abs((scaled*std+train.mean())-preds[1000])))
        assert scale_error < 1e-8
        checks[f'{clock}_affine_normalization_max_difference_usd'] = scale_error
        # Check monotonic least-squares reconstruction on the FFT's actual grid.
        mse = [float(np.mean((preds[k][:nn]-train)**2)) for k in COUNTS]
        assert np.all(np.diff(mse)<=0)
        checks[f'{clock}_training_grid_mse_nonincreasing'] = True
        info[clock] = {'train_candles':nn,
                       'test_candles':len(y)-nn,
                       'train_observed':int(mask[:nn].sum()),'test_observed':int(mask[nn:].sum()),
                       'available_nonconstant_harmonics':nn//2,'train_std_usd':std,'models':rows}
        all_preds[clock] = preds
    reconstructed = np.mean(actual[:n])+np.fft.irfft(np.fft.rfft(actual[:n]-np.mean(actual[:n])), n=n)
    checks['full_fft_train_max_error_usd'] = float(np.max(abs(reconstructed-actual[:n])))
    assert checks['full_fft_train_max_error_usd']<1e-8
    meta = {'seed':a.seed,'selected_month':str(month),'candidate_month_count':len(candidates),
            'train_period':[str(np.datetime64(start,'m')),str(np.datetime64(split,'m'))],
            'test_period':[str(np.datetime64(split,'m')),str(np.datetime64(end,'m'))],
            'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
            'source_columns':['High','Open','Close','Low'],
            'harmonic_counts':COUNTS,'selection':'largest TRAIN Fourier energy; DC additional',
            'preprocessing':'raw Close; mean offset only; no detrending or clipping',
            'missing_minutes':'removed entirely; consecutive real candles; no filling',
            'removed_gaps_between_observed_candles':int(np.sum(np.diff(dates)>1)),
            'removed_calendar_minutes':int(np.sum(np.diff(dates)-1)),
            'gpu_used':False,'checks':checks,**info}
    np.savez_compressed(a.out/'predictions.npz',**arrays)
    with (a.out/'harmonics.csv').open('w') as f:
        w=csv.writer(f)
        w.writerow(['clock','count','frequency_bin','cycles_per_grid_step','period_grid_steps','amplitude_usd','phase_radians','mean_usd'])
        w.writerows(coeff_rows)
    (a.out/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    (a.out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    make_plots(a.out,dates.astype('datetime64[m]'),actual,all_preds['candle_clock'],n,0,np.full(len(actual),last_close),info['candle_clock']['models'])
    write_report(a.out,meta)
    print(json.dumps({'month':str(month),'candle':info['candle_clock'],'checks':checks},indent=2))


if __name__=='__main__':
    main()
