"""Descriptive STFT / complex Morlet CWT on consecutive observed candles.

No forecasting model is fitted. Centered CWT is explicitly retrospective.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pywt
import scipy
from scipy.signal import detrend, windows


HOP = 32
PERIODS = np.geomspace(4, 1024, 48)
WAVELET = 'cmor1.5-1.0'


def stft(x, length=512, hop=HOP, linear=False):
    segments = np.lib.stride_tricks.sliding_window_view(x, length)[::hop]
    clean = detrend(segments, axis=1, type='linear' if linear else 'constant')
    window = windows.hann(length, sym=False)
    z = np.fft.rfft(clean*window, axis=1).T
    power = 2*abs(z)**2 / np.sum(window**2)
    freq = np.fft.rfftfreq(length)
    keep = (freq >= 2/length) & (freq <= .25)
    z, power, freq = z[keep][::-1], power[keep][::-1], freq[keep][::-1]
    return {'z':z, 'power':power, 'period':1/freq, 'freq':freq,
            'end':np.arange(z.shape[1])*hop+length-1, 'length':length, 'hop':hop}


def cwt(x):
    scale = pywt.frequency2scale(WAVELET, 1/PERIODS)
    z, freq = pywt.cwt(x, scale, WAVELET, sampling_period=1, method='fft')
    return z, scale, 1/freq


def reference(power, times, split):
    return np.maximum(np.median(power[:, times < split], axis=1), 1e-20)


def relative_db(power, ref):
    return 10*np.log10(np.maximum(power,1e-20)/ref[:,None])


def corr(a,b):
    return float(np.corrcoef(a,b)[0,1])


def stats(x):
    spec = stft(x,512,512)
    z, p = spec['z'], spec['power']
    delta = np.angle(z[:,1:]*np.conj(z[:,:-1])) - 2*np.pi*spec['freq'][:,None]*512
    weight = np.sqrt(p[:,1:]*p[:,:-1])
    return {
        'nonoverlap_phase_continuation_cosine':float(np.sum(weight*np.cos(delta))/weight.sum()),
        'nonoverlap_windows':int(z.shape[1]),
        'mean_largest_frequency_power_share':float(np.mean(np.max(p,axis=0)/np.sum(p,axis=0))),
        'mean_spectral_flatness':float(np.mean(np.exp(np.mean(np.log(np.maximum(p,1e-20)),axis=0))/np.mean(p,axis=0))),
        'signed_return_autocorrelation':{str(lag):corr(x[:-lag],x[lag:]) for lag in [1,5,15,60]},
        'absolute_return_autocorrelation':{str(lag):corr(abs(x[:-lag]),abs(x[lag:])) for lag in [1,5,15,60]},
    }


def decoration(ax, dates, split, event=None, limits=None):
    n = len(dates)
    def tick(v,pos):
        i = int(np.clip(round(v),0,n-1))
        return str(dates[i]).replace('T','\n')
    ax.xaxis.set_major_formatter(FuncFormatter(tick))
    ax.set_xlim(*(limits or (0,n-1)))
    ax.axvline(split,color='#9e315a',linestyle='--',linewidth=1)
    if event is not None:
        ax.axvline(event,color='#111827',linestyle=':',linewidth=1)
    ax.set_xlabel('Consecutive observed candles; closures removed')


def heat(ax,time,period,data,title,dates,split,event=None,limits=None,phase=False,cone=False):
    im=ax.pcolormesh(time,period,data,shading='auto',cmap='twilight' if phase else 'magma',
                     vmin=-np.pi if phase else -12,vmax=np.pi if phase else 18,rasterized=True)
    ax.set_yscale('log',base=2)
    ticks=[4,8,16,32,64,128,256,512,1024]
    ax.set_yticks([t for t in ticks if period.min()<=t<=period.max()],
                  [str(t) for t in ticks if period.min()<=t<=period.max()])
    ax.set_ylim(period.min(),period.max())
    ax.set_ylabel('Period (candles)')
    ax.set_title(title,loc='left',fontsize=11)
    decoration(ax,dates,split,event,limits)
    if cone:
        # e^-1 amplitude envelope distance for cmor B=1.5; a heuristic COI,
        # NOT the strict finite-kernel support and NOT a causal boundary.
        radius=np.sqrt(1.5)*period
        ax.fill_betweenx(period,0,radius,color='#a8adb5',alpha=.45)
        ax.fill_betweenx(period,len(dates)-1-radius,len(dates)-1,color='#a8adb5',alpha=.45)
    return im


def plot_all(out,dates,close,split,event,r,price_stft,specs,wprice,wr,delayed,scales,controls):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'text.parse_math':False})
    t=np.arange(len(close)); wt=t[::HOP]
    rref=reference(abs(wr[:,::HOP])**2,wt,split)
    pref=reference(abs(wprice[:,::HOP])**2,wt,split)
    sref={k:reference(v['power'],v['end'],split) for k,v in specs.items()}
    for name,limits in [('overview.png',None),('event_detail.png',(event-768,event+1536))]:
        fig,ax=plt.subplots(5,1,figsize=(15,16),constrained_layout=True)
        fig.suptitle('STFT and wavelets | June 2021 + July 1-7 | all shown prices are observed\n'
                     'STFT: trailing windows. Centered CWT: retrospective, includes future neighbors.',fontsize=14)
        ax[0].plot(t,close,color='#172033',linewidth=.8)
        ax[0].set_ylabel('Close ($)');ax[0].set_title('Actual price; dotted line = largest TRAIN one-candle drop',loc='left')
        decoration(ax[0],dates,split,event,limits)
        if limits:
            a,b=limits; seg=close[a:b];ax[0].set_ylim(seg.min()-2,seg.max()+2)
        im=heat(ax[1],price_stft['end'],price_stft['period'],
                relative_db(price_stft['power'],reference(price_stft['power'],price_stft['end'],split)),
                'STFT / locally linearly detrended log price / 512 trailing candles',dates,split,event,limits)
        heat(ax[2],specs[512]['end'],specs[512]['period'],relative_db(specs[512]['power'],sref[512]),
             'STFT / log returns / 512 trailing candles',dates,split,event,limits)
        heat(ax[3],specs[2048]['end'],specs[2048]['period'],relative_db(specs[2048]['power'],sref[2048]),
             'STFT / log returns / 2048 trailing candles',dates,split,event,limits)
        heat(ax[4],wt,PERIODS,relative_db(abs(wr[:,::HOP])**2,rref),
             'Complex Morlet CWT / log returns / CENTERED: pre-event glow uses future observations',
             dates,split,event,limits,cone=True)
        fig.colorbar(im,ax=ax[1:],label='Power / TRAIN median for each frequency or scale (dB)',shrink=.75)
        fig.savefig(out/name,dpi=145);plt.close(fig)
    fig,ax=plt.subplots(4,1,figsize=(15,14),constrained_layout=True)
    fig.suptitle('Complex Morlet wavelets | centered map versus strict delayed availability',fontsize=15)
    im=heat(ax[0],wt,PERIODS,relative_db(abs(wprice[:,::HOP])**2,pref),
            'Centered log-price CWT; mean offset only; trend is still present',dates,split,cone=True)
    heat(ax[1],wt,PERIODS,relative_db(abs(wr[:,::HOP])**2,rref),
         'Centered return CWT; retrospective',dates,split,cone=True)
    db=relative_db(abs(wr[:,::HOP])**2,rref)
    phase=np.angle(wr[:,::HOP]);phase[db<6]=np.nan
    pi=heat(ax[2],wt,PERIODS,phase,
            'Centered return CWT phase; shown only above 4x TRAIN median power; not a direction forecast',
            dates,split,phase=True,cone=True)
    heat(ax[3],wt,PERIODS,relative_db(abs(delayed)**2,rref),
         'Same coefficients displayed ONLY when fully available: delay = ceil(8*scale)+4 candles',dates,split)
    fig.colorbar(im,ax=[ax[0],ax[1],ax[3]],label='Power vs TRAIN scale median (dB)',shrink=.8)
    fig.colorbar(pi,ax=ax[2],label='Phase (radians)',shrink=.8)
    fig.savefig(out/'wavelets.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(17,11),constrained_layout=True)
    fig.suptitle('Controls: bright multiscale patches also occur without persistent direction\n'
                 'Same color references fitted on actual June returns; all CWT panels are centered.',fontsize=14)
    entries=[('Actual',r,specs[512],wr),*controls]
    for col,(name,x,s,w) in enumerate(entries):
        axes[0,col].plot(t[::4],x[::4],linewidth=.45,color='#25364c')
        axes[0,col].set_title(name);axes[0,col].set_ylabel('Log change (bps)')
        decoration(axes[0,col],dates,split)
        axes[0,col].xaxis.set_major_locator(MaxNLocator(3))
        im=heat(axes[1,col],s['end'],s['period'],relative_db(s['power'],sref[512]),'Trailing STFT 512',dates,split)
        heat(axes[2,col],wt,PERIODS,relative_db(abs(w[:,::HOP])**2,rref),'Centered CWT',dates,split,cone=True)
        axes[1,col].xaxis.set_major_locator(MaxNLocator(3))
        axes[2,col].xaxis.set_major_locator(MaxNLocator(3))
    fig.colorbar(im,ax=axes[1:,:].ravel().tolist(),label='Power vs actual TRAIN frequency/scale median (dB)',shrink=.8)
    fig.savefig(out/'controls.png',dpi=140);plt.close(fig)
    return rref,sref


def checks(r,split,spec,wr,scales,event):
    result={}
    # Independent fixed sinusoid verifies both reported period scales.
    t=np.arange(8192);x=np.cos(2*np.pi*t/64)
    ss=stft(x)
    period=float(ss['period'][np.argmax(np.mean(ss['power'],axis=1))])
    assert abs(period-64)<1e-8
    wz,_,wp=cwt(x)
    peak=float(wp[np.argmax(np.mean(abs(wz[:,2048:-2048])**2,axis=1))])
    assert abs(np.log2(peak/64))<.25
    result['synthetic_period64_stft_peak']=period
    result['synthetic_period64_cwt_peak']=peak
    # Mutation begins at the shock. Compare coefficients at available moments.
    cut=event
    rng=np.random.default_rng(987)
    mutated=r.copy();mutated[cut:]=rng.normal(0,10,len(r)-cut)
    ms=stft(mutated)
    before=spec['end']<cut
    err=float(np.max(abs(ms['z'][:,before]-spec['z'][:,before])))
    assert err==0
    result['future_mutation_trailing_stft_max_difference']=err
    mw,_,_=cwt(mutated)
    j=int(np.argmin(abs(PERIODS-64)))
    center=cut-1
    result['centered_cwt_at_last_visible_candle_changes_with_future']=float(abs(mw[j,center]-wr[j,center]))
    assert result['centered_cwt_at_last_visible_candle_changes_with_future']>1e-8
    changes=[]
    for j in [0,12,24,36]:
        c=cut-1-int(np.ceil(8*scales[j]))-4
        changes.append(float(abs(mw[j,c]-wr[j,c])))
    assert max(changes)<1e-8
    result['delayed_cwt_future_mutation_max_difference']=max(changes)
    # Prefix recomputation models actual availability, not only mutation.
    prefix,_,_=cwt(r[:cut])
    errors=[]
    for j in [0,12,24,36]:
        c=cut-1-int(np.ceil(8*scales[j]))-4
        errors.append(float(abs(prefix[j,c]-wr[j,c])))
    assert max(errors)<1e-8
    result['delayed_cwt_prefix_recompute_max_difference']=max(errors)
    return result


def report(out,s):
    text=f'''# STFT и комплексные вейвлеты: июнь 2021 + 1–7 июля

Визуальное исследование того же случайно выбранного месяца, что в глобальном FFT. {s['train_candles']} свечей июня и {s['following_week_candles']} свечей следующей недели. Все свечи склеены, никаких вставок или заполнения выходных. На оси X — номер реальной свечи, даты только подписывают её. На оси Y — период в свечах, не календарных минутах. Скачки цены между соседними реальными свечами сохраняются.

## Представления и методы

- STFT: Hann, окна 512/2048 свечей, шаг 32, коэффициенты привязаны к последней доступной свече окна. Частоты с периодами 4…L/2 свечей. Для log-цены внутри каждого окна удалена линейная компонента; у доходностей — только среднее окна. Это локальное описание, не глобальное продолжение синусоид.
- Вейвлет: комплексный Morlet `{WAVELET}`, 48 логарифмически расположенных периодов 4–1024 свечи. Центрированная CWT по log-цене и по её изменениям. Для log-цены только вычтен TRAIN mean, тренд остаётся.
- Изменения: 10000*Δlog(Close), базисные пункты. В первой точке 0, т.к. предыдущая цена вне загруженного набора. Паузы удалены, поэтому отдельное изменение может быть гэпом через закрытие рынка.
- Цвет: энергия относительно медианы июня ДЛЯ КАЖДОЙ частоты/масштаба, −12…+18 dB. Одинаковый цвет на разных масштабах не означает одинаковую абсолютную энергию. Цветовая шкала и порог отображения фазы не подгоняются по июльской неделе. В фазовой панели показаны только участки с энергией >=4x медианы TRAIN; это фильтр визуализации, не доказательство устойчивого цикла.

Полная CWT вычислена для каждой свечи; для файлов коэффициентов и просмотра сохранён шаг 32. Такое прореживание предназначено для просмотра, а не для восстановления фазовой траектории каждого минутного компонента. Для будущей модели частоту сохранения признаков нужно выбирать под задачу.

Все точки месяца и недели — уже наблюдённые цены. Пунктирная граница месяца не является границей между fit и forecast для этих карт. **Прогнозов здесь нет.**

Цветовая шкала рассчитана по всему июню и служит ретроспективному просмотру. Для будущего realtime-пайплайна такую шкалу нужно зафиксировать на более раннем периоде или пересчитывать только по прошлому. Тест causal-доступности ниже проверяет исходные комплексные коэффициенты доходностей, а не подгонку всей цветовой шкалы.

## Наблюдения

Видны всплески, захватывающие сразу много масштабов, и чередование тихих/активных отрезков. Случайная замена знаков при сохранении каждого абсолютного движения сохраняет похожую структуру активности. Отдельное перемешивание доходностей делает активность более однородной по времени. Это согласуется с кластеризацией волатильности; сами яркие пятна не являются свидетельством устойчивого направленного цикла.

Автокорреляция абсолютных изменений: lag 1 = {s['train_statistics']['absolute_return_autocorrelation']['1']:.3f}, lag 60 = {s['train_statistics']['absolute_return_autocorrelation']['60']:.3f}. Для подписанных изменений: {s['train_statistics']['signed_return_autocorrelation']['1']:.3f} и {s['train_statistics']['signed_return_autocorrelation']['60']:.3f}. Это описательные статистики одного периода, не оценка готовой торговой модели.

Проверенный агрегированный способ продолжения фаз не выделился на фоне контролей. Средняя доля энергии самой сильной частоты также попала в контрольные диапазоны. Обнаруженного очевидного устойчивого цикла для направления пока нет. Это не исключает отдельные локальные события, нелинейные зависимости или полезность спектральных признаков для последующей модели.

## Что нельзя считать предварительным сигналом

Обычная CWT использует соседей и слева, и справа. Перед большим событием на карте может появляться свечение, вызванное самим будущим событием. Контрольное изменение всех будущих данных изменило центрированный коэффициент в последней видимой точке на {s['checks']['centered_cwt_at_last_visible_candle_changes_with_future']:.6g}. Поэтому «полосу видно до падения» не означает, что её можно было видеть тогда.

Серые области — приблизительная cone of influence для Gaussian envelope e^-1. Даже вне них центрированная карта использует будущее. Это не маска causal-доступности.

Нижняя панель wavelets.png показывает те же коэффициенты в момент, когда их конечный kernel полностью доступен: delay=ceil(8*scale)+4 свечи. Это консервативная задержка полной поддержки PyWavelets, не оптимальный realtime wavelet. На крупных масштабах она велика; панель не прогнозирует актуальную фазу на текущем краю. Проверены изменение будущего и пересчёт по доступному префиксу. STFT trailing-окон на прошлых точках остаётся неизменным при изменении будущего.

Перекрывающиеся окна STFT разделяют почти все данные. Поэтому для статистики фазовой устойчивости использованы НЕперекрывающиеся окна по 512 свечей; карта со шагом 32 служит только визуализации.

## Контроли

1. Перемешанные доходности: сохраняют эмпирическое распределение, разрушают порядок. TRAIN и последующая неделя перемешиваются отдельно, будущее не переносится в прошлое.
2. Случайные знаки: |доходность| в каждый момент сохранена точно; знаки независимо назначены ±1. Это сохраняет кластеризацию активности, но убирает исходную направленную структуру. На этих рядах тоже возможны яркие многомасштабные пятна.

По 49 TRAIN-реализаций каждого контроля. Интервалы в summary.json — распределения суррогатных статистик, не доверительные интервалы торгового преимущества и не тест всех гипотез о рынке. Это выбранный ранее один месяц, а не новый тест предсказательной модели.

Средний cos ошибки фазового продолжения между неперекрывающимися окнами:
- рынок: {s['train_statistics']['nonoverlap_phase_continuation_cosine']:.4f};
- перемешивание: median {s['surrogates']['shuffle']['phase_median']:.4f}, 2.5–97.5% {s['surrogates']['shuffle']['phase_interval']};
- случайные знаки: median {s['surrogates']['random_sign']['phase_median']:.4f}, 2.5–97.5% {s['surrogates']['random_sign']['phase_interval']}.

Это проверка одного конкретного способа фазового продолжения, а не доказательство наличия/отсутствия любой предсказуемой структуры. Есть вес по энергии, поэтому сильные события влияют больше. Агрегация всех частот может скрывать частные компоненты.

## Самое сильное наблюдённое событие

Выбрано постфактум по наибольшему абсолютному изменению В ИЮНЕ: {s['largest_train_event']['timestamp']}, ΔClose={s['largest_train_event']['close_change_usd']:.2f} USD. event_detail.png показывает локальные карты вокруг него. Эта иллюстрация — объяснение уже произошедшего события, не оценка частоты заранее найденных сигналов.

## Артефакты и воспроизведение

- overview.png: цена, два разрешения STFT и CWT.
- event_detail.png: карты вокруг крупнейшего TRAIN изменения.
- wavelets.png: цена/доходности, фаза и строгая задержанная доступность.
- controls.png: рынок рядом с двумя контрольными рядами.
- diagnostics.png: статистики неперекрывающихся окон и автокорреляции.
- review.html: интерактивный просмотр любого участка и выбор карты.
- coefficients.npz: комплексные STFT/CWT коэффициенты (CWT сохранена с шагом 32), периоды, индексы доступности и контрольные ряды.
- summary.json / checks.json: настройки, статистики, версии и проверки.

`python experiments/local_spectral/run.py --source reports/global_fourier_month_20261012/predictions.npz --out <new-directory>`

Python/NumPy/SciPy/Matplotlib/PyWavelets; GPU не нужен и не использовался. Источник и код имеют SHA256 в MANIFEST.json. Нейросеть, торговая стратегия и экстраполяция не обучались.
'''
    (out/'REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.source) as z:
        close=z['close'];dates=z['timestamp_minutes'].astype('datetime64[m]');split=int(z['train_candles'])
    r=np.r_[0,np.diff(np.log(close))*10000]
    logprice=np.log(close)*10000;logprice-=np.mean(logprice[:split])
    event=int(np.argmax(abs(r[:split])))
    specs={length:stft(r,length) for length in [512,2048]}
    ps=stft(logprice,512,linear=True)
    wr,scales,period=cwt(r);wp,_,_=cwt(logprice)
    rng=np.random.default_rng(20261013)
    shuffled=np.r_[rng.permutation(r[:split]),rng.permutation(r[split:])]
    signs=r*rng.choice([-1,1],len(r))
    assert np.array_equal(abs(signs),abs(r))
    controls=[]
    for name,x in [('Shuffled returns',shuffled),('Random signs, exact magnitudes',signs)]:
        w,_,_=cwt(x);controls.append((name,x,stft(x),w))
    wt=np.arange(len(r))[::HOP]
    delayed=np.full((len(PERIODS),len(wt)),np.nan+1j*np.nan)
    for j,scale in enumerate(scales):
        centers=wt-int(np.ceil(8*scale))-4;valid=centers>=0
        delayed[j,valid]=wr[j,centers[valid]]
    refs,_=plot_all(a.out,dates,close,split,event,r,ps,specs,wp,wr,delayed,scales,controls)
    tests=checks(r,split,specs[512],wr,scales,event)
    actual=stats(r[:split])
    surrogate={};arrays={}
    for kind in ['shuffle','random_sign']:
        metrics=[]
        for i in range(49):
            x=rng.permutation(r[:split]) if kind=='shuffle' else r[:split]*rng.choice([-1,1],split)
            metrics.append(stats(x))
        v=np.array([m['nonoverlap_phase_continuation_cosine'] for m in metrics])
        shares=np.array([m['mean_largest_frequency_power_share'] for m in metrics])
        surrogate[kind]={'phase_median':float(np.median(v)),'phase_interval':np.quantile(v,[.025,.975]).tolist(),
                         'peak_share_median':float(np.median(shares)),'peak_share_interval':np.quantile(shares,[.025,.975]).tolist()}
        arrays[f'{kind}_phase_scores']=v
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for kind,color in [('shuffle','#4687cc'),('random_sign','#df813d')]:
        ax[0].hist(arrays[f'{kind}_phase_scores'],bins=14,alpha=.55,label=kind,color=color)
    ax[0].axvline(actual['nonoverlap_phase_continuation_cosine'],color='black',label='Actual June')
    ax[0].set_xlabel('Weighted cos of phase error: non-overlapping windows');ax[0].legend(fontsize=8)
    for j,key in enumerate(['signed_return_autocorrelation','absolute_return_autocorrelation'],1):
        lags=[1,5,15,60];ax[j].plot(range(4),[actual[key][str(k)] for k in lags],'o-',label='Actual')
        ss=stats(signs[:split]);ax[j].plot(range(4),[ss[key][str(k)] for k in lags],'o--',label='Random signs')
        ax[j].set_xticks(range(4),lags);ax[j].set_xlabel('Lag (candles)');ax[j].set_title(key.replace('_',' '));ax[j].legend()
    fig.savefig(a.out/'diagnostics.png',dpi=150);plt.close(fig)
    summary={'train_candles':split,'following_week_candles':len(r)-split,
             'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
             'time_axis':'consecutive observed candles; all gaps removed; no fill',
             'stft_windows':[512,2048],'hop':HOP,'wavelet':WAVELET,'periods':PERIODS.tolist(),
             'centered_cwt_uses_future':True,'gpu_used':False,'seed':20261013,
             'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'pywavelets':pywt.__version__},
             'train_statistics':actual,'surrogates':surrogate,'surrogate_repetitions_each':49,
             'largest_train_event':{'index':event,'timestamp':str(dates[event]),'close_change_usd':float(close[event]-close[event-1])},
             'checks':tests}
    arrays.update({'timestamp_minutes':dates.astype('int64'),'close':close,'returns_bps':r,'train_candles':np.array(split),
                   'cwt_times':wt,'cwt_periods':period,'cwt_scales':scales,'cwt_logprice':wp[:,::HOP],
                   'cwt_returns':wr[:,::HOP],'cwt_delayed':delayed,'cwt_train_reference_power':refs,
                   'shuffled_returns':shuffled,'random_sign_returns':signs})
    for length,s in specs.items():
        arrays[f'stft_{length}']=s['z'];arrays[f'stft_{length}_end']=s['end'];arrays[f'stft_{length}_period']=s['period']
    np.savez_compressed(a.out/'coefficients.npz',**arrays)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.out/'checks.json').write_text(json.dumps(tests,indent=2)+'\n')
    report(a.out,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
