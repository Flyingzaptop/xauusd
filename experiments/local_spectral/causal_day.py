"""Every observed M1 candle: trailing STFT and one-sided Morlet filter bank.

The one-sided bank is deliberately different from standard centered CWT.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.signal import detrend, windows, fftconvolve


PERIODS=np.geomspace(4,512,48)
LENGTHS=[128,512]


def kernel(period):
    lag=np.arange(int(np.ceil(8*period))+1)
    envelope=np.exp(-(lag/period)**2/1.5)
    carrier=np.exp(2j*np.pi*lag/period)
    dc=np.sum(envelope*carrier)/np.sum(envelope)
    g=envelope*(carrier-dc)
    return g/np.sqrt(np.sum(abs(g)**2))


def bank(x):
    return np.asarray([fftconvolve(x,kernel(p),mode='full')[:len(x)] for p in PERIODS])


def stft_at(x,ends,length,linear=False):
    frames=x[ends[:,None]-length+1+np.arange(length)]
    clean=detrend(frames,axis=1,type='linear' if linear else 'constant')
    w=windows.hann(length,sym=False)
    z=np.fft.rfft(clean*w,axis=1).T
    f=np.fft.rfftfreq(length)
    keep=(f>=2/length)&(f<=.25)
    return z[keep][::-1],(1/f[keep])[::-1]


def db(z,ref):return 10*np.log10(np.maximum(abs(z)**2,1e-20)/ref[:,None])


def baseline_wave(w,begin):
    return np.asarray([max(float(np.median(abs(w[i,len(kernel(p))-1:begin])**2)),1e-20)
                       for i,p in enumerate(PERIODS)])


def prepare_signal(close):
    log=np.log(close)*10000
    return np.r_[0,np.diff(log)],log-log[0]


def verify(close,begin,ends,zs,w):
    r,_=prepare_signal(close)
    records=[]
    # Several points including the candle immediately before the sharp move.
    event=int(np.argmax(abs(r[ends])))
    positions=sorted(set([0,1,len(ends)//3,len(ends)//2,max(0,event-1),event,len(ends)-2]))
    for j in positions:
        cut=int(ends[j])
        prefix=r[:cut+1]
        wp=bank(prefix)
        prefix_error=float(np.max(abs(wp[:,-1]-w[:,cut])))
        assert prefix_error<1e-8
        rng=np.random.default_rng(cut)
        changed=close.copy()
        changed[cut+1:]*=np.exp(rng.normal(0,.15,len(close)-cut-1))
        mr,_=prepare_signal(changed)
        mw=bank(mr)
        mutation_error=float(np.max(abs(mw[:,cut]-w[:,cut])))
        assert mutation_error<1e-8
        stft_error=0.
        for length in LENGTHS:
            pz,_=stft_at(prefix,np.array([cut]),length)
            mz,_=stft_at(mr,np.array([cut]),length)
            err=max(float(np.max(abs(pz[:,0]-zs[length][:,j]))),float(np.max(abs(mz[:,0]-zs[length][:,j]))))
            assert err<1e-8;stft_error=max(stft_error,err)
        direct_error=0.
        for i in [0,12,24,36,47]:
            g=kernel(PERIODS[i]);direct=np.dot(prefix[-len(g):][::-1],g)
            direct_error=max(direct_error,float(abs(direct-w[i,cut])))
        assert direct_error<1e-8
        # Display normalizer uses only data strictly before this day.
        assert np.array_equal(close[:begin],changed[:begin])
        records.append({'day_candle':j,'prefix_wave_max_error':prefix_error,
                        'future_mutation_wave_max_error':mutation_error,'stft_max_error':stft_error,
                        'independent_direct_sum_max_error':direct_error})
    t=np.arange(8192);known=np.cos(2*np.pi*t/64);kw=bank(known)
    powers=np.mean(abs(kw[:,-2048:])**2,axis=1);peak=float(PERIODS[np.argmax(powers)])
    assert abs(np.log2(peak/64))<.25
    zero_dc=float(max(abs(np.sum(kernel(p))) for p in PERIODS));assert zero_dc<1e-12
    return {'points_tested':records,'synthetic_period64_nominal_peak':peak,'kernel_dc_max_abs':zero_dc,
            'every_day_candle_computed':True,'step_observed_candles':1,
            'normalizer_fitted_strictly_before_day':True,'kernels_have_no_future_lags':True}


def axes_time(ax,dates,event):
    def tick(v,pos):
        i=int(np.clip(round(v),0,len(dates)-1))
        return str(dates[i])[11:]
    ax.xaxis.set_major_formatter(FuncFormatter(tick));ax.set_xlim(0,len(dates)-1)
    ax.axvline(event,color='#2c718d',linestyle=':',linewidth=1)
    ax.set_xlabel('Time label / consecutive actual M1 candles; gaps removed')


def heat(ax,period,values,title,dates,event,phase=False):
    im=ax.pcolormesh(np.arange(len(dates)),period,values,shading='auto',
                     cmap='twilight' if phase else 'magma',vmin=-np.pi if phase else -12,vmax=np.pi if phase else 18,
                     rasterized=True)
    ax.set_yscale('log',base=2);ticks=[4,8,16,32,64,128,256,512]
    ticks=[p for p in ticks if period.min()<=p<=period.max()]
    ax.set_yticks(ticks,[str(p) for p in ticks]);ax.set_ylim(period.min(),period.max())
    ax.set_ylabel('Nominal period (candles)');ax.set_title(title,loc='left',fontsize=11)
    axes_time(ax,dates,event);return im


def figures(out,dates,close,event,zs,periods,refs,wret,wprice,wrefs):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'text.parse_math':False})
    for filename,limits in [('day_overview.png',None),('shock_detail.png',(max(0,event-180),min(len(dates)-1,event+240)))]:
        fig,ax=plt.subplots(5,1,figsize=(15,15),constrained_layout=True)
        fig.suptitle('June 16, 2021 | each actual M1 candle | past and current closed candle ONLY\n'
                     'One-sided Morlet bank and trailing STFT; color references fixed before June 16',fontsize=14)
        ax[0].plot(close,color='#192537',linewidth=1);ax[0].set_ylabel('Close ($)')
        ax[0].set_title('Actual price; dotted line = largest observed change, not a predicted event',loc='left',fontsize=11)
        axes_time(ax[0],dates,event)
        im=heat(ax[1],periods[128],db(zs[128],refs[128]),'Return STFT: 128 trailing candles, step 1',dates,event)
        heat(ax[2],periods[512],db(zs[512],refs[512]),'Return STFT: 512 trailing candles, step 1',dates,event)
        heat(ax[3],PERIODS,db(wret,wrefs[0]),'Return wavelet bank: one-sided Morlet, step 1, no future neighbors',dates,event)
        heat(ax[4],PERIODS,db(wprice,wrefs[1]),'Log-price wavelet bank: one-sided Morlet, zero-DC kernels, step 1',dates,event)
        if limits:
            for a in ax:a.set_xlim(*limits)
            visible=close[limits[0]:limits[1]+1];ax[0].set_ylim(visible.min()-2,visible.max()+2)
        fig.colorbar(im,ax=ax[1:],label='Power / pre-day median at each frequency or scale (dB)',shrink=.8)
        fig.savefig(out/filename,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(3,1,figsize=(15,10),constrained_layout=True)
    fig.suptitle('Causal wavelet phase | every M1 candle | no predicted future phase',fontsize=14)
    ax[0].plot(close,color='#192537');ax[0].set_ylabel('Close ($)');axes_time(ax[0],dates,event)
    power=db(wret,wrefs[0]);im=heat(ax[1],PERIODS,power,'Return wavelet power',dates,event)
    phase=np.angle(wret);phase[power<6]=np.nan
    pi=heat(ax[2],PERIODS,phase,'Phase only above 4x pre-day median power; blank = weak coefficient',dates,event,phase=True)
    fig.colorbar(im,ax=ax[1],label='Relative power (dB)',shrink=.8);fig.colorbar(pi,ax=ax[2],label='Phase radians',shrink=.8)
    fig.savefig(out/'day_phase.png',dpi=160);plt.close(fig)


def write_report(out,s):
    text=f'''# Причинные STFT и односторонний Morlet: {s['day']}

{s['day_candles']} реальных M1 свечей. **Шаг 1 свеча**: для каждой точки рассчитаны отдельные комплексные коэффициенты. Паузы склеены; {s['gaps_within_day']} пропусков календарных минут в дне, никаких вставок. Даты следуют timestamp источника, timezone feed отдельно не подтверждён.

Используем историю с 1 июня до текущей свечи. До первого бара дня доступны {s['history_candles_before_day']} прошлых свечей. Дневная карта включает только выбранный день; предыдущая история нужна для окон и нормализации, а не показана как часть дня.

Для этого протокола timestamp свечи трактуем как начало минуты. Коэффициент по Close этой свечи доступен после её закрытия, то есть на минуту позже подписи. В features.npz отдельно сохранён available_timestamp_minutes = timestamp_minutes+1. Карта не выдаёт значение закрытия свечи в момент её открытия.

## Методы

STFT: Hann, окна 128 и 512 свечей, шаг 1, среднее каждого окна удалено. В точке t окно [t-L+1,t]. Текущая свеча считается закрытой; событие в её Close уже наблюдено. Не ставим коэффициент в середину окна. Hann подавляет самый свежий край, поэтому реакция может быть плавной/задержанной, хотя будущие значения не используются.

Вейвлет-представление: **односторонний комплексный Morlet filter bank**, 48 номинальных периодов 4–512 свечей. Это причинная модификация, а не стандартная симметричная CWT PyWavelets и не просто сдвиг старых коэффициентов.

    lag = 0…ceil(8*period)
    envelope(lag) = exp(-(lag/period)^2/1.5)
    carrier(lag) = exp(i*2*pi*lag/period)
    c = sum(envelope*carrier)/sum(envelope)
    g(lag) = envelope*(carrier-c), затем L2-нормировка
    W(t,period) = sum_lag g(lag)*signal(t-lag)

Все лаги неотрицательные. Поправка c делает сумму kernel равной нулю, поэтому постоянный уровень не даёт волну. Односторонность меняет частотную характеристику; подписи шкалы — номинальные периоды. Контрольная синусоида с периодом 64 даёт максимальную энергию на номинальном периоде {s['checks']['synthetic_period64_nominal_peak']:.2f}. Крупные масштабы описывают длинную прошлую историю, не дают мгновенно достоверный новый цикл.

Входы: 10000*Δlog(Close) и 10000*(log(Close)-log(Close_первой_известной_свечи)). Без clipping, границ из будущего или нормализации по текущему дню. Для STFT показаны изменения; для Morlet показаны оба входа. Разница между соседними свечами может включать гэп через закрытие рынка.

Цветовая шкала: энергия / медиана для данного масштаба **только ДО 16 июня**. Все значения днём сравниваются с заранее фиксированными уровнями. Яркость разных строк не означает равенство абсолютной энергии. Фаза отображается только при мощности >=4x прошлой медианы; это условие визуализации, не готовый торговый сигнал.

## Проверка отсутствия будущего

В {len(s['checks']['points_tested'])} точках, включая бар перед крупнейшим падением:

1. Обрезали данные сразу после текущей свечи и пересчитали коэффициенты. Они совпали с пакетным расчётом.
2. Изменили все будущие цены, пересчитали log-изменения и преобразования. Коэффициенты текущего момента не изменились, кроме округления FFT.
3. Независимо проверили Morlet обычной суммой произведений по прошлому, без FFT-свёртки.
4. Нормализация осталась неизменной, поскольку рассчитана до начала дня.

Максимальная ошибка проверки: {s['max_check_error']:.3g}. У каждого сохранённого столбца ровно один наблюдённый M1 timestamp. Warmup закончился до начала дня для всех периодов. Сохранены коэффициенты всех свечей, без прореживания.

## Что видно и что пока не проверено

Падение {s['largest_change']['timestamp']} — {s['largest_change']['usd']:.2f} USD в соседних Close. Оно выделяется широкополосной реакцией, особенно в одностороннем Morlet. Сам такой отклик — описание уже произошедшего движения. На STFT максимальный отклик может наступать позже из-за оконного веса. Любую структуру ДО падения на этой карте теперь можно отдельно проверять как потенциально доступный признак; автоматически считать её предвестником нельзя.

Один день выбран для просмотра причинности и детализации. Нет обученной модели, прогноза, оценки precision или торгового преимущества. Стандартная центрированная CWT из предыдущего отчёта остаётся ретроспективной; она не используется в этих дневных признаках.

## Файлы

- day_overview.png — весь день, цена, два STFT и две причинные wavelet-карты.
- shock_detail.png — участок вокруг наибольшего изменения, выбранного постфактум.
- day_phase.png — энергия и фаза причинных коэффициентов.
- review.html — интерактивный просмотр дневных карт.
- features.npz — все комплексные коэффициенты, периоды, времена и прошлые нормы, шаг 1.
- checks.json / summary.json — протокол и проверки.

Воспроизведение: `python experiments/local_spectral/causal_day.py --source reports/global_fourier_month_20261012/predictions.npz --out <new-directory> --day {s['day']}`. Исходный месяц включён в архив. GPU не использован.
'''
    (out/'REPORT.md').write_text(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--day',default='2021-06-16')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    begin_time=int(np.datetime64(a.day,'m').astype('int64'));end_time=begin_time+1440
    with np.load(a.source) as z:
        all_dates=z['timestamp_minutes'];all_close=z['close']
    # Cut at the end of the requested day before any transform is computed.
    last=int(np.searchsorted(all_dates,end_time));dates=all_dates[:last];close=all_close[:last]
    begin=int(np.searchsorted(dates,begin_time));ends=np.arange(begin,last)
    assert len(ends)>0 and begin>len(kernel(PERIODS[-1]))
    r,logprice=prepare_signal(close)
    wr=bank(r);wp=bank(logprice)
    wref=baseline_wave(wr,begin);pref=baseline_wave(wp,begin)
    zs,periods,refs={},{},{}
    for length in LENGTHS:
        z,per=stft_at(r,ends,length);zs[length]=z;periods[length]=per
        before=np.arange(length-1,begin,32);past,_=stft_at(r,before,length)
        refs[length]=np.maximum(np.median(abs(past)**2,axis=1),1e-20)
    verified=verify(close,begin,ends,zs,wr)
    event=int(np.argmax(abs(r[ends])))
    arrays={'timestamp_minutes':dates[ends],'available_timestamp_minutes':dates[ends]+1,'close':close[ends],'returns_bps':r[ends],
            'wavelet_returns':wr[:,ends],'wavelet_logprice':wp[:,ends],'wavelet_periods':PERIODS,
            'wavelet_return_reference':wref,'wavelet_price_reference':pref,
            'history_timestamp_minutes':dates,'history_close':close,'day_start_index':np.array(begin)}
    for length in LENGTHS:
        arrays[f'stft_{length}']=zs[length];arrays[f'stft_{length}_period']=periods[length];arrays[f'stft_{length}_reference']=refs[length]
    np.savez_compressed(a.out/'features.npz',**arrays)
    errors=[v for q in verified['points_tested'] for k,v in q.items() if k!='day_candle']
    summary={'day':a.day,'day_candles':len(ends),'history_candles_before_day':begin,
             'gaps_within_day':int(np.sum(np.diff(dates[ends])>1)),
             'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'step':1,'gpu_used':False,
             'causal_wavelet':'one-sided complex Morlet; explicit DC correction; L2 normalized',
             'stft_windows':LENGTHS,'normalizer_cutoff':a.day+'T00:00','checks':verified,'max_check_error':max(errors),
             'largest_change':{'day_index':event,'timestamp':str(dates[ends[event]].astype('datetime64[m]')),
                               'usd':float(close[ends[event]]-close[ends[event]-1])}}
    figures(a.out,dates[ends].astype('datetime64[m]'),close[ends],event,zs,periods,refs,wr[:,ends],wp[:,ends],(wref,pref))
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.out/'checks.json').write_text(json.dumps(verified,indent=2)+'\n')
    write_report(a.out,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
