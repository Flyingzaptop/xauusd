"""Numeric-only transforms; no plots, prediction or exploratory statistics."""
import numpy as np
from scipy.signal import fftconvolve, windows

BASE_PERIODS=np.geomspace(4,512,48)
PERIODS=np.r_[BASE_PERIODS,np.geomspace(512,8192,28)[1:]]
LENGTHS=[128,512,8192]


def kernel(period):
    lag=np.arange(int(np.ceil(8*period))+1)
    envelope=np.exp(-(lag/period)**2/1.5)
    carrier=np.exp(2j*np.pi*lag/period)
    dc=np.sum(envelope*carrier)/np.sum(envelope)
    g=envelope*(carrier-dc)
    return g/np.sqrt(np.sum(abs(g)**2))


def signal(close):
    log=np.log(close)*10000
    return np.r_[0,np.diff(log)],log-log[0]


def bank(x,progress=None):
    result=np.empty((len(x),len(PERIODS)),dtype=np.complex128)
    for j,p in enumerate(PERIODS):
        result[:,j]=fftconvolve(x,kernel(p),mode='full')[:len(x)]
        if progress and (j+1)%25==0:progress(j+1)
    return result


def bins(length):
    # Existing 128/512 settings remain unchanged. The added 8192 window
    # includes k=1 so its largest nominal period is exactly 8192 candles.
    return np.arange(1 if length==8192 else 2,length//4+1)[::-1]


def stft_at(x,ends,length):
    frames=np.lib.stride_tricks.sliding_window_view(x,length)[ends-length+1].copy()
    frames-=np.mean(frames,axis=1,keepdims=True)
    frames*=windows.hann(length,sym=False)
    z=np.fft.rfft(frames,axis=1)
    return np.ascontiguousarray(z[:,bins(length)])


def wrap_phase(x):return (x+np.pi)%(2*np.pi)-np.pi
