"""Representation sanity check: stable-looking phase can be induced by overlap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,numpy as np,json
from scipy.signal import fftconvolve,windows
sys.path.insert(0,'/workspace/xauusd/experiments/spectral_year')
from core import kernel,PERIODS,stft_at
from analyze import OUT,save,corr
ROOT='/workspace/xauusd/reports/causal_spectral_year_2021_20261012/'
a=np.load(ROOT+'input_context.npz');r=np.r_[0,np.diff(1e4*np.log(a['close']))];ts=a['timestamp'];test=(ts>=np.datetime64('2021-10-01').astype('datetime64[s]').astype(int))&(ts<np.datetime64('2022-01-01').astype('datetime64[s]').astype(int));ix=np.flatnonzero(test);ix=ix[ix+60<len(r)]
specs=[]
for p0 in [64,256,1024,8192]:
 p=float(PERIODS[np.argmin(abs(np.log(PERIODS/p0)))]);specs.append(('wavelet_returns',p,kernel(p)))
for L,p in [(128,32),(512,128),(8192,512),(8192,8192)]:
 q=np.arange(L);w=windows.hann(L,sym=False);v=w*np.exp(-2j*np.pi*q/p);v-=v.sum()/L;g=v[::-1];specs.append((f'stft_{L}',float(p),g))
rng=np.random.default_rng(20261014);results={};checks=[]
for name,p,g in specs:
 rows=[]
 for rep in range(20):
  x=r if rep==0 else r*rng.choice([-1,1],len(r));z=fftconvolve(x,g,'full')[:len(x)];phase=z/np.maximum(abs(z),1e-30);future=np.r_[np.cumsum(x)[60:]-np.cumsum(x)[:-60],np.zeros(60)];cumulative=np.r_[0,np.cumsum(x*x)];future_rv=np.sqrt(np.maximum((cumulative[ix+61]-cumulative[ix+1])/60,1e-20));lag=int(round(p))
  row={'phase_advance_agreement_lag1':np.real(phase[ix]*np.conj(phase[ix-1])*np.exp(-2j*np.pi/p)).mean(),'phase_advance_agreement_one_period':np.real(phase[ix]*np.conj(phase[ix-lag])*np.exp(-2j*np.pi*lag/p)).mean(),'phase_sin_future60_corr':corr(phase[ix].imag,future[ix]),'energy_future_rv_corr':corr(np.log1p(abs(z[ix])**2),np.log(future_rv))};rows.append(row)
  if rep==0 and name.startswith('stft'):
   L=int(name.split('_')[1]);ends=ix[[0,100,1000,-1]];zz=stft_at(r,ends,L);periods=L/(np.arange(1 if L==8192 else 2,L//4+1)[::-1]);j=np.argmin(abs(periods-p));err=float(np.max(abs(zz[:,j]-z[ends])));assert err<1e-7;checks.append(err)
 results[f'{name}_p{p:.5f}']={'actual':rows[0],'random_sign_controls':{'n':19,**{key:{'median':float(np.median([row[key] for row in rows[1:]])),'min':float(min(row[key] for row in rows[1:])),'max':float(max(row[key] for row in rows[1:]))} for key in rows[0]}}};print(name,p,flush=True)
save('phase_controls.json',{'purpose':'diagnostic only; 19 magnitude-preserving random-sign paths, not formal multiple-testing adjusted inference; no strategy selected here','test_interval':'Oct-Dec2021','stft_convolution_max_error':max(checks),'results':results})
