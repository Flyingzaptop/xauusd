"""Reproduce the prepared 2021 coefficients exactly, then create causal model inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter1d,minimum_filter1d
from core import PERIODS,kernel,bins,stft_at
P=Path('/content/spectral');P.mkdir(exist_ok=True);t0=time.time()
def save(n,x):(P/n).write_text(json.dumps(x,indent=2))
def roll(x,w):
 c=np.r_[0,np.cumsum(x,dtype=float)];out=np.zeros(len(x));out[w-1:]=(c[w:]-c[:-w])/w;out[:w-1]=out[w-1];return out
raw=np.load(P/'input.npz');ts=raw['timestamp'];price=raw['ohlc'];c=price[:,2];lp=1e4*np.log(c);r=np.r_[0,np.diff(lp)];start=int(np.searchsorted(ts,np.datetime64('2021-01-01').astype('datetime64[s]').astype(int)))-512;N=len(c)-start;ref=np.load(P/'reference.npz');gold=np.load(P/'golden.npz');goldix=gold['rows']+512
periods=np.r_[*[l/bins(l) for l in [128,512,8192]],PERIODS,PERIODS];sizes=[31,127,2048,75,75];methods=['stft_128','stft_512','stft_8192','wavelet_returns','wavelet_logprice'];zbase=np.lib.format.open_memmap(P/'spectral.npy',mode='w+',dtype='float16',shape=(N,2356,3));extras=[];enames=[];errors={};offset=0
for method,nf in zip(methods,sizes):
 p=periods[offset:offset+nf];reference=ref[method];waves=method.startswith('wavelet')
 if waves:
  x=r if method=='wavelet_returns' else lp-lp[0];coeff=np.empty((N,nf),complex)
  for j,p0 in enumerate(p):coeff[:,j]=fftconvolve(x,kernel(p0),'full')[start:len(x)]
 else:
  L=int(method.split('_')[1]);coeff=np.empty((N,nf),complex)
  for b in range(0,N,1024):coeff[b:b+1024]=stft_at(r,np.arange(start+b,min(start+b+1024,len(r))),L)
 errors[method]=float(np.max(abs(coeff[goldix]-gold[method])));assert np.allclose(coeff[goldix],gold[method],rtol=1e-9,atol=1e-8),errors
 power=abs(coeff)**2/reference;phase=coeff/np.maximum(abs(coeff),1e-30)
 if not waves:phase*=np.exp(2j*np.pi*(L-1)/p)
 zbase[:,offset:offset+nf,0]=np.log1p(power);zbase[:,offset:offset+nf,1]=phase.real;zbase[:,offset:offset+nf,2]=phase.imag
 for low in [2**j for j in range(2,14)]:
  k=(p>=low)&(p<low*2)
  if not k.any():continue
  w=power[:,k];wn=w/np.maximum(w.sum(1,keepdims=True),1e-20);u=phase[:,k]
  features=[np.log1p(w.mean(1)),(wn*u.real).sum(1),(wn*u.imag).sum(1),(-wn*np.log(np.maximum(wn,1e-20))).sum(1)/np.log(max(2,k.sum())),(wn*np.log2(p[k]/low)).sum(1)]
  for name,v in zip(['energy','cos','sin','entropy','centroid'],features):extras.append(v.astype('float32'));enames.append(f'{method}.{name}.{low}')
 offset+=nf;zbase.flush();del coeff,power,phase;print(json.dumps({'prepared':method,'seconds':time.time()-t0}),flush=True)
# Cross-scale and cross-representation phase agreement. All terms are current/past.
for j,p in enumerate(PERIODS):
 a=zbase[:,2206+j,1:].astype('float32');b=zbase[:,2281+j,1:].astype('float32');extras.extend([(a*b).sum(1),a[:,1]*b[:,0]-a[:,0]*b[:,1]]);enames.extend([f'return_price_phase_cos.{p}',f'return_price_phase_sin.{p}'])
 if p<=4096:
  k=int(np.argmin(abs(np.log(PERIODS/(2*p)))));b=zbase[:,2206+k,1:].astype('float32');b2=np.column_stack([b[:,0]**2-b[:,1]**2,2*b[:,0]*b[:,1]]);extras.append((a*b2).sum(1));enames.append(f'harmonic_alignment.{p}')
# Price inputs contain relative displacements/shape only, no absolute price level.
vol=np.sqrt(np.maximum(roll(r*r,60),1e-10));base=[];bn=[]
def add(name,v):bn.append(name);base.append(np.asarray(v[start:],np.float32))
for w in [1,5,15,30,60,120,240,512,2048,8192]:add(f'momentum_{w}',np.r_[np.zeros(w),lp[w:]-lp[:-w]]/(vol*np.sqrt(w)))
for w in [5,15,60,240,1440]:add(f'logvol_{w}',np.log(np.sqrt(np.maximum(roll(r*r,w),1e-10))))
for w in [15,60,240,1440]:
 mx=maximum_filter1d(c,size=w,origin=(w-1)//2,mode='nearest');mn=minimum_filter1d(c,size=w,origin=(w-1)//2,mode='nearest');add(f'position_{w}',(c-mn)/np.maximum(mx-mn,.01));add(f'range_{w}',(mx-mn)/c*1e4/(vol*np.sqrt(w)))
for k in [1,2,3]:add(f'tod_sin{k}',np.sin((ts%86400)/86400*2*np.pi*k));add(f'tod_cos{k}',np.cos((ts%86400)/86400*2*np.pi*k))
add('body',(c-price[:,1])/c*1e4/vol);add('upper_wick',(price[:,0]-np.maximum(c,price[:,1]))/c*1e4/vol);add('lower_wick',(np.minimum(c,price[:,1])-price[:,3])/c*1e4/vol)
add('gap_logseconds',np.log1p(np.r_[0,np.maximum(np.diff(ts)-60,0)]))
base=np.column_stack(base);extra=np.column_stack(extras);del extras
# Feature changes, novelty and shock-reaction descriptors on known history.
old=extra.copy()
for lag in [1,5,30]:
 change=np.zeros_like(old);change[lag:]=old[lag:]-old[:-lag];extra=np.column_stack([extra,change]);enames.extend([f'delta{lag}.{n}' for n in enames[:old.shape[1]]])
# Relative shock magnitude, observed retracement, direction and volatility change.
shock=np.zeros(N);shock[5:]=(lp[start+5:]-lp[start:-5])/(vol[start:-5]*np.sqrt(5));reaction=np.zeros(N);reaction[5:]=shock[:-5];extra=np.column_stack([extra,shock,reaction,shock-reaction]);enames+=['shock5','shock5_lag5','shock_change5']
assert np.isfinite(base).all() and np.isfinite(extra).all();np.save(P/'base.npy',base);np.save(P/'extra.npy',extra.astype('float32'));np.save(P/'periods.npy',periods)
# Fixed event: first CLOSED price to hit +/- max($2, past60bar sigma*sqrt60), by120bars.
ts=ts[start:];price=price[start:];c=price[:,2];v=vol[start:];scale=np.maximum(2,c*v*1e-4*np.sqrt(60));horizons=[15,30,60,120];y=np.zeros((N,4),np.float32);event=np.zeros(N,np.int64);first=np.zeros(N,np.int64);direction=np.zeros(N,np.int64)
for h in range(1,121):
 delta=c[h:]-c[:-h];hit=(abs(delta)>=scale[:-h])&(first[:-h]==0);first[:-h][hit]=h;direction[:-h][hit]=np.sign(delta[hit]).astype(int)
 if h in horizons:y[:-h,horizons.index(h)]=delta/scale[:-h]*np.sqrt(60/h)
bucket=np.searchsorted(horizons,first,side='left');event[first>0]=1+bucket[first>0]*2+(direction[first>0]<0)
dates=ts.astype('datetime64[s]').astype(str);month=np.array([int(s[5:7]) for s in dates]);day=np.array([int(s[8:10]) for s in dates]);dev=(np.arange(N)>=512)&(month<=9);test=(np.arange(N)>=512)&(month>=10)&(np.arange(N)<N-120)
# Validation blocks occur in EVERY development month, not one tiny final sliver.
val=dev&(day>=20)&(day<=22);excluded=np.convolve(val.astype(int),np.ones(2*256+1,dtype=int),'same')>0;train=dev&~excluded
# Purge labels crossing end of September. Test is never used for selection.
bound=int(np.flatnonzero((np.arange(N)>=512)&(month>=10))[0]);train[bound-120:]=False;val[bound-120:]=False;dev[bound-120:]=False
np.savez(P/'targets.npz',y=y,event=event,first=first,direction=direction,scale=scale,timestamp=ts,ohlc=price,train=np.flatnonzero(train),val=np.flatnonzero(val),development=np.flatnonzero(dev),test=np.flatnonzero(test),month=month,source_row=raw['source_row'][start:])
save('prepared.json',{'rows':N,'year_rows':N-512,'base_features':len(bn),'extra_features':extra.shape[1],'base_names':bn,'extra_names':enames,'components':2356,'spectral_base_channels':['log1p_relative_power','cos_phase','sin_phase'],'engineered_component_channels':9,'spectral_channels_total':2356*9,'golden_rows':len(goldix),'golden_max_complex_error':errors,'train_rows':int(train.sum()),'validation_rows':int(val.sum()),'final_train_rows':int(dev.sum()),'test_rows':int(test.sum()),'validation':'days20-22in eachJan-Sep month, symmetric256observedbar embargo','event':'first Close crossing +/-max($2,past60bar_sigma*sqrt60), bins15/30/60/120; NONE otherwise','feature_dtype':'float16 coefficients encoding, float32 context features','seconds':time.time()-t0});print('PREPARED',time.time()-t0,flush=True)
