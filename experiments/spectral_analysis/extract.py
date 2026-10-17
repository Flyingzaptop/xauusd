"""Read every coefficient and every row; produce bounded-memory research features."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,h5py,json,time
from pathlib import Path
from scipy.ndimage import maximum_filter1d, minimum_filter1d
ROOT=Path('/workspace/xauusd/reports/causal_spectral_year_2021_20261012')
OUT=Path('/workspace/xauusd/reports/spectral_analysis_20261012'); OUT.mkdir(exist_ok=True)
METHODS=['stft_128','stft_512','stft_8192','wavelet_returns','wavelet_logprice']
H=[15,60,120]
def rolling(x,w):
 c=np.r_[0,np.cumsum(x,dtype=float)]; out=np.full(len(x),np.nan);out[w-1:]=(c[w:]-c[:-w])/w;return out

def main():
 started=time.time(); a=np.load(ROOT/'input_context.npz'); ts=a['timestamp'];c=a['close'];src=a['source_row']; lp=1e4*np.log(c);r=np.r_[0,np.diff(lp)]
 take=(ts>=np.datetime64('2021-01-01').astype('datetime64[s]').astype(int))&(ts<np.datetime64('2022-01-01').astype('datetime64[s]').astype(int));ix=np.flatnonzero(take);n=len(ix)
 raw=np.load('/workspace/xauusd/reports/pcgrad_mse_month_20261009/dataset.npz');op=np.exp(raw['log_prices'][src,1]);hi=np.exp(raw['log_prices'][src,0]);lo=np.exp(raw['log_prices'][src,3]);del raw
 months=np.array([int(s[5:7]) for s in ts[ix].astype('datetime64[s]').astype(str)]); split=np.where(months<=6,0,np.where(months<=9,1,2))
 vol=np.sqrt(np.maximum(rolling(r*r,60),1e-10)); feats=[];names=[]
 def add(name,v): names.append(name);feats.append(np.asarray(v[ix],np.float32))
 for w in [1,5,15,30,60,120,240,512,2048,8192]:
  past=np.r_[np.full(w,np.nan),lp[w:]-lp[:-w]];add(f'base.momentum_{w}',past/(vol*np.sqrt(w)))
 for w in [5,15,60,240,1440]:add(f'base.logvol_{w}',np.log(np.sqrt(np.maximum(rolling(r*r,w),1e-10))))
 for w in [15,60,240,1440]:
  mx=maximum_filter1d(c,size=w,origin=(w-1)//2,mode='nearest');mn=minimum_filter1d(c,size=w,origin=(w-1)//2,mode='nearest');add(f'base.position_{w}',(c-mn)/np.maximum(mx-mn,.01));add(f'base.range_{w}',(mx-mn)/c*1e4/(vol*np.sqrt(w)))
 sec=ts%86400
 for k in [1,2,3]:add(f'base.tod_sin{k}',np.sin(sec/86400*2*np.pi*k));add(f'base.tod_cos{k}',np.cos(sec/86400*2*np.pi*k))
 add('base.body',(c-op)/c*1e4/vol);add('base.upper_wick',(hi-np.maximum(c,op))/c*1e4/vol);add('base.lower_wick',(np.minimum(c,op)-lo)/c*1e4/vol)
 y=np.full((n,3),np.nan);pnl=y.copy();rv=y.copy();valid=np.zeros((n,3),bool)
 gaps=np.r_[0,np.cumsum(np.r_[False,np.diff(ts)!=60])]
 csq=np.r_[0,np.cumsum(r*r)]
 for j,h in enumerate(H):
  ok=ix+h<len(c);q=ix[ok];cont=(gaps[q+h+1]-gaps[q+1])==0
  valid[ok,j]=cont
  pnl[ok,j]=c[q+h]-op[q+1];y[ok,j]=1e4*np.log(c[q+h]/op[q+1])/(vol[q]*np.sqrt(h));rv[ok,j]=np.sqrt((csq[q+h+1]-csq[q+1])/h)
 np.savez_compressed(OUT/'market.npz',timestamp=ts[ix],close=c[ix],open=op[ix],high=hi[ix],low=lo[ix],returns=r[ix],vol=vol[ix],source_row=src[ix],split=split,months=months,y=y,pnl=pnl,rv=rv,valid=valid)
 base=np.column_stack(feats);np.save(OUT/'base.npy',base);names_base=names.copy();del feats
 specnames=[]; axes={};moments={}; total=0;matrix=None
 for fpath in sorted((ROOT/'months').glob('*.h5')):
  with h5py.File(fpath) as f:
   nn=len(f['candles/timestamp']); ss=int(split[total]); arrays=[]; localnames=[]
   for method in METHODS:
    g=f[method];p=g['periods'][:];ref=g['reference_power'][:];axes[method]=p.tolist();nf=len(p)
    # full component screening uses ALL bins and ALL valid M1 rows
    if method not in moments:moments[method]=np.zeros((3,3,3,8,nf),float)
    chosen=np.arange(nf) if method.startswith('wavelet') else np.unique(np.argmin(abs(np.log(p[:,None]/np.geomspace(p[0],p[-1],24)[None,:])),axis=0))
    bands=[(2**b,2**(b+1)) for b in range(2,14) if np.any((p>=2**b)&(p<2**(b+1)))]
    cols=3*len(chosen)+4*len(bands); out=np.empty((nn,cols),np.float32)
    for b0 in range(0,nn,2048):
     b1=min(b0+2048,nn);z=g['coefficients'][b0:b1];power=abs(z)**2;en=np.log1p(power/ref);unit=z/np.maximum(abs(z),1e-30)
     if method.startswith('stft'):unit*=np.exp(2j*np.pi*(int(g.attrs['window_length'])-1)/p)
     variants=[en,unit.real,unit.imag]
     ids=np.arange(total+b0,total+b1)
     for hj in range(3):
      future_ids=np.minimum(ids+H[hj],n-1);good=valid[ids,hj]&(ids+H[hj]<n)&(split[future_ids]==split[ids]);yy=y[ids[good],hj];aa=abs(yy)
      for typ,x0 in enumerate(variants):
       x=x0[good];v=moments[method][ss,hj,typ];v[0]+=len(yy);v[1]+=x.sum(0);v[2]+=(x*x).sum(0);v[3]+=yy.sum();v[4]+=(yy*yy).sum();v[5]+=yy@x;v[6]+=aa.sum();v[7]+=aa@x
     features=[en[:,chosen],unit.real[:,chosen],unit.imag[:,chosen]]
     for low,high in bands:
      k=(p>=low)&(p<high);w=power[:,k]/ref[k];den=np.maximum(w.sum(1),1e-30);u=unit[:,k];weights=w/den[:,None]
      features.extend([np.log1p(w.mean(1))[:,None],(weights*u.real).sum(1)[:,None],(weights*u.imag).sum(1)[:,None],(-np.sum(weights*np.log(np.maximum(weights,1e-30)),axis=1)/np.log(max(2,k.sum())))[:,None]])
     out[b0:b1]=np.column_stack(features)
    for typ in ['energy','cos','sin']:
     localnames.extend([f'{method}.{typ}.p{p[k]:.5f}' for k in chosen])
    for low,high in bands:localnames.extend([f'{method}.{typ}.band{low}_{high}' for typ in ['energy','cos','sin','entropy']])
    arrays.append(out)
   block=np.column_stack(arrays)
   if matrix is None:
    matrix=np.lib.format.open_memmap(OUT/'spectral.npy',mode='w+',dtype='float32',shape=(n,block.shape[1]));specnames=localnames
   assert localnames==specnames
   matrix[total:total+nn]=block;total+=nn;matrix.flush();print(json.dumps({'month':fpath.stem,'rows':total,'features':len(specnames),'seconds':round(time.time()-started,1)}),flush=True)
 np.savez_compressed(OUT/'full_component_moments.npz',**moments)
 (OUT/'features.json').write_text(json.dumps({'base':names_base,'spectral':specnames,'axes':axes,'rows':n,'read_all_complex_coefficients':True,'elapsed_seconds':time.time()-started},indent=2))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
