"""Technical verification only: storage, chronology, mathematics, causality."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import fftconvolve
from core import PERIODS,LENGTHS,kernel,signal,stft_at,bins
from dataset import SpectralYear


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--previous-day',type=Path)
    a=p.parse_args();meta=json.loads((a.out/'dataset.json').read_text())
    with np.load(a.out/'input_context.npz') as z:
        ts=z['timestamp'];close=z['close'];source_row=z['source_row']
    with np.load(a.out/'normalization.npz') as z:refs={k:z[k] for k in z.files}
    r,lp=signal(close);begin=meta['pre_year_context_candles'];rows=0;max_transform=0.;max_identity=0.;fixed_points=[]
    for month in meta['months']:
        lo=begin+month['global_start'];hi=begin+month['global_stop'];n=hi-lo
        with h5py.File(a.out/month['file'],'r') as f:
            assert f.attrs['complete'] and np.array_equal(f['candles/timestamp'][:],ts[lo:hi])
            assert np.array_equal(f['candles/available_timestamp'][:],ts[lo:hi]+60)
            for key,expected in [('close',close[lo:hi]),('returns_bps',r[lo:hi]),
                                 ('logprice_relative_bps',lp[lo:hi]),('source_row',source_row[lo:hi]),
                                 ('global_row',np.arange(lo-begin,hi-begin)),
                                 ('gap_seconds_before',ts[lo:hi]-ts[lo-1:hi-1]-60)]:
                assert np.array_equal(f['candles/'+key][:],expected)
            selected=np.unique(np.linspace(0,n-1,64).astype(int));ends=lo+selected
            for method in ['stft_128','stft_512','stft_8192','wavelet_returns','wavelet_logprice']:
                g=f[method];per=PERIODS if method.startswith('wavelet') else int(method[5:])/bins(int(method[5:]))
                assert np.array_equal(g['periods'][:],per)
                assert np.array_equal(g['reference_power'][:],refs[method])
                for key,dtype in [('coefficients','complex128'),('power','float64'),('phase','float64')]:
                    assert g[key].shape==(n,len(per)) and g[key].dtype==np.dtype(dtype)
                c=g['coefficients'][selected,:];power=g['power'][selected,:];phase=g['phase'][selected,:]
                assert np.isfinite(c).all() and np.isfinite(power).all() and np.isfinite(phase).all()
                assert np.allclose(power,abs(c)**2,rtol=1e-14,atol=1e-20)
                assert np.allclose(phase,np.angle(c),rtol=0,atol=1e-14)
                max_identity=max(max_identity,float(np.max(abs(phase-np.angle(c)))))
                if method.startswith('stft'):
                    calc=stft_at(r,ends,int(method[5:]));error=float(np.max(abs(calc-c)))
                    assert error<1e-8;max_transform=max(max_transform,error)
                else:
                    x=r if method=='wavelet_returns' else lp
                    for sample in [0,31,63]:
                        index=int(ends[sample]);j=int(selected[sample])
                        for k in [0,24,47,60,74]:
                            gkernel=kernel(PERIODS[k]);calc=np.dot(x[index-len(gkernel)+1:index+1][::-1],gkernel)
                            error=float(abs(calc-c[sample,k]));assert error<1e-8
                            max_transform=max(max_transform,error)
            fixed_points.append((lo,n,month['file']))
        rows+=n
        print(json.dumps({'verified_month':month['month'],'rows_so_far':rows}),flush=True)
    assert rows==meta['year_candles']==len(ts)-begin
    with (a.out/'days.csv').open() as f:days=list(csv.DictReader(f))
    assert len(days)==365 and sum(int(d['candles']) for d in days)==rows
    previous=0
    for d in days:
        assert int(d['global_start'])==previous
        previous=int(d['global_stop']);assert previous-int(d['global_start'])==int(d['candles'])
    assert previous==rows
    causality=[]
    for record in [fixed_points[0],fixed_points[5],fixed_points[-1]]:
        lo,n,file=record;index=lo+n//2;local=index-lo
        rng=np.random.default_rng(index);changed=close.copy()
        changed[index+1:]*=np.exp(rng.normal(0,.1,len(close)-index-1))
        mr,mp=signal(changed)
        assert np.array_equal(close[:begin],changed[:begin])
        prefix_error=mutation_error=0.
        with h5py.File(a.out/file,'r') as f:
            for method,x,mx in [('wavelet_returns',r,mr),('wavelet_logprice',lp,mp)]:
                for k in [0,47,60,74]:
                    g=kernel(PERIODS[k]);stored=f[method+'/coefficients'][local,k]
                    prefix=fftconvolve(x[:index+1],g,mode='full')[index]
                    mutated=fftconvolve(mx,g,mode='full')[index]
                    prefix_error=max(prefix_error,float(abs(prefix-stored)))
                    mutation_error=max(mutation_error,float(abs(mutated-stored)))
            for length in LENGTHS:
                stored=f[f'stft_{length}/coefficients'][local,:]
                prefix=stft_at(r[:index+1],np.array([index]),length)[0]
                mutated=stft_at(mr,np.array([index]),length)[0]
                prefix_error=max(prefix_error,float(np.max(abs(prefix-stored))))
                mutation_error=max(mutation_error,float(np.max(abs(mutated-stored))))
        assert prefix_error<1e-8 and mutation_error<1e-8
        causality.append({'timestamp':str(np.datetime64(int(ts[index]),'s')),
                          'prefix_error':prefix_error,'future_mutation_error':mutation_error})
    loader=SpectralYear(a.out)
    one=loader.load_day('2021-06-16',methods=['stft_8192','wavelet_returns'],fields=['coefficients','relative_power_db','phase_at_end'],period_range=(32,8192))
    assert len(one['candles']['timestamp'])==1380
    assert loader.load_day('2021-01-01')['candles']['timestamp'].shape==(0,)
    block=next(loader.iter_blocks(block_size=7,methods=['stft_128'],fields=['coefficients']))
    assert len(block['candles']['timestamp'])==7
    daily=None
    if a.previous_day:
        with np.load(a.previous_day) as old:
            day=loader.load_day('2021-06-16',methods=['stft_128','stft_512','wavelet_returns','wavelet_logprice'],fields=['coefficients'])
            assert np.array_equal(day['candles']['timestamp']//60,old['timestamp_minutes'])
            differences={}
            for method in ['stft_128','stft_512','wavelet_returns','wavelet_logprice']:
                key='wavelet_returns' if method=='wavelet_returns' else 'wavelet_logprice' if method=='wavelet_logprice' else method
                actual=day[method]['coefficients'];expected=old[key].T
                if method.startswith('wavelet'):actual=actual[:,:48]
                error=float(np.max(abs(actual-expected)));assert error<1e-8;differences[method]=error
            daily={'rows':1380,'maximum_coefficient_differences':differences,
                   'normalizers_not_compared':'annual normalization fixed before 2021; prior demo fixed before June16'}
    result={'passed':True,'year_rows':rows,'months_verified':12,'calendar_days_indexed':365,
            'technical_sample_rows_per_month':64,'sampled_power_phase_identities_valid':True,
            'sampled_transform_max_error':max_transform,'phase_identity_max_error':max_identity,
            'future_and_prefix_checks':causality,'previous_day_compatibility':daily,
            'loader_day_empty_day_block_checks':True,'all_written_blocks_finite_checked_during_preparation':True,
            'images_created':False,'scientific_analysis_performed':False}
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
