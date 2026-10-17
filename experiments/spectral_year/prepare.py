"""Prepare every observed candle of 2021, without scientific analysis or images."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve
from core import PERIODS,LENGTHS,kernel,bank,signal,stft_at,bins


BATCH=1024


def stamp(value):return int(np.datetime64(value,'s').astype('int64'))


def hashes(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dataset(group,name,shape,dtype):
    chunks=(min(512,shape[0]),min(128,shape[1])) if len(shape)==2 else (min(4096,shape[0]),)
    return group.create_dataset(name,shape=shape,dtype=dtype,chunks=chunks,
                                compression='gzip',compression_opts=1,shuffle=True,fletcher32=True)


def groups(f,n,refs):
    result={}
    for length in LENGTHS:
        name=f'stft_{length}';g=f.create_group(name);per=length/bins(length)
        g.create_dataset('periods',data=per);g.create_dataset('reference_power',data=refs[name])
        g.attrs['phase_reference']='local trailing-window index 0; end phase is derivable from coefficients'
        g.attrs['window_length']=length;g.attrs['hop']=1
        result[name]={key:dataset(g,key,(n,len(per)),dtype) for key,dtype in
                      [('coefficients','c16'),('power','f8'),('phase','f8')]}
    for name in ['wavelet_returns','wavelet_logprice']:
        g=f.create_group(name);g.create_dataset('periods',data=PERIODS)
        g.create_dataset('reference_power',data=refs[name]);g.attrs['phase_reference']='one-sided kernel, lag 0 = current CLOSED candle'
        g.attrs['hop']=1
        result[name]={key:dataset(g,key,(n,len(PERIODS)),dtype) for key,dtype in
                      [('coefficients','c16'),('power','f8'),('phase','f8')]}
    return result


def save_values(d,sl,z):
    assert z.dtype==np.complex128 and np.isfinite(z).all()
    d['coefficients'][sl]=z;d['power'][sl]=abs(z)**2;d['phase'][sl]=np.angle(z)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);(a.out/'months').mkdir(exist_ok=True)
    start_clock=time.monotonic()
    with np.load(a.source) as z:
        ts=z['timestamp'];close=z['close'];source_rows=z['source_row']
    assert np.all(np.diff(ts)>0) and np.all(ts%60==0) and np.isfinite(close).all() and np.all(close>0)
    cal_begin=int(np.searchsorted(ts,stamp('2020-12-01')))
    begin=int(np.searchsorted(ts,stamp('2021-01-01')));end=int(np.searchsorted(ts,stamp('2022-01-01')))
    assert cal_begin>=len(kernel(PERIODS[-1])) and begin>cal_begin and end==len(ts)
    r,lp=signal(close);total=end-begin
    source_hash=hashes(a.source)
    config={'year':2021,'step':1,'stft_windows':LENGTHS,'wavelet_periods':PERIODS.tolist(),
            'wavelet':'one-sided DC-corrected L2 Morlet, bandwidth 1.5, support ceil(8*period)+1',
            'normalization_period':['2020-12-01','2021-01-01'],'context_from':'2020-08-01',
            'source_sha256':source_hash,'coefficients_dtype':'complex128','derived_dtype':'float64',
            'time_axis':'consecutive observed candles, all gaps removed; no filling',
            'timestamp_convention':'minute-start assumed; availability=timestamp+60 seconds',
            'stft_frequency_min_period':4,'images_created':False,'scientific_analysis_performed':False,
            'gpu_used':False,'compression':'lossless HDF5 gzip level1 + byte shuffle + fletcher32'}
    config['numeric_core_sha256']=hashes(Path(__file__).with_name('core.py'))
    (a.out/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    def progress(stage,**extra):
        status={'stage':stage,'elapsed_seconds':round(time.monotonic()-start_clock,2),'total_candles':total,**extra}
        (a.out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    refs={}
    progress('normalization_pre_2021')
    # Normalization is actually computed on the pre-year prefix, not merely
    # indexed from a transform that already contains the following year.
    for name,x in [('wavelet_returns',r),('wavelet_logprice',lp)]:
        w=bank(x[:begin]);refs[name]=np.maximum(np.median(abs(w[cal_begin:begin])**2,axis=0),1e-20);del w
    for length in LENGTHS:
        power=np.empty((begin-cal_begin,len(bins(length))),dtype=np.float64)
        for start in range(cal_begin,begin,BATCH):
            ends=np.arange(start,min(begin,start+BATCH));z=stft_at(r[:begin],ends,length)
            power[start-cal_begin:start-cal_begin+len(ends)]=abs(z)**2
        refs[f'stft_{length}']=np.maximum(np.median(power,axis=0),1e-20);del power
    np.savez(a.out/'normalization.npz',**refs)
    progress('wavelet_returns')
    wr=bank(r,lambda completed:progress('wavelet_returns',scales_completed=completed))
    progress('wavelet_logprice')
    wp=bank(lp,lambda completed:progress('wavelet_logprice',scales_completed=completed))
    months=[];day_index=[]
    for month in np.arange(np.datetime64('2021-01','M'),np.datetime64('2022-01','M'),dtype='datetime64[M]'):
        lo=int(np.searchsorted(ts,stamp(str(month))))
        hi=int(np.searchsorted(ts,stamp(str(month+np.timedelta64(1,'M')))))
        n=hi-lo;name=str(month)+'.h5';final=a.out/'months'/name;partial=final.with_suffix('.partial.h5')
        progress('write_month',month=str(month),completed_candles=lo-begin,month_candles=n)
        if final.exists():
            with h5py.File(final,'r') as f:
                assert f.attrs['source_sha256']==source_hash and f.attrs['complete'] and len(f['candles/timestamp'])==n
                assert f.attrs['protocol_json']==json.dumps(config)
        else:
            with h5py.File(partial,'w') as f:
                f.attrs['source_sha256']=source_hash;f.attrs['complete']=False;f.attrs['protocol_json']=json.dumps(config)
                cg=f.create_group('candles')
                fields={'timestamp':ts[lo:hi],'available_timestamp':ts[lo:hi]+60,
                        'close':close[lo:hi],'returns_bps':r[lo:hi],'logprice_relative_bps':lp[lo:hi],
                        'source_row':source_rows[lo:hi],'global_row':np.arange(lo-begin,hi-begin),
                        'gap_seconds_before':ts[lo:hi]-ts[lo-1:hi-1]-60}
                for key,x in fields.items():
                    d=dataset(cg,key,x.shape,x.dtype);d[:]=x
                d=groups(f,n,refs)
                for start in range(lo,hi,BATCH):
                    stop=min(hi,start+BATCH);ends=np.arange(start,stop);sl=slice(start-lo,stop-lo)
                    for length in LENGTHS:save_values(d[f'stft_{length}'],sl,stft_at(r,ends,length))
                    save_values(d['wavelet_returns'],sl,wr[start:stop]);save_values(d['wavelet_logprice'],sl,wp[start:stop])
                    if (start-lo)//BATCH%8==7:
                        progress('write_month',month=str(month),completed_candles=stop-begin,month_candles=n)
                f.attrs['complete']=True
            partial.replace(final)
        months.append({'month':str(month),'file':'months/'+name,'rows':n,'global_start':lo-begin,'global_stop':hi-begin,
                       'bytes':final.stat().st_size,'sha256':hashes(final)})
        for day in np.arange(month.astype('datetime64[D]'),(month+np.timedelta64(1,'M')).astype('datetime64[D]')):
            da=int(np.searchsorted(ts,stamp(str(day))));db=int(np.searchsorted(ts,stamp(str(day+np.timedelta64(1,'D')))))
            day_index.append({'date':str(day),'file':'months/'+name,'row_start':da-lo,'row_stop':db-lo,'candles':db-da,
                              'global_start':da-begin,'global_stop':db-begin})
        progress('month_complete',month=str(month),completed_candles=hi-begin,file_bytes=final.stat().st_size)
    with (a.out/'days.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(day_index[0]));writer.writeheader();writer.writerows(day_index)
    summary={'protocol':config,'year_candles':total,'pre_year_context_candles':begin,'calibration_candles':begin-cal_begin,
             'first_timestamp':str(np.datetime64(int(ts[begin]),'s')),'last_timestamp':str(np.datetime64(int(ts[end-1]),'s')),
             'months':months,'total_hdf5_bytes':sum(m['bytes'] for m in months),'elapsed_seconds':time.monotonic()-start_clock,
             'versions':{k:importlib.metadata.version(k) for k in ['numpy','scipy','h5py']},
             'stft_frequency_counts':{str(l):len(bins(l)) for l in LENGTHS},'wavelet_scale_count':len(PERIODS)}
    (a.out/'dataset.json').write_text(json.dumps(summary,indent=2)+'\n')
    progress('calculation_complete',completed_candles=total,total_hdf5_bytes=summary['total_hdf5_bytes'])


if __name__=='__main__':main()
