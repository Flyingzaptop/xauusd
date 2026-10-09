"""Streaming DBN audit and bounded raw-event windows; never reads future events into inputs."""
import argparse
import hashlib
import logging
import re
import time
from datetime import datetime, timezone
import databento as db
import numpy as np
import pandas as pd
from config import *

LOG = logging.getLogger('mbo.prepare')

def sec(s): return int(pd.Timestamp(s, tz='UTC').timestamp())

def candles():
    df = pd.read_parquet(ROOT/'data/M1/xauusd_m1.parquet')
    ts = df.timestamp.to_numpy(dtype='datetime64[s]').astype(np.int64)
    use = (ts >= sec(START)) & (ts < sec(END))
    ts = ts[use]; p = df.loc[use, ['high','open','close','low']].to_numpy(float)
    assert np.all(np.diff(ts)>0) and np.all(ts%60==0)
    assert np.isfinite(p).all() and (p>0).all()
    assert (p[:,0]>=p.max(1)).all() and (p[:,3]<=p.min(1)).all()
    lp = np.log(p); segments=np.cumsum(np.r_[True,np.diff(ts)!=60])
    # Target index j: 512 angle tokens ending j-1; oldest lag endpoint j-567.
    ix=np.arange(CONTEXT+max(LAGS),len(ts))
    ix=ix[segments[ix]==segments[ix-CONTEXT-max(LAGS)]]
    fit=np.flatnonzero(ts<sec(VALID)); scales=[]
    for k in LAGS:
        x=fit[fit>=k]; x=x[segments[x]==segments[x-k]]
        r=lp[x,2]-lp[x-k,2]; scales.append(float(np.median(abs(r-np.median(r)))))
    h,intercept=np.polyfit(np.log(LAGS),np.log(scales),1)
    scale=dict(S0=float(np.exp(intercept)),H=float(h),fit_end_exclusive=VALID)
    angles=np.zeros((len(lp),36),np.float32)
    for n,k in enumerate(LAGS):
        angles[k:,4*n:4*n+4]=(2/np.pi)*np.arctan((lp[k:]-lp[:-k])/(scale['S0']*k**h))
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(exist_ok=True)
    np.savez(OUT/'candles.npz',ts=ts,lp=lp,angles=angles,eligible=ix)
    dump(OUT/'scale.json',scale)
    return ts,lp,ix,scale

def outright_map(md,day):
    date=datetime.fromtimestamp(day*86400,timezone.utc).date()
    result={}
    for name,intervals in md.mappings.items():
        if not re.fullmatch(r'GC[FGHJKMNQUVXZ]\d{1,2}',name): continue
        for v in intervals:
            if v['start_date']<=date<v['end_date']: result[int(v['symbol'])]=name
    return result

def select_contract(volumes):
    return max(volumes,key=lambda k:(volumes[k],k)) if volumes else None

def raw_windows(a,cutoffs):
    """Left-padded raw events strictly before each target-minute opening."""
    end=np.searchsorted(a['ts_recv'],cutoffs,side='left')
    pos=end[:,None]+np.arange(-EVENTS,0)[None,:]
    out=np.zeros(pos.shape,dtype=RAW)
    if len(a):
        safe=np.maximum(pos,0); out[:]=a[safe]
        valid=(pos>=0)&(out['ts_recv']<cutoffs[:,None])
        oldest=np.maximum(cutoffs.astype(np.int64)-MAX_AGE*10**9,0).astype(np.uint64)
        valid &= out['ts_recv'] >= oldest[:,None]
        out[~valid]=np.zeros((),dtype=RAW)
    return out

def price_features(a,cutoffs,s0,last_xau):
    """Five completed GC trade-minute OHLCs; no target-minute trade included."""
    trades=a[a['action']==b'T']; result=np.zeros((len(cutoffs),26),np.float32)
    if not len(trades): return result
    minute=trades['ts_recv']//np.uint64(60*10**9)
    keys,starts=np.unique(minute,return_index=True); ends=np.r_[starts[1:],len(trades)]
    px=np.log(trades['price'].astype(float)/1e9)
    bars=np.column_stack([np.maximum.reduceat(px,starts),px[starts],px[ends-1],np.minimum.reduceat(px,starts)])
    wanted=cutoffs//np.uint64(60*10**9)
    for i,m in enumerate(wanted):
        ix=np.searchsorted(keys,m)-1
        if ix<0:continue
        reference=bars[ix,2]
        for j,k in enumerate(range(5,0,-1)):
            q=np.searchsorted(keys,m-np.uint64(k))
            if q<len(keys) and keys[q]==m-np.uint64(k):
                result[i,4*j:4*j+4]=(bars[q]-reference)/s0;result[i,20+j]=1
        result[i,25]=(reference-last_xau[i])/s0
    return result

def prepare(limit_days=None):
    ts,lp,eligible,scale=candles(); files=sorted((ROOT/'data/MBO').glob('*.mbo.dbn.zst'))
    manifest=dict(status='preparing',mode='past_only_next_M1',context=CONTEXT,max_events=EVENTS,
        max_event_age_seconds=MAX_AGE,train=[START,VALID],validation=[VALID,TEST],test=[TEST,END],
        test_previously_research_exposed=True,bar_timestamp_assumption='M1 timestamps denote interval start; decision at next minute opening',
        sources=[dict(name=p.name,size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files],days=[])
    dump(OUT/'manifest.json',manifest)
    day=None; mapping={}; selected=None; volumes={}; previous_volumes={}; parts=[]; audit={}; last_recv=0; done=0
    def finish():
        nonlocal done
        if day is None:return
        date=datetime.fromtimestamp(day*86400,timezone.utc).date().isoformat()
        ix=eligible[(ts[eligible]//86400)==day]
        rec=dict(date=date,contract=selected,selection='previous observed UTC day trade size, outright only',
                 selection_volumes=previous_volumes,counts=audit.copy(),candidates=len(ix),samples=0)
        if parts and len(ix):
            a=np.concatenate(parts)
            assert np.all(a['ts_recv'][1:]>=a['ts_recv'][:-1])
            cutoff=(ts[ix].astype(np.uint64)*np.uint64(10**9))
            ends=np.searchsorted(a['ts_recv'],cutoff,side='left')
            oldest=np.maximum(cutoff.astype(np.int64)-MAX_AGE*10**9,0).astype(np.uint64)
            begins=np.searchsorted(a['ts_recv'],oldest,side='left')
            enough=ends-begins>=128
            enough &= cutoff-a['ts_recv'][np.maximum(ends-1,0)] <= 60*10**9
            ix=ix[enough];cutoff=cutoff[enough]
            if len(ix):
                path=CACHE/f'{date}.npy'
                mm=np.lib.format.open_memmap(path,mode='w+',dtype=RAW,shape=(len(ix),EVENTS))
                for b in range(0,len(ix),32): mm[b:b+32]=raw_windows(a,cutoff[b:b+32])
                counts=np.count_nonzero(mm['ts_recv'],axis=1)
                first=mm['ts_recv'][np.arange(len(ix)),EVENTS-counts]
                duration=(cutoff-first)/1e9
                last_age=(cutoff-mm['ts_recv'][:,-1])/1e9
                mm.flush(); del mm
                gc=price_features(a,cutoff,scale['S0'],lp[ix-1,2])
                np.savez(CACHE/f'{date}.npz',ix=ix,gc=gc,counts=counts)
                rec.update(samples=len(ix),events_available_min=int(counts.min()),events_available_median=float(np.median(counts)),
                           events_available_max=int(counts.max()),
                           raw_window_seconds_quantiles=np.quantile(duration,[0,.5,.95,1]).tolist(),
                           newest_event_age_seconds_quantiles=np.quantile(last_age,[0,.5,.95,1]).tolist(),
                           file=str(path),bytes=path.stat().st_size)
        manifest['days'].append(rec);done+=1
        manifest.update(processed_days=done,samples=sum(v['samples'] for v in manifest['days']))
        dump(OUT/'manifest.json',manifest);LOG.info('DAY %s',rec)
    for file in files:
        store=db.DBNStore.from_file(file); md=store.metadata
        for batch in store.to_ndarray(count=2_000_000):
            recv=batch['ts_recv']
            assert np.all(recv[1:]>=recv[:-1]),'Nonmonotone receipt timestamps need explicit reordering'
            assert int(recv[0])>=last_recv; last_recv=int(recv[-1])
            days=recv//np.uint64(86400*10**9); cuts=np.r_[0,np.flatnonzero(days[1:]!=days[:-1])+1,len(batch)]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                d=int(days[lo]); a=batch[lo:hi]
                if d!=day:
                    finish()
                    if limit_days and done>=limit_days:
                        manifest['status']='smoke_prepared';dump(OUT/'manifest.json',manifest);return
                    previous_volumes=volumes;selected=select_contract(previous_volumes)
                    day=d;mapping=outright_map(md,day);volumes={};parts=[];audit=dict(records=0,snapshot_or_bad_recv=0,selected_events=0)
                else:
                    mapping=outright_map(md,day)
                audit['records']+=len(a)
                bad=(a['flags']&40)!=0;audit['snapshot_or_bad_recv']+=int(bad.sum())
                valid=~bad & (a['rtype']==160)&(a['price']>0)&(a['price']!=np.iinfo(np.int64).max)
                trades=a[valid&(a['action']==b'T')]
                if len(trades):
                    ids,reverse=np.unique(trades['instrument_id'],return_inverse=True)
                    sums=np.bincount(reverse,weights=trades['size'])
                    for ident,v in zip(ids,sums):
                        name=mapping.get(int(ident))
                        if name:volumes[name]=volumes.get(name,0)+int(v)
                ids=[key for key,name in mapping.items() if name==selected]
                selected_rows=a[valid&np.isin(a['instrument_id'],ids)]
                if len(selected_rows):
                    raw=np.empty(len(selected_rows),dtype=RAW)
                    for name in RAW.names:raw[name]=selected_rows[name]
                    parts.append(raw);audit['selected_events']+=len(raw)
    finish(); manifest['status']='prepared'
    for name,lo,hi in [('train',START,VALID),('validation',VALID,TEST),('test',TEST,END)]:
        manifest[name+'_samples']=sum(d['samples'] for d in manifest['days'] if lo<=d['date']<hi)
    assert all(manifest[k+'_samples']>0 for k in ['train','validation','test'])
    dump(OUT/'manifest.json',manifest)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    p=argparse.ArgumentParser();p.add_argument('--limit-days',type=int);args=p.parse_args()
    prepare(args.limit_days)
