"""Independent exhaustive cache timing checks before GPU training."""
import hashlib
import databento as db
import numpy as np
import pandas as pd
from config import *

def source_spot_check(manifest,ts,lp):
    """Re-read DBN independently and compare one entire cached raw window exactly."""
    day=next(d for d in manifest['days'] if d['samples'])
    meta=np.load(CACHE/(day['date']+'.npz'));j=int(meta['ix'][0]);cutoff=int(ts[j])*10**9
    start=int(pd.Timestamp(day['date'],tz='UTC').timestamp())*10**9
    filename=next((ROOT/'data/MBO').glob('glbx-mdp3-'+day['date'][:7].replace('-','')+'*.mbo.dbn.zst'))
    store=db.DBNStore.from_file(filename);date=pd.Timestamp(day['date']).date()
    ids=[int(v['symbol']) for v in store.metadata.mappings[day['contract']] if v['start_date']<=date<v['end_date']]
    kept=[]
    for batch in store.to_ndarray(count=500_000):
        use=(batch['ts_recv']>=max(start,cutoff-MAX_AGE*10**9))&(batch['ts_recv']<cutoff)
        use &= np.isin(batch['instrument_id'],ids)&((batch['flags']&40)==0)&(batch['rtype']==160)
        use &= (batch['price']>0)&(batch['price']<np.iinfo(np.int64).max)
        rows=batch[use]
        if len(rows):kept.append(rows.copy())
        if int(batch['ts_recv'][-1])>=cutoff:break
    source=np.concatenate(kept);expected=source[-EVENTS:]
    cached=np.load(day['file'],mmap_mode='r')[0];observed=cached[cached['ts_recv']>0]
    assert len(observed)==len(expected)
    for name in RAW.names:assert np.array_equal(observed[name],expected[name]),name
    trades=source[source['action']==b'T']
    series=pd.Series(np.log(trades['price'].astype(float)/1e9),index=pd.to_datetime(trades['ts_recv'],unit='ns',utc=True))
    bars=series.resample('1min',closed='left',label='left').ohlc().dropna()
    last=bars.iloc[-1]['close'];control=np.zeros(26)
    for n,k in enumerate(range(5,0,-1)):
        time=pd.Timestamp(int(ts[j])-k*60,unit='s',tz='UTC')
        if time in bars.index:
            control[4*n:4*n+4]=(bars.loc[time,['high','open','close','low']].to_numpy()-last)/json.loads((OUT/'scale.json').read_text())['S0']
            control[20+n]=1
    control[25]=(last-lp[j-1,2])/json.loads((OUT/'scale.json').read_text())['S0']
    assert np.allclose(control,meta['gc'][0],atol=1e-5,rtol=1e-5)
    return dict(date=day['date'],target_index=j,exact_raw_records=len(expected),gc_control_independently_resampled=True)

def verify():
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['status']=='prepared'
    z=np.load(OUT/'candles.npz');ts=z['ts'];eligible=set(z['eligible'].tolist());all_ix=[];counts=[]
    for day in manifest['days']:
        if not day['samples']:continue
        m=np.load(CACHE/(day['date']+'.npz'));ix=m['ix'];assert len(ix)==day['samples']
        assert all(int(i) in eligible for i in ix)
        assert np.isfinite(m['gc']).all()
        assert day['selection_volumes'] and day['contract']==max(day['selection_volumes'],key=lambda k:(day['selection_volumes'][k],k))
        a=np.load(day['file'],mmap_mode='r');assert a.shape==(len(ix),EVENTS)
        for b in range(0,len(ix),32):
            r=a[b:b+32];jj=ix[b:b+32];cutoff=ts[jj].astype(np.uint64)*10**9
            valid=r['ts_recv']>0
            assert (valid.sum(1)>=128).all()
            assert np.all(r['ts_recv'] < cutoff[:,None])
            assert np.all((~valid)|(r['ts_recv']>=cutoff[:,None]-MAX_AGE*10**9))
            assert np.all(r['ts_recv'][:,1:]>=r['ts_recv'][:,:-1])
            assert np.all((r['flags']&40)==0)
            assert all(np.all(np.diff(ts[j-CONTEXT-max(LAGS):j+1])==60) for j in jj)
        all_ix.extend(ix.tolist());counts.extend(m['counts'].tolist());del a
    assert len(set(all_ix))==len(all_ix) and np.all(np.diff(all_ix)>0)
    split_counts={name:sum(d['samples'] for d in manifest['days'] if lo<=d['date']<hi)
                  for name,lo,hi in [('train',START,VALID),('validation',VALID,TEST),('test',TEST,END)]}
    assert all(split_counts.values())
    spot=source_spot_check(manifest,ts,z['lp'])
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    dump(OUT/'verification.json',dict(status='passed',samples=len(all_ix),split_counts=split_counts,
        event_count_quantiles=np.quantile(counts,[0,.25,.5,.75,1]).tolist(),
        source_code_sha256=sources,independent_source_check=spot,checks=['strict MBO cutoff for every event','age limit','padding and ordering',
        'snapshot exclusion','contract selection from previous day','M1 continuous past support','unique chronological coverage']))
    print(json.dumps(split_counts))

if __name__=='__main__':verify()
