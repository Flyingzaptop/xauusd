"""Pass 1 over a monthly MBO file: per (UTC date, instrument_id) message counts by action and traded volume.
Also writes outright GC instrument_id map. Output: /home/user/mbo_out/scan_<tag>.json"""
import sys,json,re,time,collections,numpy as np,databento as db
f=sys.argv[1];tag=sys.argv[2]
st=db.DBNStore.from_file(f);m=st.metadata
outr={}
for sym,rows in m.mappings.items():
    if re.fullmatch(r'GC[FGHJKMNQUVZ]\d',sym):
        for r in rows:outr.setdefault(int(r['symbol']),sym)
print('outright ids',outr,flush=True)
cnt=collections.defaultdict(lambda:np.zeros(8,np.int64));t=time.time();n=0
for a in st.to_ndarray(count=5_000_000):
    n+=len(a);day=(a['ts_event']//86_400_000_000_000).astype(np.int64)
    for iid in outr:
        s=a['instrument_id']==iid
        if not s.any():continue
        x=a[s];d=day[s]
        for dd in np.unique(d):
            y=x[d==dd];k=(int(dd),iid);c=cnt[k]
            for j,ch in enumerate([b'A',b'C',b'M',b'T',b'F',b'R']):c[j]+=(y['action']==ch).sum()
            c[6]+=y['size'][y['action']==b'T'].sum();c[7]+=len(y)
    print(n,'%.0fs'%(time.time()-t),flush=True)
out={'outright':{str(k):v for k,v in outr.items()},'rows':n,'counts':{f'{d}|{i}':c.tolist() for (d,i),c in cnt.items()}}
json.dump(out,open(f'/home/user/mbo_out/scan_{tag}.json','w'));print('done',n)
