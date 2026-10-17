"""Signed mid move after large-player events (latency 300 ms), $/oz, events de-duplicated (first per type/side per 10 s).
usage: event_returns.py EVENTS.npz GRIDDIR [OUT.json]"""
import sys,json,glob,numpy as np
E=np.load(sys.argv[1])['ev'];gd=sys.argv[2]
Z=[np.load(f) for f in sorted(glob.glob(gd+'/chunk_*.npz'))];gts=np.concatenate([z['ts'] for z in Z]);px=np.concatenate([z['px'] for z in Z]);del Z
mid=(px[:,0]+px[:,1])/2;sp=px[:,1]-px[:,0];G=100_000_000
def m_at(t):
    i=np.clip(np.searchsorted(gts,t,side='right')-1,0,len(gts)-1);ok=(t-gts[i]<=G)&(sp[i]>0)&(sp[i]<10);return np.where(ok,mid[i],np.nan)
L=300_000_000;H=[1,10,60,300];names={1:'ICEBERG',2:'BIG_ADD',3:'BIG_PULL',4:'BIG_AGGR'};res={}
hour=(E[:,0]//10**9)%86400/3600;E=E[~((hour>=22)&(hour<23))]
import os
if os.environ.get('MAXTS'):E=E[E[:,0]<int(os.environ['MAXTS'])]
for ty in (1,2,3,4):
    for cond,lab in [(lambda e:np.ones(len(e),bool),'all'),(lambda e:e[:,3]>=10,'size>=10'),(lambda e:e[:,4]<=0,'at best')]:
        e=E[E[:,1]==ty];e=e[cond(e)]
        if len(e)<50:continue
        # dedupe: first event per side per 10 s bucket
        key=e[:,2]*10**12+(e[:,0]//10**10);_,first=np.unique(key,return_index=True);e=e[np.sort(first)]
        t0=e[:,0]+L;m0=m_at(t0);row={'n':int(len(e))}
        for h in H:
            m1=m_at(t0+h*10**9);r=e[:,2]*(m1-m0);ok=np.isfinite(r);r=r[ok]
            # cluster by minute for t-stat
            mins=(e[ok,0]//(60*10**9));u,inv=np.unique(mins,return_inverse=True);cl=np.bincount(inv,r)
            t=cl.mean()/(cl.std()/np.sqrt(len(cl)))*np.sqrt(1) if len(cl)>10 else 0
            row[f'h{h}s']={'mean_usd':float(r.mean()),'t_clustered':float(r.sum()/np.sqrt((cl**2).sum())) ,'hit':float((r>0).mean())}
        res[f'{names[ty]}|{lab}']=row
        print(f"{names[ty]:9} {lab:9} n={len(e):6d} "+' '.join(f"{h:>3}s {row[f'h{h}s']['mean_usd']:+.3f}$ (t {row[f'h{h}s']['t_clustered']:+.1f})" for h in H),flush=True)
if len(sys.argv)>3:json.dump(res,open(sys.argv[3],'w'),indent=1)
