"""Markouts of simulated passive fills against the 100 ms grid. usage: markout.py PROBES.npz GRIDDIR [FIRST_DAY_FILTER]"""
import sys,glob,numpy as np
P=np.load(sys.argv[1]);gd=sys.argv[2];TICK=0.1;G=100_000_000
Z=[np.load(f) for f in sorted(glob.glob(gd+'/chunk_*.npz'))];gts=np.concatenate([z['ts'] for z in Z]);gpx=np.concatenate([z['px'] for z in Z])
R=P['res'];base=int(P['base']);fill=R[R[:,7]==1];allp=R[R[:,7]>=0]
print('probes (non-empty book)',len(allp),'fills',len(fill),'fill rate %.3f'%(len(fill)/len(allp)))
for sd,nm in [(1,'BUY (join bid)'),(-1,'SELL (join ask)')]:
    a=allp[allp[:,3]==sd];f=fill[fill[:,3]==sd]
    print(f'{nm}: probes {len(a)} fills {len(f)} fill rate {len(f)/max(len(a),1):.3f}  median queue ahead at arrival {np.median(a[:,5]) if len(a) else 0:.0f} lots; median time-to-fill {np.median(f[:,2]-f[:,1])/1e9:.2f}s')
def at(t):
    i=np.searchsorted(gts,t,side='right')-1;return np.clip(i,0,len(gts)-1)
for h in [1,5,30,60]:
    row=[]
    for sd in (1,-1):
        f=fill[fill[:,3]==sd];t=f[:,2]+int(h*1e9);i=at(t);i0=at(f[:,2])
        px=base/1e9+f[:,4]*TICK;bid=gpx[i,0];ask=gpx[i,1];mid=(bid+ask)/2
        ok=(gts[i]>=t-G)&np.isfinite(mid)
        mk=sd*(mid-px)/TICK;tk=(bid-px)/TICK if sd>0 else (px-ask)/TICK
        row.append((sd,ok.sum(),mk[ok].mean(),tk[ok].mean(),np.median(mk[ok])))
    print(f'h={h:>3}s '+' | '.join(f'{"BUY" if s>0 else "SELL"} n={n} markout(mid) {m:+.2f} ticks (median {md:+.1f}); exit-as-taker {t_:+.2f}' for s,n,m,t_,md in row))
