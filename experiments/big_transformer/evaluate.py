"""Evaluate downloaded test predictions: per-year corr (15/30/60/120), 15-min non-overlapping threshold trading
(threshold = 95th pct of |val pred|; 1-min latency already in targets; cost $0.25/oz), all222 vs gold paired by
decision time, side-permutation p-value."""
import json,sys,glob,numpy as np
from pathlib import Path
R=Path(sys.argv[1]);T=np.load(sys.argv[2] if len(sys.argv)>2 else '/home/user/data_prep/panel/targets.npz')
HZ=[15,30,60,120];Y={h:T[f'y{h}'] for h in HZ};ent=T['entry'];gns=T['grid_ns']
year=lambda i:(gns[i].astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970)
minute=(gns//60_000_000_000)%1440
def trade(idx,pred,thr,seed=0):
    sel=((minute[idx]-479)%15==0)&(np.abs(pred)>thr);i=idx[sel];s=np.sign(pred[sel]);r=Y[15][i]
    pnl=s*r*ent[i]/1e4-.25;g=np.random.default_rng(seed);perm=np.array([np.mean(g.permutation(s)*r) for _ in range(2000)])
    days=len(np.unique(gns[idx]//86_400_000_000_000))
    return {'trades':int(sel.sum()),'gross_bp':float(np.mean(s*r)) if sel.any() else 0.,'net_usd_day':float(pnl.sum()/days),'long_share':float((s>0).mean()) if sel.any() else 0.,
            'perm_p':float((perm>=np.mean(s*r)).mean()) if sel.any() else 1.}
out={}
for d in sorted(R.glob('fold*_*')):
    if not (d/'preds.npz').exists():continue
    z=np.load(d/'preds.npz');ti=z['test_idx'];tp=z['test_pred'];vp=z['val_pred'];r={}
    for Yr in np.unique(year(ti)):
        m=year(ti)==Yr;row={f'corr{h}':float(np.corrcoef(tp[m,j],Y[h][ti[m]])[0,1]) for j,h in enumerate(HZ)}
        row.update(trade(ti[m],tp[m,0],np.quantile(np.abs(vp[:,0]),.95)));r[int(Yr)]=row
    out[d.name]=r
    for Yr,row in r.items():print(d.name,Yr,' '.join(f'{k}={v:+.4f}' if isinstance(v,float) else f'{k}={v}' for k,v in row.items()))
json.dump(out,open(R/'evaluation.json','w'),indent=1)
