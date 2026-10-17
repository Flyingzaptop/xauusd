"""Walk-forward gold-only ridge, 15-min horizon on a 15-min decision grid (trades never overlap).
Threshold = quantile q of |prediction| on the training window. Net per year after $0.25 roundtrip
(and sensitivity $0.10/$0.40). Per-trade bp and $/oz."""
import json,numpy as np
Z=np.load('/home/user/data_prep/intermarket.npz',mmap_mode='r');names=np.array(Z['names']);day=Z['day']
sub=np.arange(len(day))[::3];yr=np.array([int(d[:4]) for d in day[sub]])
gold=np.array([n.startswith('XAUUSD_41:') for n in names]);X=np.asarray(Z['x'][sub][:,gold]);y=np.asarray(Z['fwd15'][sub])
ent=np.asarray(Z['entry'][sub]);ok=np.asarray(Z['entry_ok'][sub])&np.isfinite(y);dd=day[sub];out={}
for Y in range(2018,2027):
    tr=(yr<Y)&(yr>=2017)&ok;te=(yr==Y)&ok
    mu=np.nanmean(X[tr],0);sd=np.nanstd(X[tr],0)+1e-6;f=lambda m:np.clip(np.nan_to_num((X[m]-mu)/sd),-5,5)
    A=f(tr);W=np.linalg.solve(A.T@A+1e4*A.shape[1]/100*np.eye(A.shape[1]),A.T@(y[tr]-y[tr].mean()));ptr=A@W;p=f(te)@W
    r={}
    for q in [.9,.95,.99]:
        thr=np.quantile(np.abs(ptr),q);sel=np.abs(p)>thr;g=np.sign(p[sel])*y[te][sel];e=ent[te][sel]
        nd=len(np.unique(dd[te]))
        r[q]={'trades':int(sel.sum()),'gross_bp_per_trade':float(g.mean()),
              **{f'net_usd_per_day_cost{c}':float(((g*e/1e4)-c).sum()/nd) for c in [.1,.25,.4]},
              'hit':float((g>0).mean())}
    out[Y]=r;print(Y,' | '.join('q%.2f n=%d gross %.2fbp hit %.3f net$/d %+.2f'%(q,v['trades'],v['gross_bp_per_trade'],v['hit'],v['net_usd_per_day_cost0.25']) for q,v in r.items()),flush=True)
json.dump(out,open('/home/user/xauusd/reports/real_exact_meta/walkforward_trade15.json','w'),indent=1)
