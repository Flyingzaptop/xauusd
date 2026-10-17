"""Walk-forward (expanding, yearly) ridge regression of gold forward returns, gold-only vs all-222 features.
Features standardized on the training window, NaN->0 after standardization. Reports out-of-sample corr per year."""
import json,numpy as np
Z=np.load('/home/user/data_prep/intermarket.npz',mmap_mode='r');names=np.array(Z['names']);day=Z['day'];yr=np.array([int(d[:4]) for d in day])
gold=np.array([n.startswith('XAUUSD_41:') for n in names]);sub=np.arange(len(day))[::3]  # every 3rd decision (15 min) to limit memory
X=np.asarray(Z['x'][sub]);yr=yr[sub];out={}
for h in ['fwd15','fwd30','fwd60','fwd120']:
    y=np.asarray(Z[h][sub]);out[h]={}
    for fs,cols in [('gold_only',gold),('all222',np.ones(len(names),bool))]:
        Xf=X[:,cols];res={}
        for Y in range(2018,2027):
            tr=(yr<Y)&(yr>=2017)&np.isfinite(y);te=(yr==Y)&np.isfinite(y)
            mu=np.nanmean(Xf[tr],0);sd=np.nanstd(Xf[tr],0)+1e-6
            A=np.nan_to_num((Xf[tr]-mu)/sd);B=np.nan_to_num((Xf[te]-mu)/sd);A=np.clip(A,-5,5);B=np.clip(B,-5,5)
            lam=1e4*A.shape[1]/100;W=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@(y[tr]-y[tr].mean()))
            p=B@W;res[Y]=float(np.corrcoef(p,y[te])[0,1])
        v=np.array(list(res.values()));out[h][fs]={'per_year':res,'mean':float(v.mean()),'positive_years':int((v>0).sum()),'years':len(v)}
        print(h,fs,'mean %.4f pos %d/%d'%(v.mean(),(v>0).sum(),len(v)),' '.join('%d:%+.3f'%(k%100,x) for k,x in res.items()),flush=True)
json.dump(out,open('/home/user/xauusd/reports/real_exact_meta/walkforward_ridge.json','w'),indent=1)
