"""Information ceiling with intermarket features: LightGBM regression of gold forward return (bp) per horizon,
gold-only vs all-222 (same hyperparameters). Threshold policy (LONG if pred>thr+..., SHORT if pred<-thr)
executed sequentially per session, one position, non-overlapping, cost $0.25 roundtrip in bp.
Fit 2017-2020, threshold choice 2021-2022, report 2023 (selection period). Test period untouched."""
import sys,json,numpy as np
import lightgbm as lgb
Z=np.load(sys.argv[1] if len(sys.argv)>1 else '/home/user/data_prep/intermarket.npz')
X=Z['x'];names=Z['names'];day=Z['day'];slot=Z['slot'];ent=Z['entry'];eok=Z['entry_ok']
clock=np.c_[np.sin(2*np.pi*slot/120),np.cos(2*np.pi*slot/120)]
gold=np.array([n.startswith('XAUUSD_41:') for n in names])
cost=.25/np.where(eok,ent,np.nan)*1e4
TR=(day>='2017-01-01')&(day<='2020-12-31');VA=(day>='2021-01-01')&(day<='2022-12-31');SE=(day>='2023-01-01')&(day<='2023-12-31')
def trade(pred,y,mask,h,thr):
    steps=h//5;tot=0.;n=0;ds=np.unique(day[mask]);
    for d in ds:
        ii=np.where(mask&(day==d))[0];i=0
        while i<len(ii):
            k=ii[i];p=pred[k]
            if eok[k] and np.isfinite(y[k]) and abs(p)>thr:tot+=np.sign(p)*y[k]-cost[k];n+=1;i+=steps
            else:i+=1
    return float(tot/len(ds)),float(n/len(ds))
out={}
for h in [15,30,60,120]:
    y=Z[f'fwd{h}'];out[h]={}
    for fs,cols in [('gold_only',gold),('all222',np.ones(len(names),bool))]:
        Xf=np.c_[X[:,cols],clock];tr=TR&np.isfinite(y)&eok;sub=np.where(tr)[0][::2]  # every 2nd point to limit time
        m=lgb.LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=31,min_child_samples=1000,subsample=.7,subsample_freq=1,colsample_bytree=.3 if fs=='all222' else .8,reg_lambda=10,verbose=-1,n_jobs=4)
        m.fit(Xf[sub],y[sub],eval_set=[(Xf[VA&np.isfinite(y)],y[VA&np.isfinite(y)])],callbacks=[lgb.early_stopping(40,verbose=False)])
        p=m.predict(Xf);r={'best_iter':int(m.best_iteration_ or 400)}
        for nm,mk in [('val',VA),('sel',SE)]:
            ok=mk&np.isfinite(y)&eok;r[f'{nm}_corr']=float(np.corrcoef(p[ok],y[ok])[0,1]);r[f'{nm}_dir_acc']=float((np.sign(p[ok])==np.sign(y[ok])).mean())
        grid=np.quantile(np.abs(p[VA]),[.5,.8,.9,.95,.98,.99,.995]);vv={float(t):trade(p,y,VA,h,t) for t in grid};thr=max(vv,key=lambda t:vv[t][0])
        r.update(val_by_thr=vv,thr=thr,val=vv[thr],sel=trade(p,y,SE,h,thr),sel_always_long=trade(np.ones_like(p),y,SE,h,0))
        imp=m.booster_.feature_importance('gain');top=np.argsort(imp)[::-1][:10];nm_=list(names[cols])+['clock_s','clock_c'];r['top_gain']=[nm_[i] for i in top]
        out[h][fs]=r;print(h,fs,json.dumps({k:v for k,v in r.items() if k!='val_by_thr'}),flush=True)
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else '/home/user/xauusd/reports/real_exact_meta/ceiling_intermarket.json','w'),indent=1)
