"""Information ceiling: supervised regressors of money on student features (ridge, LightGBM),
threshold policy executed with the same blocking/fixed-exit rules. Thresholds chosen on outer-pool
train days; reported on selection 2023 (no test use)."""
import sys,json,numpy as np,torch
import lightgbm as lgb
Z=np.load(sys.argv[1] if len(sys.argv)>1 else '/home/user/data_prep/xau_sessions.npz')
X=Z['x'];M=Z['money'];K=Z['mask'];DAY=Z['day'];N=X.shape[1];H=Z['future'].shape[2]
sp=lambda lo,hi:np.where((DAY>=lo)&(DAY<=hi))[0]
TR=sp('2017-01-01','2020-12-31');VA=sp('2021-01-01','2022-12-31');SEL=sp('2023-01-01','2023-12-31')
def flat(idx,last_only=False):
    x=X[idx][:,:,-1] if last_only else X[idx].reshape(len(idx),N,-1);return x.reshape(len(idx)*N,-1)
def run(pl,ps,idx,thr):
    # deterministic sequential execution: one position, fixed exit H steps later
    tot=0.;n=0
    for d,(a,b) in enumerate(zip(pl,ps)):
        i=0;m=M[idx[d]];k=K[idx[d]]
        while i<N:
            if k[i] and max(a[i],b[i])>thr:
                s=1 if a[i]>=b[i] else 2;tot+=m[i,s];n+=1;i=min(i+H,N)
            else:i+=1
    return float(tot/len(idx)),float(n/len(idx))
out={}
for name in ['ridge','lgbm']:
    ytr=M[TR][...,1:].reshape(-1,2);ktr=K[TR].reshape(-1);xtr=flat(TR)[ktr];ytr=ytr[ktr]
    if name=='ridge':
        A=np.c_[xtr,np.ones(len(xtr))];W=np.linalg.solve(A.T@A+1e3*np.eye(A.shape[1]),A.T@ytr)
        pred=lambda idx:(np.c_[flat(idx),np.ones(len(idx)*N)]@W).reshape(len(idx),N,2)
    else:
        xl=flat(TR,True)[ktr];ms=[lgb.LGBMRegressor(n_estimators=300,learning_rate=.03,num_leaves=31,min_child_samples=500,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1).fit(xl,ytr[:,j]) for j in range(2)]
        pred=lambda idx:np.stack([m_.predict(flat(idx,True)) for m_ in ms],-1).reshape(len(idx),N,2)
    pva,psel=pred(VA),pred(SEL)
    yv=M[VA][...,1:][K[VA]];pv=pva[K[VA]];corr=[float(np.corrcoef(pv[:,j],yv[:,j])[0,1]) for j in range(2)]
    grid=np.quantile(pva.max(-1),[.5,.8,.9,.95,.98,.99,.995,.999])
    va={float(t):run(pva[...,0],pva[...,1],VA,t) for t in grid};best=max(va,key=lambda t:va[t][0])
    out[name]={'val_corr_long_short':corr,'val_by_threshold':va,'chosen_thr':best,'val_best':va[best],'selection_2023':run(psel[...,0],psel[...,1],SEL,best),
               'selection_corr':[float(np.corrcoef(psel[K[SEL]][:,j],M[SEL][...,1:][K[SEL]][:,j])[0,1]) for j in range(2)]}
    print(name,json.dumps(out[name]),flush=True)
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else '/home/user/xauusd/reports/real_exact_meta/ceiling.json','w'),indent=1)
