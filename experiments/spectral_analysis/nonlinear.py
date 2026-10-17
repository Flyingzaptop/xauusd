import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,json,time,joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from analyze import OUT,H,clean,save,corr,auc,day_ci,trade_stats,trades

def main():
 start=time.time();meta=json.loads((OUT/'features.json').read_text());m=dict(np.load(OUT/'market.npz'));b=np.load(OUT/'base.npy');sp=np.load(OUT/'spectral.npy');split=m['split'];n=len(b);days=m['timestamp']//86400
 idx=[i for i,s in enumerate(meta['spectral']) if '.band' in s and ('.energy.' in s or '.entropy.' in s)];changes=[]
 for lag in [5,30]:
  a=np.zeros((n,len(idx)),np.float32);a[lag:]=sp[lag:,idx]-sp[:-lag,idx];changes.append(a)
 allx=np.column_stack([b,sp,*changes]);del sp,changes
 masks=[]
 for s in range(3):
  same=np.zeros(n,bool);same[:-120]=split[:-120]==split[120:];mask=(split==s)&m['valid'][:,2]&same&(np.arange(n)>=30);masks.append(mask)
 target=np.column_stack([np.sign(m['y'][:,1]),np.log(np.maximum(m['rv'][:,1],1e-10))]);reports={};pred={};strategies={}
 for label,x in [('price_only',b),('price_plus_spectral_dynamics',allx)]:
  p=np.empty((n,2));rr={'features':x.shape[1],'targets':{}}
  for j,kind in enumerate(['direction','log_rv']):
   model=HistGradientBoostingRegressor(max_iter=300,max_leaf_nodes=7,learning_rate=.05,l2_regularization=10,min_samples_leaf=500,early_stopping=True,n_iter_no_change=20,tol=1e-5,random_state=20261012)
   model.fit(x[masks[0]],target[masks[0],j],X_val=x[masks[1]],y_val=target[masks[1],j]);# sklearn early stop can stop AFTER best; restore exact validation best via warm-start refit fixed iterations
   best=int(np.argmax(model.validation_score_));best=max(1,best)
   if best!=model.n_iter_:
    model=HistGradientBoostingRegressor(max_iter=best,max_leaf_nodes=7,learning_rate=.05,l2_regularization=10,min_samples_leaf=500,early_stopping=False,random_state=20261012).fit(x[masks[0]],target[masks[0],j])
   p[:,j]=model.predict(x);joblib.dump(model,OUT/f'hgb_{label}_{kind}.joblib',compress=3)
   stats={'iterations':model.n_iter_,'metrics':{}}
   for s,splitname in enumerate(['discovery','validation','test']):
    ok=masks[s];yy=target[ok,j];pp=p[ok,j];stats['metrics'][splitname]={'n':ok.sum(),'mse':np.mean((yy-pp)**2),'corr':corr(yy,pp)}
    if j==0:stats['metrics'][splitname].update({'auc':auc(yy,pp),'accuracy':np.mean((yy>0)==(pp>0)),'up_fraction':np.mean(yy>0)})
   rr['targets'][kind]=stats;print(json.dumps({'model':label,'target':kind,'iterations':model.n_iter_,'seconds':round(time.time()-start,1)}),flush=True)
  reports[label]=rr;pred[label]=p
  candidates=[]
  for q in [.5,.8,.9,.95]:
   th=np.quantile(abs(p[masks[1],0]),q);sig=np.where(abs(p[:,0])>=th,np.sign(p[:,0]),0);ids,profits,dd,_=trades(sig,masks[1],60,m);_,iv=np.unique(dd,return_inverse=True);daily=np.bincount(iv,weights=profits-.25);score=daily.mean()/max(daily.std(),1e-8)*np.sqrt(252);candidates.append((score,q,th,sig))
  score,q,th,sig=max(candidates,key=lambda z:z[0]);strategies[label]={'selection':{'horizon':60,'quantile':q,'threshold':th,'validation_daily_score_cost025':score},'validation':trade_stats(sig,masks[1],60,m),'test':trade_stats(sig,masks[2],60,m)}
  ids,profits,dd,_=trades(sig,masks[2],60,m);np.savez_compressed(OUT/f'trades_hgb_{label}.npz',row=ids,timestamp=m['timestamp'][ids],side=sig[ids],gross=profits)
 save('nonlinear_models.json',reports);save('nonlinear_trading.json',strategies);np.savez_compressed(OUT/'nonlinear_predictions.npz',**pred)
 comparisons={};ok=masks[2]
 for j,kind in enumerate(['direction','log_rv']):
  delta=(pred['price_only'][ok,j]-target[ok,j])**2-(pred['price_plus_spectral_dynamics'][ok,j]-target[ok,j])**2;comparisons[kind]={'mse_improvement':delta.mean(),'day_cluster_ci95':day_ci(delta,days[ok]),'monthly_improvement':{str(mo):delta[m['months'][ok]==mo].mean() for mo in [10,11,12]}}
 save('nonlinear_paired_comparison.json',comparisons);print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
