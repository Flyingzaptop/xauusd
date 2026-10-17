"""Gold-only versus all-market LightGBM regression, fixed chronological protocol."""
from pathlib import Path
import json,time,gc,os,traceback,hashlib
import numpy as np,pandas as pd,lightgbm as lgb,psutil
R=Path('/content/intermarket');OUT=R/'results';OUT.mkdir(exist_ok=True);T=time.monotonic()
def status(**kw):
 kw['seconds']=round(time.monotonic()-T,1);kw['ram_GB']=round(psutil.Process().memory_info().rss/1e9,2);(R/'train_status.json').write_text(json.dumps(kw));print(json.dumps(kw),flush=True)
def unix(x):return int(np.datetime64(x,'s').astype('int64'))
try:
 meta=json.loads((R/'prepared.json').read_text());assert meta['complete']
 assert json.loads((R/'inventory.json').read_text())['passed']
 shape=tuple(meta['shape']);mm=np.memmap(R/'features.f32',dtype='float32',mode='r',shape=shape);names=meta['feature_names'];a=np.load(R/'anchors.npy');g=np.load(R/'gold.npz');gt=g['timestamp'];ohlcv=g['ohlcv'];n=meta['grid_n'];grid0=meta['grid_start'];gi=(gt-grid0)//60
 close=np.full(n,np.nan);close[gi]=ohlcv[:,3];lc=pd.Series(np.log(close));sigma=lc.diff().rolling(60,min_periods=30).std(ddof=0).to_numpy();ai=(a-grid0)//60;scale=np.maximum(sigma[ai],1e-6);past_valid=np.isfinite(scale)
 hs=[15,30,60,120];ys=np.full((len(a),4),np.nan,dtype=np.float32);dollars=np.full_like(ys,np.nan);c=close[ai]
 for j,h in enumerate(hs):
  ok=(ai+h<n)&past_valid;fut=np.full(len(a),np.nan);fut[ok]=close[ai[ok]+h];ys[:,j]=np.log(fut/c)/(scale*np.sqrt(h));dollars[:,j]=fut-c
 cuts={'train_end':unix('2024-01-01'),'validation_end':unix('2025-01-01')};purge=120*60
 train_idx=np.where(past_valid&(a<cuts['train_end']-purge))[0];val_idx=np.where(past_valid&(a>=cuts['train_end'])&(a<cuts['validation_end']-purge))[0];final_idx=np.where(past_valid&(a<cuts['validation_end']-purge))[0];test_idx=np.where(past_valid&(a>=cuts['validation_end']))[0]
 assert a[train_idx].max()+purge<cuts['train_end'];assert a[val_idx].max()+purge<cuts['validation_end']
 np.savez_compressed(OUT/'evaluation_data.npz',anchors=a,targets=ys,dollars=dollars,scale=scale,current_close=c,train_idx=train_idx,val_idx=val_idx,final_idx=final_idx,test_idx=test_idx,gold_timestamp=gt,gold_ohlcv=ohlcv)
 common={'objective':'regression','metric':'l2','learning_rate':.04,'max_bin':63,'bin_construct_sample_cnt':100000,'feature_pre_filter':False,'min_data_in_leaf':300,'lambda_l2':10.,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':1,'num_threads':8,'verbosity':-1,'seed':177,'feature_fraction_seed':177,'bagging_seed':177,'data_random_seed':177,'force_col_wise':True}
 configs=[{'num_leaves':31,'max_depth':6},{'num_leaves':63,'max_depth':8}]
 (OUT/'protocol.json').write_text(json.dumps({'complete_before_test':True,'train_rows':len(train_idx),'validation_rows':len(val_idx),'final_train_rows':len(final_idx),'test_rows':len(test_idx),'features':shape[0],'horizons_minutes':hs,'cuts':cuts,'purge_seconds':purge,'params':common,'configs':configs,'source_sha256':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['features.py','prepare.py','train.py']}},indent=2))
 def materialize(indices,cols):
  if len(cols)==shape[0]:x=mm[:,indices].T
  else:x=np.array(mm[cols,:][:,indices].T,order='C',dtype=np.float32)
  assert x.dtype==np.float32 and x.shape==(len(indices),len(cols));return x
 def dataset(indices,cols,label_index,ref=None):
  status(phase='dataset',rows=len(indices),features=len(cols),reference=ref is not None)
  x=materialize(indices,cols);v=ys[indices,label_index];w=np.isfinite(v).astype(np.float32)
  data=lgb.Dataset(x,label=np.nan_to_num(v,nan=0,posinf=0,neginf=0),weight=w,reference=ref,params=common,free_raw_data=True,feature_name=[names[k] for k in cols]);data.construct();del x;gc.collect();return data
 def relabel(data,indices,j):
  v=ys[indices,j];data.set_label(np.nan_to_num(v,nan=0,posinf=0,neginf=0));data.set_weight(np.isfinite(v).astype(np.float32))
 choices={};selection={};test_predictions={};val_predictions={}
 for kind in ['gold','all']:
  cols=np.array([i for i,k in enumerate(names) if kind=='all' or k.startswith('XAUUSD__') or k.startswith('clock_')]);status(phase='model_begin',kind=kind,features=len(cols))
  tr=dataset(train_idx,cols,2);va=dataset(val_idx,cols,2,tr);candidates=[]
  for ci,cfg in enumerate(configs):
   status(phase='select_config',kind=kind,config=ci)
   curve={};tm=time.monotonic();model=lgb.train({**common,**cfg},tr,num_boost_round=300,valid_sets=[va],valid_names=['validation'],callbacks=[lgb.early_stopping(25,verbose=False,min_delta=1e-4),lgb.record_evaluation(curve)])
   score=float(model.best_score['validation']['l2']);candidates.append({'config':cfg,'best_iteration':model.best_iteration,'mse':score,'seconds':time.monotonic()-tm,'curve':curve});model.save_model(str(OUT/f'{kind}_config{ci}.txt'))
   (OUT/f'{kind}_config_selection.json').write_text(json.dumps(candidates,indent=2));status(phase='config_complete',kind=kind,config=ci,iteration=model.best_iteration,score=score,seconds_model=time.monotonic()-tm)
  selected=min(candidates,key=lambda z:z['mse']);cfg=selected['config'];choices[kind]={'config':cfg,'epochs':{}};vpred=np.zeros((len(val_idx),4),np.float32);selections=[]
  for j,h in enumerate(hs):
   relabel(tr,train_idx,j);relabel(va,val_idx,j);curve={};status(phase='select_horizon',kind=kind,horizon=h)
   model=lgb.train({**common,**cfg},tr,num_boost_round=300,valid_sets=[va],valid_names=['validation'],callbacks=[lgb.early_stopping(25,verbose=False,min_delta=1e-4),lgb.record_evaluation(curve)])
   choices[kind]['epochs'][str(h)]=int(model.best_iteration);vpred[:,j]=model.predict(materialize(val_idx,cols),num_threads=8).astype(np.float32)
   model.save_model(str(OUT/f'{kind}_{h}_selection.txt'));selections.append({'horizon':h,'iteration':model.best_iteration,'score':float(model.best_score['validation']['l2']),'curve':curve})
  np.savez_compressed(OUT/f'{kind}_validation.npz',row=val_idx,prediction=vpred);(OUT/f'{kind}_selection.json').write_text(json.dumps(selections,indent=2));val_predictions[kind]=vpred
  del tr,va,model;gc.collect()
  # Freeze settings BEFORE any final-test prediction. No test-driven changes.
  (OUT/'selection_frozen.json').write_text(json.dumps(choices,indent=2))
  final=dataset(final_idx,cols,0);tpred=np.zeros((len(test_idx),4),np.float32);importances=[]
  for j,h in enumerate(hs):
   relabel(final,final_idx,j);rounds=choices[kind]['epochs'][str(h)];status(phase='final_fit',kind=kind,horizon=h,rounds=rounds)
   model=lgb.train({**common,**cfg},final,num_boost_round=rounds);model.save_model(str(OUT/f'{kind}_{h}_final.txt'))
   # Prediction in bounded chunks keeps test matrices out of peak training memory.
   for lo in range(0,len(test_idx),20000):
    ix=test_idx[lo:lo+20000];tpred[lo:lo+len(ix),j]=model.predict(materialize(ix,cols),num_threads=8)
   gain=model.feature_importance('gain');top=np.argsort(gain)[::-1][:100];importances.append({'horizon':h,'features':[{'name':names[cols[k]],'gain':float(gain[k])} for k in top if gain[k]>0]})
  np.savez_compressed(OUT/f'{kind}_test.npz',row=test_idx,prediction=tpred);(OUT/f'{kind}_importance.json').write_text(json.dumps(importances,indent=2));test_predictions[kind]=tpred
  del final,model;gc.collect();status(phase='model_complete',kind=kind)
 (OUT/'complete.json').write_text(json.dumps({'complete':True,'seconds':time.monotonic()-T,'choices':choices},indent=2));status(phase='complete')
except Exception:
 (OUT/'error.txt').write_text(traceback.format_exc());status(phase='failed',error=traceback.format_exc());raise
