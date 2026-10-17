from pathlib import Path
import json,numpy as np,pandas as pd
P=Path('/workspace/xauusd/reports/intermarket_lightgbm_20261015');R=P/'remote';d=np.load(R/'evaluation_data.npz');hs=[15,30,60,120];rng=np.random.default_rng(177)
a=d['anchors'];y=d['targets'];c=d['current_close'];scale=d['scale'];tr=d['final_idx'];te=d['test_idx'];va=d['val_idx'];gt=d['gold_timestamp'];go=d['gold_ohlcv'][:,0]
pred={k:np.load(R/f'{k}_test.npz')['prediction'] for k in ['gold','all']};vp={k:np.load(R/f'{k}_validation.npz')['prediction'] for k in ['gold','all']};pred['zero']=np.zeros_like(pred['gold']);pred['mean']=np.tile(np.nanmean(y[tr],axis=0),(len(te),1))
def mean(x):return float(np.mean(x)) if len(x) else np.nan
def ci(values,ts):
 good=np.isfinite(values);v=np.asarray(values)[good];t=np.asarray(ts)[good]
 if not len(v):return [np.nan,np.nan]
 u,ix=np.unique(t//604800,return_inverse=True);s=np.bincount(ix,weights=v);n=np.bincount(ix);b=rng.integers(0,len(u),size=(1000,len(u)));means=s[b].sum(axis=1)/n[b].sum(axis=1);return np.quantile(means,[.025,.975]).tolist()
def metrics(rows,p):
 out={}
 for j,h in enumerate(hs):
  good=np.isfinite(y[rows,j]);ix=rows[good];z=y[ix,j];q=p[good,j];diff=q-z;fac=scale[ix]*np.sqrt(h);forecast=c[ix]*np.exp(q*fac);actual=c[ix]*np.exp(z*fac)
  mse=mean(diff**2);zero=mean(z**2);r={'rows':len(ix),'mse_normalized':mse,'zero_mse':zero,'improvement_vs_zero':(zero-mse)/zero,'mae_dollars':mean(np.abs(forecast-actual)),'zero_mae_dollars':mean(np.abs(c[ix]-actual)),'direction_accuracy':mean(np.sign(q[(q!=0)&(z!=0)])==np.sign(z[(q!=0)&(z!=0)])),'direction_coverage':mean(q[z!=0]!=0),'actual_up_rate':mean(z[z!=0]>0),'actual_flat_rate':mean(z==0),'prediction_std':float(np.std(q)),'target_std':float(np.std(z))}
  if np.std(q)>1e-10:r['correlation']=float(np.corrcoef(q,z)[0,1])
  else:r['correlation']=np.nan
  r['monthly']={}
  months=pd.to_datetime(a[ix],unit='s',utc=True).strftime('%Y-%m').to_numpy()
  for month in np.unique(months):
   m=months==month;r['monthly'][month]={'rows':int(m.sum()),'mse':mean(diff[m]**2),'zero_mse':mean(z[m]**2),'mae_dollars':mean(np.abs(forecast[m]-actual[m]))}
  out[str(h)]=r
 return out
M={k:metrics(te,p) for k,p in pred.items()};comparisons={}
for j,h in enumerate(hs):
 good=np.isfinite(y[te,j]);z=y[te[good],j];delta=(pred['gold'][good,j]-z)**2-(pred['all'][good,j]-z)**2
 comparisons[str(h)]={'mse_gain_all_vs_gold':mean(delta),'gain_ci95_7day_blocks':ci(delta,a[te[good]]),'relative_gain':mean(delta)/M['gold'][str(h)]['mse_normalized'],'months_all_better':sum(M['all'][str(h)]['monthly'][m]['mse']<v['mse'] for m,v in M['gold'][str(h)]['monthly'].items()),'months_total':len(M['gold'][str(h)]['monthly'])}
# Frozen validation policy, no model/policy parameters selected on final-test outcomes.
def trades(rows,p,j,threshold,cutoff=None):
 h=hs[j];out=[];blocked=-1;expired=0
 for k,row in enumerate(rows):
  t=int(a[row]);q=float(p[k,j])
  if t<blocked or abs(q)<threshold or q==0:continue
  entry=np.searchsorted(gt,t+120);exit=np.searchsorted(gt,t+(h+1)*60)
  if entry>=len(gt) or exit>=len(gt):continue
  if cutoff is not None and (gt[entry]>=cutoff or gt[exit]>=cutoff):continue
  # Real-time order expiration: if there is no entry quote before planned exit, no fill.
  if gt[entry]>=t+(h+1)*60:expired+=1;continue
  direction=1 if q>0 else -1;profit=direction*(go[exit]-go[entry]);out.append((gt[entry],gt[exit],direction,profit));blocked=gt[exit]
 return np.asarray(out,dtype=np.float64).reshape(-1,4),expired
def score(t):
 if len(t)<100:return -np.inf
 day=t[:,0].astype('int64')//86400;_,ix=np.unique(day,return_inverse=True);v=np.bincount(ix,weights=t[:,3]-.25)
 return float(v.mean()/max(v.std(),1e-9))
T={};allchoices={}
for kind in ['gold','all']:
 choices=[]
 for j,h in enumerate(hs):
  for quantile in [.5,.8,.9,.95]:
   threshold=float(np.quantile(np.abs(vp[kind][:,j]),quantile));t,expired=trades(va,vp[kind],j,threshold,cutoff=int(np.datetime64('2025-01-01','s').astype('int64')))
   choices.append({'horizon':h,'j':j,'quantile':quantile,'threshold':threshold,'validation_trades':len(t),'validation_gross_mean':mean(t[:,3]),'validation_score':score(t)})
 chosen=max(choices,key=lambda q:q['validation_score']);allchoices[kind]={'choices':choices,'chosen':chosen};tt,expired=trades(te,pred[kind],chosen['j'],chosen['threshold']);np.savez_compressed(P/f'{kind}_trades.npz',trades=tt)
 T[kind]={'policy':chosen,'test_trades':len(tt),'expired_unfilled_orders':expired,'gross_mean_dollars_per_ounce':mean(tt[:,3]),'gross_total':float(tt[:,3].sum()),'max_holding_minutes':float(np.max((tt[:,1]-tt[:,0])/60)) if len(tt) else None,'trades_over_180_minutes':int(np.sum(tt[:,1]-tt[:,0]>180*60)),'gross_winrate':mean(tt[:,3]>0),'same_entries_always_long_net025':mean(tt[:,3]*tt[:,2]-.25),'same_entries_always_short_net025':mean(-tt[:,3]*tt[:,2]-.25),'net_mean_at_cost':{str(cost):mean(tt[:,3]-cost) for cost in [0,.1,.25,.5,1]},'net025_ci95_7day':ci(tt[:,3]-.25,tt[:,0])}
if True:
 for k in ['gold','all']:
  assert len(np.load(R/f'{k}_test.npz')['row'])==len(te)
  assert np.array_equal(np.load(R/f'{k}_test.npz')['row'],te)
  assert np.isfinite(pred[k]).all()
 assert np.intersect1d(tr,te).size==0

def clean(o):
 if isinstance(o,dict):return {k:clean(v) for k,v in o.items()}
 if isinstance(o,list):return [clean(v) for v in o]
 if isinstance(o,(float,np.floating)) and not np.isfinite(o):return None
 return o
for file,obj in [('metrics.json',M),('comparisons.json',comparisons),('trading.json',T),('trading_selection.json',allchoices),('evaluation_checks.json',{'passed':True,'predictions_finite':True,'row_alignment':True,'no_train_test_overlap':True})]:
 (P/file).write_text(json.dumps(clean(obj),indent=2,allow_nan=False))
print(json.dumps(clean({'comparisons':comparisons,'trading':T}),indent=2))
