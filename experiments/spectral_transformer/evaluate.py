"""Evaluate frozen 2021 Transformer experiment; no model/threshold selection on test."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json,csv
from pathlib import Path
from sklearn.metrics import roc_auc_score
ROOT=Path('/workspace/xauusd/reports/spectral_transformer_20261013');R=ROOT/'remote';H=[15,30,60,120];rng=np.random.default_rng(20261013)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(n,x):(ROOT/n).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False))
def auc(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
def avg(x):return float(np.mean(x)) if len(x) else float('nan')
def ci(v,days,B=1500):
 if not len(v):return np.array([np.nan,np.nan])
 a,b=int(days.min()),int(days.max());s=np.bincount(days-a,weights=v,minlength=b-a+1);count=np.bincount(days-a,minlength=b-a+1);n=len(s);starts=rng.integers(0,n,(B,(n+6)//7));idx=((starts[:,:,None]+np.arange(7))%n).reshape(B,-1)[:,:n];values=s[idx].sum(1)/np.maximum(count[idx].sum(1),1);return np.quantile(values,[.025,.975])
def metrics(p,d,ids):
 y=d['y'][ids];real=y*d['scale'][ids,None]*np.sqrt(np.array(H)/60);scale=d['scale'][ids,None]*np.sqrt(np.array(H)/60);yp=p['mean'];q=p['quantiles'];probs=p['event_prob'];e=d['event'][ids];event=e>0;side=(e%2)==1;pe=1-probs[:,0];pu=probs[:,1::2].sum(1);pd=probs[:,2::2].sum(1);cond=pu/np.maximum(pu+pd,1e-8);qr=np.array([.1,.5,.9]);er=y[:,:,None]-q
 out={'rows':len(ids),'mean_mse_normalized':np.mean((yp-y)**2),'mean_pinball':np.maximum(er*qr,er*(qr-1)).mean(),'event_nll':-np.log(np.maximum(probs[np.arange(len(e)),e],1e-8)).mean(),'event_detection_auc':auc(event,pe),'event_detection_brier':np.mean((pe-event)**2),'event_rate':event.mean(),'event_9class_accuracy':np.mean(probs.argmax(1)==e),'direction_accuracy_given_event':np.mean((cond[event]>=.5)==side[event]),'direction_auc_given_event':auc(side[event],cond[event]),'horizons':{}}
 for j,h in enumerate(H):
  true=y[:,j];pred=yp[:,j];mask=true!=0;error=(pred-true)**2;zero=true**2;delta=(zero-error);dd=d['timestamp'][ids]//86400
  out['horizons'][str(h)]={'mae_dollars':np.mean(abs(pred-true)*scale[:,j]),'persistence_mae_dollars':np.mean(abs(real[:,j])),'rmse_dollars':np.sqrt(np.mean(((pred-true)*scale[:,j])**2)),'normalized_mse':error.mean(),'persistence_normalized_mse':zero.mean(),'mse_improvement_vs_persistence':delta.mean(),'mse_improvement_ci95_week':ci(delta,dd),'direction_accuracy':np.mean((pred[mask]>0)==(true[mask]>0)),'direction_auc':auc(true[mask]>0,pred[mask]),'actual_up_rate':np.mean(true[mask]>0),'q10q90_coverage':np.mean((true>=q[:,j,0])&(true<=q[:,j,2])),'mean_q10q90_width_dollars':np.mean((q[:,j,2]-q[:,j,0])*scale[:,j])}
 centres=np.array([8,23,45.5,90.5]);tp=probs[:,1:].reshape(-1,4,2).sum(2);timepred=tp@centres/np.maximum(pe,1e-8);out['timing_mae_bars_given_actual_event']=np.mean(abs(timepred[event]-d['first'][ids][event]));out['monthly']={}
 for mo in np.unique(d['month'][ids]):
  a=d['month'][ids]==mo;out['monthly'][str(mo)]={'rows':a.sum(),'mse':np.mean((yp[a]-y[a])**2),'event_nll':-np.log(np.maximum(probs[np.arange(len(e))[a],e[a]],1e-8)).mean(),'event_auc':auc(event[a],pe[a])}
 return out

def policy(ids,score,threshold,h,d):
 keep=np.flatnonzero(abs(score)>=threshold);chosen=[];end=-1
 for k in keep:
  t=int(ids[k])
  if t>end:chosen.append(k);end=t+h
 kk=np.asarray(chosen,int);rows=ids[kk];side=np.sign(score[kk]);p=(d['ohlc'][rows+h,2]-d['ohlc'][rows+1,1])*side;return rows,side,p

def main():
 assert (R/'complete.json').exists();d=dict(np.load(R/'evaluation_data.npz'));c=dict(np.load(R/'constant.npz'));test=d['test'];n=len(test);constant={k:np.broadcast_to(v,(n,)+v.shape) for k,v in c.items()};results={'constant':metrics(constant,d,test)};ps={};trades={};selections={}
 for kind in ['price','raw','engineered']:
  p=dict(np.load(R/f'{kind}_test.npz'));assert np.array_equal(p['row'],test);assert np.isfinite(p['mean']).all() and np.all(p['quantiles'][:,:,0]<=p['quantiles'][:,:,1]) and np.all(p['quantiles'][:,:,1]<=p['quantiles'][:,:,2]);assert np.allclose(p['event_prob'].sum(1),1,atol=1e-5);ps[kind]=p;results[kind]=metrics(p,d,test)
  val=dict(np.load(R/f'{kind}_selection.npz'));vid=val['row'];candidates=[]
  for j,h in enumerate(H):
   for quantile in [.5,.8,.9,.95]:
    th=np.quantile(abs(val['mean'][:,j]),quantile);rows,side,profit=policy(vid,val['mean'][:,j],th,h,d)
    if len(rows)<50:continue
    dd=d['timestamp'][rows]//86400;unique,inv=np.unique(dd,return_inverse=True);daily=np.bincount(inv,weights=profit-.25);objective=daily.mean()/max(daily.std(),1e-8);candidates.append({'horizon':h,'column':j,'threshold':float(th),'quantile':quantile,'val_trades':len(profit),'val_mean_gross':avg(profit),'val_daily_score_cost025':objective})
  best=max(candidates,key=lambda x:x['val_daily_score_cost025']);selections[kind]={'chosen':best,'all':candidates};rows,side,profit=policy(test,p['mean'][:,best['column']],best['threshold'],best['horizon'],d);dd=d['timestamp'][rows]//86400
  stats={'selected_from_development_validation':best,'test_trades':len(profit),'test_gross_mean':avg(profit),'test_gross_ci95_week':ci(profit,dd),'test_win_rate_before_cost':avg(profit>0),'cost_scenarios':{str(cost):{'mean_net':avg(profit)-cost,'total_net':np.sum(profit-cost),'win_rate':avg(profit>cost)} for cost in [0,.1,.25,.5,1]},'net025_ci95_week':ci(profit-.25,dd),'monthly_gross_mean':{str(mo):profit[d['month'][rows]==mo].mean() for mo in np.unique(d['month'][rows])},'no_future_gap_filter':True};trades[kind]=stats;np.savez_compressed(ROOT/f'{kind}_trades.npz',row=rows,side=side,gross=profit)
  # Non-overlapping evaluation and event precision at frozen validation probability threshold.
  chosen=[];end=-1
  for k,t in enumerate(test):
   if t>end:chosen.append(k);end=t+120
  kk=np.array(chosen);rr=test[kk];event=d['event'][rr]>0;pe=1-p['event_prob'][kk,0];vpe=1-val['event_prob'][:,0];threshold=np.quantile(vpe,.8);selected=pe>=threshold;sgn=d['event'][rr]%2==1;prob=p['event_prob'][kk,1::2].sum(1)-p['event_prob'][kk,2::2].sum(1);true_positive=selected&event
  results[kind]['nonoverlap_events']={'rows':len(rr),'event_auc':auc(event,pe),'probability_threshold_from_validation':threshold,'calls':selected.sum(),'event_precision':event[selected].mean() if selected.any() else np.nan,'event_recall':true_positive.sum()/max(event.sum(),1),'direction_accuracy_true_detected_events':np.mean((prob[true_positive]>0)==sgn[true_positive]) if true_positive.any() else np.nan,'joint_event_and_direction_precision':np.mean(event[selected]&((prob[selected]>0)==sgn[selected])) if selected.any() else np.nan,'baseline_event_rate':event.mean()}
 comparisons={}
 for a,b in [('price','raw'),('raw','engineered'),('price','engineered')]:
  y=d['y'][test];delta=np.mean((ps[a]['mean']-y)**2-(ps[b]['mean']-y)**2,axis=1);e=d['event'][test];ia=np.arange(len(test));eventdelta=-np.log(np.maximum(ps[a]['event_prob'][ia,e],1e-8))+np.log(np.maximum(ps[b]['event_prob'][ia,e],1e-8));comparisons[b+'_vs_'+a]={'mean_mse_improvement':delta.mean(),'mse_improvement_ci95_week':ci(delta,d['timestamp'][test]//86400),'event_nll_improvement':eventdelta.mean(),'event_nll_improvement_ci95_week':ci(eventdelta,d['timestamp'][test]//86400)}
 save('metrics.json',results);save('comparisons.json',comparisons);save('trading.json',trades);save('trading_selection.json',selections);save('evaluation_checks.json',{'passed':True,'predictions_finite':True,'quantiles_ordered':True,'event_probabilities_sum1':True,'test_rows':len(test),'no_abs_price_input':True,'2021_only':True,'final_trade_thresholds_use_only_development_selection_predictions':True,'oct_dec_previously_inspected_in_prior_research':True})
 print(json.dumps(clean({'event_auc':{k:v['event_detection_auc'] for k,v in results.items()},'mse':{k:v['mean_mse_normalized'] for k,v in results.items()},'trading':{k:v['test_gross_mean'] for k,v in trades.items()}})))
if __name__=='__main__':main()
