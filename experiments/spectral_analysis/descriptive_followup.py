"""Descriptive follow-up of observed mechanisms; no further strategy selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json
from analyze import OUT,H,save,day_ci,trade_stats,trades
from robustness import week_ci
m=dict(np.load(OUT/'market.npz'));meta=json.loads((OUT/'features.json').read_text());b=np.load(OUT/'base.npy');sp=np.load(OUT/'spectral.npy',mmap_mode='r');split=m['split'];n=len(split);days=m['timestamp']//86400
valid=m['valid'][:,1].copy();valid[:-60]&=split[:-60]==split[60:];valid[-60:]=False
# Clock schedule selected from discovery, without future direction fitting.
hours=(m['timestamp']//3600)%24;hourtable=[]
for s in range(3):
 for h in range(24):
  ok=valid&(split==s)&(hours==h)
  if not ok.any():continue
  hourtable.append({'split':s,'hour_feed':h,'n':ok.sum(),'future_rv_bps':m['rv'][ok,1].mean(),'mean_abs_move_usd':abs(m['pnl'][ok,1]).mean(),'up_rate':np.mean(m['pnl'][ok,1]>0)})
trainhours=sorted([x for x in hourtable if x['split']==0 and x['n']>=1000],key=lambda r:r['future_rv_bps']);sets={'quiet_discovery_hours':[x['hour_feed'] for x in trainhours[:3]],'active_discovery_hours':[x['hour_feed'] for x in trainhours[-3:]]};clock={}
for key,hs in sets.items():
 ok=valid&(split==2)&np.isin(hours,hs);clock[key]={'hours':hs,'test_n':ok.sum(),'test_future_rv_bps':m['rv'][ok,1].mean(),'test_abs_move_usd':abs(m['pnl'][ok,1]).mean(),'up_rate':np.mean(m['pnl'][ok,1]>0)}
save('clock_activity.json',{'timezone':'source feed clock, unverified timezone; do not label as London/NY/UTC sessions','all_hours':hourtable,'discovery_selected_hour_groups':clock})
# Absolute future activity vs RELATIVE activity: do not confuse falling ratio with smaller raw moves.
activity={}
for name in ['base.logvol_60','wavelet_returns.energy.band16_32','wavelet_returns.energy.band64_128']:
 v=b[:,meta['base'].index(name)] if name.startswith('base') else sp[:,meta['spectral'].index(name)];threshold=np.quantile(v[valid&(split==0)],[.2,.8]);groups=np.where(v<=threshold[0],0,np.where(v>=threshold[1],1,-1));rows=[]
 for s in range(3):
  for g in range(2):
   ok=valid&(split==s)&(groups==g);rows.append({'split':s,'group':['low','high'][g],'n':ok.sum(),'past_rv_bps_mean':m['vol'][ok].mean(),'future_rv_bps_mean':m['rv'][ok,1].mean(),'future_past_rv_ratio_median':np.median(m['rv'][ok,1]/m['vol'][ok]),'absolute_normalized_future_move_mean':abs(m['y'][ok,1]).mean()})
 activity[name]=rows
save('volatility_decay.json',activity)
# Fixed natural retracement categories illustrate distance-to-target confounding.
ev=dict(np.load(OUT/'shock_events.npz'));groups=np.searchsorted([0,.25,.5],ev['retracement']);rows=[]
for s in range(3):
 for g in range(4):
  ok=(ev['split']==s)&(ev['already_returned']==0)&(groups==g);yy=ev['future_return'][ok];p=-ev['continuation_pnl'][ok]
  rows.append({'split':s,'retracement_category':['none_or_further_extension','0_to_25pct','25_to_50pct','over_50pct'][g],'n':ok.sum(),'anchor_return_rate':yy.mean() if ok.any() else np.nan,'mean_fade_60bar_pnl_before_cost':p.mean() if ok.any() else np.nan,'pnl_day_ci95':day_ci(p,ev['day'][ok])})
save('retracement_geometry.json',{'status':'post-result descriptive decomposition; no new heldout trading claim or optimized threshold','rows':rows})
# Fixed strategies without ex-post future-gap exclusion. Actual next available Open and observed-bar exits.
configs=json.loads((OUT/'trading.json').read_text());configs.update({'hgb_'+k:v for k,v in json.loads((OUT/'nonlinear_trading.json').read_text()).items()});pr=dict(np.load(OUT/'predictions.npz'));pn=dict(np.load(OUT/'nonlinear_predictions.npz'));out={}
for label,cfg in configs.items():
 sel=cfg['selection'];h=sel['horizon'];j=H.index(h)
 if label.startswith('hgb_'):score=pn[label[4:]][:,0]
 else:score=pr[label][:,j+(3 if sel['kind']=='sign' else 0)]
 sig=np.where(abs(score)>=sel['threshold'],np.sign(score),0);same=np.zeros(n,bool);same[:-h]=split[:-h]==split[h:];mask=(split==2)&same&np.isfinite(m['pnl'][:,j]);stats=trade_stats(sig,mask,h,m);ids,p,dd,_=trades(sig,mask,h,m);stats.update({'positions_crossing_missing_minutes':int((~m['valid'][ids,j]).sum()),'net025_week_ci95':week_ci(p-.25,dd)});out[label]=stats
save('gap_inclusive_trading.json',{'protocol':'same validation-selected models/thresholds, NO future-gap row filter, next available observed Open; h observed bars until exit. May include overnight/weekend carry; constant cost scenarios omit carry and actual bid/ask. Diagnostic not live execution backtest.','test':out})
print('COMPLETE')
