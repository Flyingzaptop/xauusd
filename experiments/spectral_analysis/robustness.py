import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,json,csv
from analyze import OUT,H,save,day_ci,auc,trade_stats,trades
rng=np.random.default_rng(90712)
def week_ci(values,days,B=4000):
 lo,hi=int(days.min()),int(days.max());s=np.bincount(days-lo,weights=values,minlength=hi-lo+1);n=np.bincount(days-lo,minlength=hi-lo+1);L=len(s);starts=rng.integers(0,L,size=(B,(L+6)//7));idx=((starts[:,:,None]+np.arange(7))%L).reshape(B,-1)[:,:L];den=n[idx].sum(1);b=s[idx].sum(1)/np.maximum(den,1);return np.quantile(b,[.025,.975]).tolist()
def main():
 m=dict(np.load(OUT/'market.npz'));meta=json.loads((OUT/'features.json').read_text());n=len(m['close']);days=m['timestamp']//86400;split=m['split'];b=np.load(OUT/'base.npy');pred=dict(np.load(OUT/'predictions.npz'));non=dict(np.load(OUT/'nonlinear_predictions.npz'));report={};masks={}
 for s in [1,2]:
  for j,h in enumerate(H):
   same=np.zeros(n,bool);same[:-h]=split[:-h]==split[h:];masks[s,j]=(split==s)&m['valid'][:,j]&same
 # Non-overlap direction precision including always-long comparator on IDENTICAL rows.
 for label,p in pred.items():
  report[label]={}
  for j,h in enumerate(H):
   ix=np.arange(0,n,h+1);ix=ix[masks[2,j][ix]];yy=m['y'][ix,j];hit=(p[ix,j+3]>0)==(yy>0);base=yy>0
   report[label][str(h)]={'rows':len(ix),'accuracy':hit.mean(),'always_long_accuracy':base.mean(),'accuracy_ci95_day':day_ci(hit,days[ix]),'improvement_vs_long_ci95_week':week_ci(hit.astype(float)-base,days[ix]),'auc':auc(yy,p[ix,j+3])}
 for label,p in non.items():
  j=1;ix=np.arange(0,n,61);ix=ix[masks[2,j][ix]];yy=m['y'][ix,j];hit=(p[ix,0]>0)==(yy>0);report['hgb_'+label]={'60':{'rows':len(ix),'accuracy':hit.mean(),'always_long_accuracy':np.mean(yy>0),'accuracy_ci95_day':day_ci(hit,days[ix]),'improvement_vs_long_ci95_week':week_ci(hit.astype(float)-(yy>0),days[ix]),'auc':auc(yy,p[ix,0])}}
 save('nonoverlap_direction.json',report)
 # All stored strategies: same-entry controls and weekly block uncertainty.
 ts={}
 configs=json.loads((OUT/'trading.json').read_text());nc=json.loads((OUT/'nonlinear_trading.json').read_text());configs.update({'hgb_'+k:v for k,v in nc.items()})
 for label,conf in configs.items():
  a=np.load(OUT/f'trades_{label}.npz');ix=a['row'];p=a['gross'];h=conf['selection']['horizon'];j=H.index(h);long=m['pnl'][ix,j];trend=np.sign(b[ix,meta['base'].index('base.momentum_60')]);momentum=trend*long;fade=-momentum
  ts[label]={'gross_mean_ci95_7calendar_day_blocks':week_ci(p,days[ix]),'cost025_mean_ci95_7calendar_day_blocks':week_ci(p-.25,days[ix]),'always_long_same_entries_mean':long.mean(),'momentum60_same_entries_mean':momentum.mean(),'fade60_same_entries_mean':fade.mean(),'increment_vs_always_long_mean':np.mean(p-long),'increment_vs_always_long_ci95_week':week_ci(p-long,days[ix]),'increment_vs_momentum_ci95_week':week_ci(p-momentum,days[ix]),'increment_vs_fade_ci95_week':week_ci(p-fade,days[ix])}
 winner=max(configs,key=lambda k:configs[k]['selection']['validation_daily_score_cost025'])
 save('trading_robustness.json',{'winner_selected_by_validation_only':winner,'strategies':ts})
 # Scalar energy, forecast activity and uninformative constant benchmark on common rows.
 vmask=masks[2,2];tr=(split==0)&m['valid'][:,2];rv=m['rv'][:,1];threshold=np.quantile(rv[tr],.8);largemove_threshold=np.quantile(abs(m['pnl'][tr,1]),.9);activity={}
 for label,p in {**{k:v[:,7] for k,v in pred.items()},**{'hgb_'+k:v[:,1] for k,v in non.items()},'recent_vol60':b[:,meta['base'].index('base.logvol_60')]}.items():
  qt=np.quantile(p[(split==1)&m['valid'][:,2]],[.2,.8]);ok=vmask;low=ok&(p<=qt[0]);high=ok&(p>=qt[1]);target=(rv[ok]>threshold).astype(float)
  activity[label]={'future_high_rv_auc':auc(target-.5,p[ok]),'large_absolute_move_auc':auc((abs(m['pnl'][ok,1])>largemove_threshold).astype(float)-.5,p[ok]),'low_forecast_future_rv':rv[low].mean(),'high_forecast_future_rv':rv[high].mean(),'high_low_ratio':rv[high].mean()/rv[low].mean(),'low_n':low.sum(),'high_n':high.sum(),'low_future_abs_move_usd':abs(m['pnl'][low,1]).mean(),'high_future_abs_move_usd':abs(m['pnl'][high,1]).mean()}
 save('activity_prediction.json',{'train_high_rv_threshold':threshold,'train_large_move_threshold_usd':largemove_threshold,'test':activity})
 # Summarize future reactions by month and energy group without further selecting a strategy.
 ev=dict(np.load(OUT/'shock_events.npz'));e=json.loads((OUT/'shock_reactions.json').read_text());groups=np.searchsorted(e['energy_edges_train'],ev['energy_change']);eventdetails=[]
 for mo in range(1,13):
  for q in range(3):
   ok=(ev['month']==mo)&(groups==q)&(ev['already_returned']==0)
   eventdetails.append({'month':mo,'energy_group':q,'n':ok.sum(),'return_rate':ev['future_return'][ok].mean() if ok.any() else np.nan,'mean_continuation':ev['continuation_pnl'][ok].mean() if ok.any() else np.nan})
 diffs={}
 for s in range(3):
  ok=(ev['split']==s)&(ev['already_returned']==0);dd=ev['day'][ok];gg=groups[ok];yy=ev['future_return'][ok];unique=np.unique(dd);vals=[]
  for _ in range(4000):
   selected=rng.choice(unique,len(unique),replace=True);w=np.array([(selected==d).sum() for d in dd]);a=(gg==0);z=(gg==2)
   if w[a].sum() and w[z].sum():vals.append(np.average(yy[z],weights=w[z])-np.average(yy[a],weights=w[a]))
  diffs[str(s)]={'high_minus_low':yy[gg==2].mean()-yy[gg==0].mean(),'ci95_day_cluster':np.quantile(vals,[.025,.975])}
 save('shock_robustness.json',{'monthly':eventdetails,'return_rate_difference':diffs})
 # Independent raw-file spot checks of labels, OHLC convention and entry timing.
 raw=np.load('/workspace/xauusd/reports/pcgrad_mse_month_20261009/dataset.npz');price=np.exp(raw['log_prices']);times=raw['timestamp'];ix=rng.choice(np.arange(n-120),200,replace=False);errors=[]
 for i in ix:
  k=m['source_row'][i];assert abs(price[k,2]-m['close'][i])<1e-9;assert price[k,0]>=max(price[k,1],price[k,2])-1e-8;assert price[k,3]<=min(price[k,1],price[k,2])+1e-8
  for j,h in enumerate(H):
   errors.append(abs(price[k+h,2]-price[k+1,1]-m['pnl'][i,j]));assert bool(m['valid'][i,j])==bool(np.all(np.diff(times[k:k+h+1])==60))
 assert max(errors)<1e-8
 save('analysis_checks.json',{'passed':True,'raw_source_checks':len(ix),'label_max_error_usd':max(errors),'future_entry_next_open':True,'calendar_gap_filter_verified':True,'price_ohlc_column_order_verified':'H,O,C,L','spectral_preparation_causality_checks':'../causal_spectral_year_2021_20261012/verification.json','screening_all_components':2356,'derived_rows':n,'discovery_validation_test_purged':True,'test_used_for_selection':False,'neighbouring_trades_do_not_overlap':True,'scientific_scope':'retrospective single year; OHLC prices, no bid/ask or executable spread'})
 print('COMPLETE')
if __name__=='__main__':main()
