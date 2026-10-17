from pathlib import Path
import numpy as np,pandas as pd,json,time
P=Path('/workspace/xauusd/reports/m1_entry_evaluator_20261015')
d=pd.read_parquet(P/'analysis_inputs.parquet');d['day']=d.timestamp.dt.floor('D')
all_days=d.groupby('day').agg(total_points=('timestamp','size'),complete_future=('full_120_intraday','sum'))
good=d.full_120_intraday & d.past_120_complete & np.isfinite(d.past_scale_dollars)
a=d.loc[good].copy();daily=all_days.join(a.groupby('day').size().rename('eligible_points')).fillna({'eligible_points':0});daily['eligible_points']=daily.eligible_points.astype(int)
count=a.groupby('day').size();eligible_days=count.index
# A descriptive grid, never a reward or a claim of predictive entry selection.
rows=[];grid_daily=[]
def longest_false(x):
 best=cur=0
 for yes in x:
  cur=0 if yes else cur+1;best=max(best,cur)
 return best
for min_mean in [.25,.5,1.]:
 for max_risk in [.25,.5,1.]:
  masks=[]
  for side in ['long','short']:
   masks.append((a[side+'_mean_pnl']/a.past_scale_dollars>=min_mean)&(a[side+'_max_adverse']/a.past_scale_dollars<=max_risk))
  hits=a.loc[masks[0]|masks[1],['timestamp','day']]
  per=hits.groupby('day').size().reindex(eligible_days,fill_value=0)
  # Greedy non-overlapping 120-minute opportunities; ex-post existence, not trades.
  times=pd.DatetimeIndex(hits.timestamp).as_unit('s').asi8;selected=[];until=-1
  for k,t in enumerate(times):
   if t>=until:selected.append(k);until=t+7200
  non=hits.iloc[selected].groupby('day').size().reindex(eligible_days,fill_value=0)
  assert ((per>0)==(non>0)).all(), 'Every eligible day with a hit must have a nonoverlap window'
  assert len(selected)<2 or np.all(np.diff(times[selected])>=7200)
  for period,sel in [('all',np.ones(len(per),bool)),('2016_2024',per.index.year<=2024),('2025_2026',per.index.year>=2025)]:
   v=per[sel];w=non[sel]
   rows.append({'period':period,'minimum_mean_pnl_s':min_mean,'maximum_adverse_s':max_risk,'days':len(v),'days_with_opportunity':int((v>0).sum()),'day_coverage':float((v>0).mean()),'median_points_per_day':float(v.median()),'median_nonoverlap_windows_per_day':float(w.median()),'nonoverlap_windows_q10':float(w.quantile(.1)),'longest_no_opportunity_trading_days':longest_false((v>0).to_numpy())})
  grid_daily.append(pd.DataFrame({'day':eligible_days,'minimum_mean_pnl_s':min_mean,'maximum_adverse_s':max_risk,'qualifying_points':per.values,'nonoverlap_windows':non.values}))
pd.DataFrame(rows).to_csv(P/'opportunity_sensitivity.csv',index=False)
pd.concat(grid_daily,ignore_index=True).to_csv(P/'daily_opportunity_grid.csv',index=False)
daily.to_csv(P/'daily_coverage.csv')
# Exact 3-objective Pareto representatives over complete history, each direction.
# Min adverse, max mean PNL, max profitable-close fraction. No scalar weights.
# Fraction of 120 closes has only 121 possible values.
front=[]
for side in ['long','short']:
 risk=(a[side+'_max_adverse']/a.past_scale_dollars).to_numpy(dtype=float)
 gain=(a[side+'_mean_pnl']/a.past_scale_dollars).to_numpy(dtype=float)
 fraction=np.rint(a[side+'_profitable_close_fraction'].to_numpy()*120).astype(int)
 order=np.lexsort((-fraction,-gain,risk));best=np.full(121,-np.inf);kept=[]
 for i in order:
  f=fraction[i];g=gain[i]
  if best[f]>=g:continue
  kept.append(i);best[:f+1]=np.maximum(best[:f+1],g)
 # Earlier records of equal risk may be dominated by later higher gain/fraction?
 # Lexsort puts greater gain then fraction first, so no later point dominates.
 q=a.iloc[kept][['timestamp','day','past_scale_dollars']].copy();q['side']=side
 q['mean_pnl_s']=gain[kept];q['max_adverse_s']=risk[kept];q['profitable_close_fraction']=fraction[kept]/120
 q['mean_pnl_dollars']=a.iloc[kept][side+'_mean_pnl'].to_numpy();front.append(q)
 print('Pareto',side,len(q),flush=True)
front=pd.concat(front,ignore_index=True);front.to_csv(P/'pareto_representatives.csv',index=False)
# An independent, deterministic subsample checks Pareto selection invariants.
for side in ['long','short']:
 f=front[front.side==side];v=f[['mean_pnl_s','max_adverse_s','profitable_close_fraction']].to_numpy()
 if len(v)<5000:
  for i in range(len(v)):
   dominates=(v[:,0]>=v[i,0])&(v[:,1]<=v[i,1])&(v[:,2]>=v[i,2])&((v[:,0]>v[i,0])|(v[:,1]<v[i,1])|(v[:,2]>v[i,2]))
   assert not dominates.any()
summary={'rows':len(d),'eligible_full_windows':len(a),'observed_UTC_days':len(daily),'eligible_UTC_days':len(eligible_days),'days_without_eligible_full_window':int((daily.eligible_points==0).sum()),'global_pareto_representatives':len(front),'days_with_global_pareto':int(front.day.nunique()),'grid':rows,'diagnostic_only':True,'neural_model_trained':False,'gpu_used':False,'costs_included':False}
(P/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='grid'}),flush=True)
