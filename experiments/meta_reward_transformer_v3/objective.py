"""Explicit net-profit/concentration objective; no bonus for making trades."""
import numpy as np

def trade_metrics(records,days):
 a=np.asarray(records,dtype=float).reshape(-1,4);days=np.asarray(days,dtype=np.int64)
 daily=np.zeros(len(days));p=a[:,3] if len(a) else np.array([])
 if len(a):
  ix=np.searchsorted(days,a[:,0].astype('int64')//86400)
  np.add.at(daily,ix,p)
 # Unit coefficient is a declared risk preference, not inferred from the target.
 # Equal total positive profit across more trades has lower concentration cost.
 concentration=float(np.linalg.norm(p));net=float(p.sum());n=max(len(days),1)
 equity=np.r_[0,np.cumsum(daily)];dd=np.maximum.accumulate(equity)-equity
 positive=np.maximum(p,0);effective=float(positive.sum()**2/(positive@positive)) if (positive@positive)>0 else 0.
 weekly=[float(daily[j:j+7].sum()) for j in range(0,len(days),7)]
 return {'utility_per_day':(net-concentration)/n,'net_total':net,'mean_net_dollars_per_day':net/n,'concentration_cost':concentration,'trades':len(p),'days':len(days),'trades_per_day':len(p)/n,'effective_profitable_trades':effective,'max_daily_drawdown':float(dd.max()),'profitable_day_fraction':float((daily>0).mean()) if len(days) else 0.,'weekly_net':weekly,'daily_net':daily.tolist(),'daily_dates':days.tolist()}

def contiguous_ids(ts,ids,start,calendar_days):
 lo=int(start);hi=lo+int(calendar_days)*86400
 return ids[(ts[ids]>=lo)&(ts[ids]<hi)]

def centered_teacher_reward(scores,baseline,actions):
 import torch
 index=(actions-1).clamp(min=0)
 chosen=scores.gather(1,index[:,None]).squeeze(-1)
 base=baseline[index]
 return torch.where(actions==0,torch.zeros_like(chosen),2*(chosen-base))

def entropy_coefficient(update,start=.10,end=.03,steps=1500):
 import math
 progress=min(max(update,0)/steps,1.)
 return end+(start-end)*.5*(1+math.cos(math.pi*progress))

def robust_excess(metrics,controls):
 # The nonlinear worst-window operator prevents a single rising window from
 # hiding failures in a falling window. Controls are fixed for each window.
 values=[m['utility_per_day']-max(0.,c['long']['utility_per_day'],c['short']['utility_per_day']) for m,c in zip(metrics,controls)]
 return float(min(values)),values
