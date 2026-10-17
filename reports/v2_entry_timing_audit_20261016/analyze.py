from pathlib import Path
import json,time
import numpy as np
P=Path(__file__).parent
R=Path('/workspace/xauusd/reports/meta_reward_transformer_v2_foreground_20261015/remote/results')
a=np.load(R/'learned_teacher_test.npz');rows=a['rows'];actions=a['actions'];days=a['day'];original=a['trades']
cached=P/'test_execution_vectors.npz'
if cached.exists():
 z=np.load(cached);entryts=z['entryts'];exitts=z['exitts'];pnl=z['pnl'];assert np.array_equal(z['rows'],rows)
else:
 ts=np.load('/workspace/scratch/v3_anchors.npy');prices=np.load('/workspace/scratch/v3_gold_ohlcv.npy',mmap_mode='r')
 planned=np.minimum(ts[rows]+7200,(ts[rows]//86400+1)*86400-60);ex=np.searchsorted(ts,planned)
 pnl=(prices[ex,0]-prices[rows,0]).astype('float32');entryts=ts[rows];exitts=ts[ex]
 np.savez_compressed(cached,entryts=entryts,exitts=exitts,pnl=pnl,actions=actions,days=days,rows=rows)
nextrow=np.searchsorted(entryts,exitts);di=np.searchsorted(days,entryts//86400)

def execute(act,records=False):
 candidates=np.flatnonzero(act);k=0;daily=np.zeros(len(days));count=0;tr=[]
 while k<len(candidates):
  j=int(candidates[k]);sgn=1 if act[j]==1 else -1;v=sgn*float(pnl[j])-.25;daily[di[j]]+=v;count+=1
  if records:tr.append([entryts[j],exitts[j],sgn,v])
  k=int(np.searchsorted(candidates,max(j+1,nextrow[j]),side='left'))
 return daily,count,np.array(tr)
real,n,tr=execute(actions,True);assert tr.shape==original.shape and np.allclose(tr,original,atol=1e-7,rtol=0),'Exact V2 reproduction failed'
longmask=(actions==1).astype('int8');rl,nl,_=execute(longmask);assert np.allclose(rl,real) and nl==n
always,na,_=execute(np.ones(len(rows),dtype='int8'))
groups=[np.flatnonzero(di==j) for j in range(len(days))];rng=np.random.default_rng(20261016);N=2000;out={};arrays={}
for mode in ['constant_LONG_probability','global_shuffle','within_day_shuffle','within_day_circular_shift']:
 daily=np.empty((N,len(days)));counts=np.empty(N,dtype=int);t=time.time()
 for i in range(N):
  if mode=='constant_LONG_probability':act=(rng.random(len(rows))<longmask.mean()).astype('int8')
  elif mode=='global_shuffle':act=rng.permutation(longmask)
  else:
   act=np.empty_like(longmask)
   for g in groups:
    if len(g):act[g]=rng.permutation(longmask[g]) if mode=='within_day_shuffle' else np.roll(longmask[g],int(rng.integers(len(g))))
  daily[i],counts[i],_=execute(act)
 totals=daily.sum(1);delta=real-daily.mean(0);weeks=delta.reshape(12,7).sum(1);boots=weeks[rng.integers(0,12,(20000,12))].sum(1)
 out[mode]={'simulations':N,'mean_profit':float(totals.mean()),'median_profit':float(np.median(totals)),'profit_95pct_range':np.quantile(totals,[.025,.975]).tolist(),'mean_trades':float(counts.mean()),'trade_95pct_range':np.quantile(counts,[.025,.975]).tolist(),'fraction_random_at_least_actual':float((1+np.sum(totals>=real.sum()))/(N+1)),'actual_minus_control_mean':float(real.sum()-totals.mean()),'weekly_block_bootstrap_delta_95pct':np.quantile(boots,[.025,.975]).tolist(),'weekly_actual_minus_control':weeks.tolist(),'runtime_seconds':time.time()-t}
 arrays[mode+'_daily']=daily;arrays[mode+'_trades']=counts
 print(mode,json.dumps(out[mode]),flush=True)
np.savez_compressed(P/'randomization_runs.npz',actual_daily=real,always_long_daily=always,**arrays)
result={'exact_original_execution_reproduced':True,'period_start_UTC':str(np.datetime64(int(days[0]),'D')),'calendar_days':len(days),'cost_per_trade':.25,'actual':{'profit':float(real.sum()),'trades':n,'profit_per_trade':float(real.sum()/n),'LONG_signal_fraction':float(longmask.mean()),'weekly_profit':real.reshape(12,7).sum(1).tolist()},'always_long':{'profit':float(always.sum()),'trades':na},'controls':out,'limitations':['One already observed 84-day test and one trained student seed; exploratory posthoc analysis.','Randomizing sampled actions tests timing of realized policy, not conditional probabilities; no full per-minute policy probabilities were stored.','Fixed 120-minute/UTC-boundary exit, single position and fixed $0.25 cost; no variable spread/slippage.','Weekly blocks reflect serial dependence only approximately; four correlated controls are diagnostics, not independent discoveries.']}
(P/'results.json').write_text(json.dumps(result,indent=2))
