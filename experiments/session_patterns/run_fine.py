"""Exploratory extension (declared after the main scan): 30-min UTC slots (drift, slot->next slot), and weekday x session drift.
Same walk-forward, cost and candidate criterion as run.py."""
import json,sys,numpy as np
sys.argv=[sys.argv[0]]
import importlib.util
spec=importlib.util.spec_from_file_location('base','/home/user/xauusd/experiments/session_patterns/run.py')
src=open('/home/user/xauusd/experiments/session_patterns/run.py').read().split('R=[]')[0]   # reuse helpers only
g={};exec(src,g)
win,wf,utc_h,days,SES,M=g['win'],g['wf'],g['utc_h'],g['days'],g['SES'],g['M']
R=[];slots=[]
for k in range(46):           # 00:00 .. 22:30 UTC start
    s=utc_h(k*0.5);e=utc_h(k*0.5+0.5);r,a=win(s,e);slots.append((r,a))
    R.append(wf(f'slot {k//2:02d}:{30*(k%2):02d}-{(k+1)//2:02d}:{30*((k+1)%2):02d} drift','drift',r,a))
for k in range(45):
    R.append(wf(f'slot {k//2:02d}:{30*(k%2):02d} -> next slot','pred',slots[k+1][0],slots[k+1][1],slots[k][0]))
dow=days.dayofweek.values
for n,s,e in SES:
    r,a=win(s,e)
    for wd,nm in enumerate(['Mon','Tue','Wed','Thu','Fri']):
        rr=np.where(dow==wd,r,np.nan);R.append(wf(f'{nm} {n} drift','drift',rr,a))
cand=[r for r in R if r['candidate']];Rs=sorted(R,key=lambda r:-r['t_net'])
json.dump({'n_tests':len(R),'candidates':cand,'all':Rs},open('/home/user/xauusd/reports/session_patterns/results_fine.json','w'),indent=1)
print('tests',len(R),'candidates',len(cand))
for r in Rs[:12]:print(f"{r['test'][:44]:44} net {r['net_bp_per_trade']:+6.2f} gross {r['gross_bp_per_trade']:+6.2f} t {r['t_net']:+5.2f} +yrs {r['positive_years']}/{r['years']} days {r['oos_days']}")
