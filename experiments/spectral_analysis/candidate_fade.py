"""Post-hoc lead, not an independently confirmed discovery. No future-gap exclusions."""
import numpy as np
from analyze import OUT,save,trade_stats,trades
from robustness import week_ci
m=dict(np.load(OUT/'market.npz'));c=m['close'];vol=m['vol'];N=len(c);split=m['split'];sig=np.zeros(N);shock=np.zeros(N);shock[5:]=1e4*np.log(c[5:]/c[:-5])/(vol[:-5]*np.sqrt(5));last=-1;detect=0
for t in np.flatnonzero(abs(shock)>=2.5):
 if t<5 or t<=last or t+65>=N:continue
 last=t+60;detect+=1;q=t+5
 if m['timestamp'][q]-m['timestamp'][t-5]!=600:continue
 side=int(np.sign(shock[t]));anchor=c[t-5];retr=side*(c[t]-c[q]);already=(m['low'][t+1:q+1].min()<=anchor) if side>0 else (m['high'][t+1:q+1].max()>=anchor)
 if retr<=0 and not already:sig[q]=-side
stats={}
for s in range(3):
 same=np.zeros(N,bool);same[:-60]=split[:-60]==split[60:];mask=(split==s)&same&np.isfinite(m['pnl'][:,1]);ss=trade_stats(sig,mask,60,m);ids,p,days,_=trades(sig,mask,60,m);ss['gross_week_ci95']=week_ci(p,days);ss['net025_week_ci95']=week_ci(p-.25,days);ss['crosses_missing_minutes']=int((~m['valid'][ids,1]).sum());stats[['discovery','validation','test'][s]]=ss
np.savez_compressed(OUT/'candidate_fade_signals.npz',row=np.flatnonzero(sig),side=sig[sig!=0])
save('candidate_fade.json',{'status':'POST-HOC exploratory lead identified after inspecting retracement subgroup outcomes; all2021 is now discovery for this candidate; requires new-year confirmation. No claim of test significance.','rule':'5bar shock >=2.5 pre-shock60bar sigma*sqrt5; causal detector cooldown60bars. Wait5bars. No return to pre-shock anchor yet, Close extended or unchanged relative to shock endpoint. Fade at next available Open, exit after60 observed bars. One position.','no_future_gap_filter':True,'splits':stats})
print('COMPLETE')
