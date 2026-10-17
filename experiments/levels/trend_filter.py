"""Exploratory: trend-following away from session VWAP after k-sigma deviation, only when the distance to VWAP at entry d >= D dollars
(ex-ante, known at entry). Target = d further, stop = VWAP (risk d). Conservative same-bar rule, cost $0.25."""
import json,numpy as np,pandas as pd
src=open('/home/user/xauusd/experiments/levels/mirror.py').read().split('\nRES=[]\n')[0];g={};exec(src,g)
run_trend,summarize,C,H,L,V,sess,year,N,TMAX=[g[k] for k in ['run_trend','summarize','C','H','L','V','sess','year','N','TMAX']]
vwap=(pd.Series(C*V).groupby(sess).cumsum()/pd.Series(V).groupby(sess).cumsum().replace(0,np.nan)).values
dev=C-vwap;sd=pd.Series(dev).rolling(120,min_periods=60).std().values;RES=[]
for kk in (2,3):
    for D in (0,1,2,3,5):
        up=(dev>=kk*sd)&(sd>0)&(dev>=D);dn=(dev<=-kk*sd)&(sd>0)&(-dev>=D)
        out=np.zeros((N,5));n=run_trend(up,dn,np.nan_to_num(vwap),H,L,C,sess,TMAX,1.0,out);r=summarize(f'TREND VWAP {kk}sd, d>=${D}',out[:n],0.5);RES.append(r)
        print(f"{r['test']:24} n={r['n']:6d} P {r['p_target']:.3f} gross ${r['mean_gross']:+.3f} net ${r['mean_net']:+.3f} t {r['t_net']:+.2f} +yrs {r['pos_years']}/{r['years']}  per-year:",{k:round(v) for k,v in r['per_year'].items()})
json.dump(RES,open('/home/user/xauusd/reports/levels/trend_filter.json','w'),indent=1)
