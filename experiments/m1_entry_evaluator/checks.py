import numpy as np,json
from core import metadata,block
n=600;t=1704067200+np.arange(n)*60
c=100+np.arange(n)*.1;o=np.c_[c,c+.08,c-.02,c+.05]
l,p,s=metadata(t,o[:,3]);b=block(t,o,l,p,s,150,151)
assert b.long_pnl_120m.iloc[0]>11.9 and b.short_pnl_120m.iloc[0]<-11.9
assert abs(b.long_max_adverse.iloc[0]-.02)<1e-6
for h in [5,15,30,60,120]:assert abs(b[f'long_pnl_{h}m'].iloc[0]+b[f'short_pnl_{h}m'].iloc[0])<1e-6
# Translation invariance and scale equivariance.
for factor,shift in [(1,5000),(3,0)]:
 v=o*factor+shift;ll,pp,ss=metadata(t,v[:,3]);bb=block(t,v,ll,pp,ss,150,151)
 assert np.allclose(bb.long_mean_pnl/bb.past_scale_dollars,b.long_mean_pnl/b.past_scale_dollars,rtol=1e-5)
# Future changes do not alter the past normalizer.
v=o.copy();v[150:]*=2;assert metadata(t,v[:,3])[2][150]==s[150]
# Calendar gaps truncate, never stitched into 120 minutes.
tgap=t.copy();tgap[180:]+=3600;lg,_,_=metadata(tgap,o[:,3]);assert lg[150]==30
# Midnight truncation, not an overnight holding assumption.
tday=1704067200+np.arange(1440)*60;cd=np.arange(1440,dtype=float)+100
assert metadata(tday,cd)[0][1430]==10
# At the last source bar only a one-minute outcome is available.
assert l[-1]==1
# No adverse excursion should be invented on wholly favourable OHLC.
up=o.copy();up[150:,2]=up[150:,0];bu=block(t,up,l,p,s,150,151);assert bu.long_max_adverse.iloc[0]==0
print(json.dumps({'passed':True,'checks':['known path','side symmetry','translation invariance','scale equivariance','past-only normalization','gap truncation','UTC-day truncation','dataset tail','OHLC adverse excursion']}))
