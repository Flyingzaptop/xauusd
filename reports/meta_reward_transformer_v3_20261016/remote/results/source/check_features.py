import numpy as np,pyarrow as pa,sys,json
from features import encode,NAMES
rng=np.random.default_rng(177);n=5000;start=1609459200;t=start+np.arange(n)*60;close=100*np.exp(np.cumsum(rng.normal(0,.0002,n)));open_=np.r_[close[0],close[:-1]];high=np.maximum(close,open_)*1.0001;low=np.minimum(close,open_)*.9999;mask=np.ones(n,bool);mask[1200:1500]=False
D={'timestamp':pa.array(t[mask]*1000,type=pa.timestamp('ms',tz='UTC')),'open':open_[mask],'high':high[mask],'low':low[mask],'close':close[mask],'volume':rng.integers(1,100,n)[mask]};tab=pa.table(D);a=t[np.arange(600,3000,5)]
x,_=encode(tab,start,n,a)
D2=dict(D)
for key in ['open','high','low','close']:
 v=np.asarray(D[key]).copy();v[t[mask]>t[3000]]*=2;D2[key]=v
x2,_=encode(pa.table(D2),start,n,a);np.testing.assert_allclose(x,x2,equal_nan=True)
prefix=tab.filter(pa.compute.less_equal(tab['timestamp'],pa.scalar(int(t[3000]*1000),type=pa.timestamp('ms',tz='UTC'))))
x3,_=encode(prefix,start,3001,a);np.testing.assert_allclose(x,x3,equal_nan=True)
age_index=NAMES.index('log_quote_age');assert x[age_index,np.where(a==t[1400])[0][0]]>0
assert x.shape==(48,len(a));print(json.dumps({'passed':True,'features':48,'future_mutation':True,'prefix_recompute':True,'gap_age_explicit':True}))
