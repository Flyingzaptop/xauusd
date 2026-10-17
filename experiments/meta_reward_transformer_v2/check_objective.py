import numpy as np,json
from objective import trade_metrics,contiguous_ids
D=np.arange(14)
def score(p):return trade_metrics([(int(i%14)*86400,int(i%14)*86400+60,1,x) for i,x in enumerate(p)],D)
a=score([10.,10.,10.]);b=score([30/146]*146);z=score([]);loss=score([-1]*20)
assert b['utility_per_day']>a['utility_per_day']>z['utility_per_day']>loss['utility_per_day']
assert abs(a['net_total']-b['net_total'])<1e-10
assert score([0]*100)['utility_per_day']==0
assert score([1,-1]*50)['utility_per_day']<0
assert b['effective_profitable_trades']>145.9
ix=contiguous_ids(np.arange(30)*86400,np.arange(30),7*86400,14);assert np.array_equal(ix,np.arange(7,21))
print(json.dumps({'passed':True,'same_profit_3_trades':a['utility_per_day'],'same_profit_146_trades':b['utility_per_day'],'wait':0,'loss_20_trades':loss['utility_per_day'],'note':'Synthetic reward audit, not market performance'}))
