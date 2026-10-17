import json,numpy as np,torch
from objective import trade_metrics,centered_teacher_reward,entropy_coefficient,robust_excess
b=torch.tensor([.8,.2]);scores=b.expand(6,-1);a=torch.tensor([0,1,2,1,2,0]);assert torch.equal(centered_teacher_reward(scores,b,a),torch.zeros(6))
scores=scores.clone();scores[1,0]+=.1;scores[2,1]+=.1;r=centered_teacher_reward(scores,b,a);assert r[1]>0 and r[2]>0
schedule=np.array([entropy_coefficient(i) for i in range(2000)]);assert np.all(np.diff(schedule)<=1e-12) and abs(schedule[-1]-.03)<1e-10
controls=[{'long':{'utility_per_day':2},'short':{'utility_per_day':-3}},{'long':{'utility_per_day':-3},'short':{'utility_per_day':2}}]
long=[{'utility_per_day':2},{'utility_per_day':-3}];conditional=[{'utility_per_day':1},{'utility_per_day':1}]
assert robust_excess(conditional,controls)[0]>robust_excess(long,controls)[0]
assert robust_excess([{'utility_per_day':10},{'utility_per_day':-3}],controls)[0]==robust_excess(long,controls)[0]
d=np.arange(14);one=trade_metrics([(0,60,1,30)],d);many=trade_metrics([(i*60,i*60+60,1,1) for i in range(30)],d);assert many['utility_per_day']>one['utility_per_day']
print(json.dumps({'passed':True,'checks':['constant directional teacher preference yields zero expected reward','both context-specific LONG and SHORT improvements can be rewarded','entropy coefficient decreases smoothly without feedback','better bull-only result cannot hide unchanged bear failure','profit concentration penalized']}))
