"""Independently enumerate thresholds, including tied scores and abstention."""
import numpy as np
from policy import fit_policy,route,wilson,fixtures
print(fixtures())
rng=np.random.default_rng(103)
for _ in range(100):
    p=np.round(rng.dirichlet([1,8,1],size=40),3)
    y=rng.choice(3,size=40);days=np.arange(40)//8;policy=fit_policy(p,y,days)
    for c in [0,2]:
        candidates=[]
        for threshold in np.unique(p[:,c]):
            active=(route(p)==c)&(p[:,c]>=threshold)
            n=int(active.sum());tp=int((active&(y==c)).sum())
            if n>=5 and len(np.unique(days[active]))>=2:candidates.append((wilson(tp,n),tp/n,n))
        rule=policy['rules'][str(c)]
        assert rule['enabled']==bool(candidates)
        if candidates:assert (rule['lower_score'],rule['precision'],rule['signals'])==max(candidates)
print('100 tied-score policy fixtures match independent exhaustive search')
