"""Precision-first operating rules; only validation labels fit thresholds."""
import numpy as np
DIRECTIONS=[0,2]


def wilson(tp,n,z=1.6448536269514722):
    if not n:return 0.
    p=tp/n;return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def route(p):return np.where(p[:,0]>=p[:,2],0,2)


def fit_policy(p,y,days,min_signals=5,min_days=2):
    routed=route(p);rules={}
    for c in DIRECTIONS:
        ix=np.flatnonzero(routed==c);ix=ix[np.argsort(-p[ix,c],kind='stable')]
        best=None;seen=set();tp=0
        for j,i in enumerate(ix):
            seen.add(int(days[i]));tp+=int(y[i]==c);n=j+1
            # Include all ties at a cutoff; never artificially pick a tied subset.
            if j+1<len(ix) and p[ix[j+1],c]==p[i,c]:continue
            if n<min_signals or len(seen)<min_days:continue
            key=(wilson(tp,n),tp/n,n)
            if best is None or key>best[0]:best=(key,dict(enabled=True,threshold=float(p[i,c]),signals=n,tp=tp,precision=tp/n,lower_score=key[0],days=len(seen)))
        rules[str(c)]=best[1] if best else dict(enabled=False,threshold=None,signals=0,tp=0,precision=None,lower_score=0.,days=0)
    return dict(rules=rules,selection_score=float(np.mean([r['lower_score'] for r in rules.values()])),min_signals=min_signals,min_days=min_days)


def apply_policy(p,policy):
    routed=route(p);pred=np.ones(len(p),dtype=int)
    for c in DIRECTIONS:
        rule=policy['rules'][str(c)]
        if rule['enabled']:pred[(routed==c)&(p[:,c]>=rule['threshold'])]=c
    return pred


def metrics(pred,y):
    per={}
    for c in DIRECTIONS:
        signal=pred==c;tp=int(((y==c)&signal).sum());n=int(signal.sum());positive=int((y==c).sum())
        per[str(c)]=dict(signals=n,tp=tp,fp=n-tp,precision=tp/n if n else None,recall=tp/positive if positive else None,actual=positive)
    n=sum(v['signals'] for v in per.values());tp=sum(v['tp'] for v in per.values())
    return dict(per_direction=per,signals=n,tp=tp,fp=n-tp,precision=tp/n if n else None,recall=tp/int((y!=1).sum()),coverage=n/len(y),confusion=np.bincount(y*3+pred,minlength=9).reshape(3,3).tolist())


def fixtures():
    p=np.array([[.8,.1,.1],[.1,.1,.8],[.2,.6,.2],[.7,.2,.1],[.1,.2,.7],[.6,.3,.1]])
    y=np.array([0,2,1,0,1,1]);d=np.arange(6)//2
    pol=fit_policy(p,y,d,min_signals=2,min_days=2);pred=apply_policy(p,pol)
    for c in DIRECTIONS:
        rule=pol['rules'][str(c)]
        if rule['enabled']:assert int((pred==c).sum())==rule['signals']
    assert metrics(np.ones(6,dtype=int),y)['precision'] is None
    tied=fit_policy(np.tile([.5,.3,.2],(6,1)),y,d,min_signals=2,min_days=2)
    assert tied['rules']['0']['signals']==6
    assert not tied['rules']['2']['enabled']
    assert wilson(4,5)>wilson(1,5) and wilson(0,5)==0
    return dict(ties_handled=True,abstention_precision_is_undefined=True,validation_support_checked=True)

if __name__=='__main__':print(fixtures())
