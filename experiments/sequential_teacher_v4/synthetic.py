"""Independent procedural sessions; known signals are available before actions."""
import numpy as np
from environment import Episode
CONTEXT=8
FEATURES=6

def make_episode(task,seed,length=48):
    rng=np.random.default_rng(seed);ts=1704067200+np.arange(length+1,dtype=np.int64)*60;feature=np.zeros((length+1,FEATURES),dtype=np.float32)
    if task in ['direction','cross_market','null']:
        sign=rng.choice([-1.,1.]);drift=sign*.35 if task!='null' else 0.
        changes=drift+rng.normal(0,.12 if task!='null' else .35,length)
        price=100+np.r_[0,np.cumsum(changes)]
        if task=='direction':feature[:,0]=sign
        if task=='cross_market':
            a=rng.choice([-1.,1.]);feature[:,1]=a;feature[:,2]=sign*a  # only joint relation determines gold drift
        if task=='null':feature[:,:3]=rng.normal(0,1,(length+1,3))
    elif task=='wait':
        # Two decisions: entering at t=0 earns only 1, waiting then LONG earns 10.
        # Signs and overall offset vary independently between sessions.
        length=3;ts=ts[:4];feature=feature[:4];sign=rng.choice([-1.,1.]);price=100+sign*np.array([0.,-9.,1.,1.])
        feature[:,0]=sign;feature[:,3]=np.arange(4)/3
    else:raise ValueError(task)
    feature[:,4]=np.arange(length+1)/length;feature[:,5]=1.
    # Exogenous signal and time are known now; no returns from the current bar.
    padded=np.concatenate([np.zeros((CONTEXT-1,FEATURES),np.float32),feature]);obs=np.stack([padded[i:i+CONTEXT] for i in range(length+1)])
    return Episode(ts,price.astype(float),obs,int(ts[-1]),f'{task}:{seed}')

def future_view(episode,index,horizon):
    # Privileged information used by reward teacher only; terminal padding masked.
    end=min(index+horizon,len(episode.opens)-1);path=episode.opens[index+1:end+1]-episode.opens[index]
    x=np.zeros((horizon,2),np.float32);x[:len(path),0]=path;x[:len(path),1]=np.isfinite(path);x[~np.isfinite(x)]=0
    return x
