"""Compare explicit risk preferences; never infer a universal coefficient from data."""
import numpy as np

def metrics(equity,weekly_net):
    e=np.r_[0,np.asarray(equity,float)];w=np.asarray(weekly_net,float);dd=np.maximum.accumulate(e)-e
    return {'net':float(e[-1]),'max_drawdown':float(dd.max()),'worst_week':float(w.min()) if len(w) else 0.,'negative_weeks':int((w<0).sum()),'weekly_std':float(w.std()) if len(w) else 0.}

def sensitivity(equity,weekly_net,penalties=(0.,.1,.5,1.)):
    m=metrics(equity,weekly_net)
    return {'metrics':m,'utility_by_drawdown_penalty':{str(p):m['net']-p*m['max_drawdown'] for p in penalties},'status':'diagnostic preferences; net money is current base reward, no risk optimum claimed'}
