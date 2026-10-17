import torch
from torch.nn import functional as F

def pinball(p,y):
    q=torch.tensor([.1,.5,.9],device=p.device);e=y.unsqueeze(-1)-p
    return torch.maximum(q*e,(q-1)*e).mean()

def objective(pred,y,stage):
    if stage==1:return F.smooth_l1_loss(pred['past'],y['past'])
    if stage==2:return pinball(pred['activity'],y['activity'])
    if stage==3:return pinball(pred['coarse'],y['coarse'])+.1*pinball(pred['activity'],y['activity'])
    return pinball(pred['final'],y['final'])+.1*pinball(pred['coarse'],y['coarse'])+.1*pinball(pred['activity'],y['activity'])
