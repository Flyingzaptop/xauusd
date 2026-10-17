"""Only deterministic stage-one reconstruction, with an exact zero optimum."""
import numpy as np
import torch
from torch.nn import functional as F

def displacement(angles):return torch.tan(angles.float()*(np.pi/2)).cumsum(1)
def values(p,y):
    angle=F.mse_loss(p,y);path=F.smooth_l1_loss(displacement(p),displacement(y))
    return dict(total=path+.1*angle,path=path,angle_mse=angle)
