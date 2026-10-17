import numpy as np
import torch
class Data:
    def __init__(self,path):
        self.raw=np.load(path);self.ids={s:self.raw[s+'_origins'] for s in ['train','val','test']}
        self.features={k:torch.tensor(self.raw[f'features_{k}'],device='cuda') for k in [1,5,60]}
        self.origins={s:torch.tensor(i,device='cuda') for s,i in self.ids.items()}
        self.y={s:torch.tensor(self.raw[s+'_angles'],device='cuda') for s in self.ids}
    def batch(self,split,a,b=None):
        i=self.origins[split][a] if b is None else self.origins[split][a:b];xs=[]
        for step,length in [(1,360),(5,576),(60,336)]:
            ix=i[:,None]//step+torch.arange(-length,0,device='cuda');xs.append(self.features[step][ix])
        y=self.y[split][a] if b is None else self.y[split][a:b];return xs,y
