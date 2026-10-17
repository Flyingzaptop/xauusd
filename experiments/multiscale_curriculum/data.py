import numpy as np
import torch
class Data:
    def __init__(self,path):
        z=np.load(path);self.raw=z;self.start=int(z['start_minute']);self.ids={s:z[s+'_origins'] for s in ['train','val','test']}
        self.grids={step:torch.tensor(z[f'grid_{step}'],device='cuda') for step in [1,5,60]}
        self.clock={step:torch.tensor(z[f'clock_{step}'],device='cuda') for step in [1,5,60]}
        self.store={s:{k:torch.tensor(z[f'{s}_{k}'],device='cuda') for k in ['origins','scale','anchor','past','activity','coarse','final']} for s in self.ids}
    def batch(self,split,a,b):
        d={k:(v[a] if b is None else v[a:b]) for k,v in self.store[split].items()};xs=[]
        for step,length in [(1,360),(5,576),(60,336)]:
            ix=d['origins'][:,None]//step+torch.arange(-length,0,device='cuda');g=self.grids[step][ix]
            price=(g[:,:,:4]-d['anchor'][:,None,None])/d['scale'][:,None,None]
            # asinh compresses extreme moves while retaining their scale and sign.
            phase=(self.clock[step][ix]%1440).float()*(2*np.pi/1440)
            xs.append(torch.cat([price.asinh(),g[:,:,4:5],phase.sin()[:,:,None],phase.cos()[:,:,None],d['scale'].log()[:,None,None].expand(-1,length,1)],-1))
        return xs,d
