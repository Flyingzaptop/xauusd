"""Three temporal scales, six causal residual blocks per scale, dropout 25%."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.pad=2*dilation
        self.c1=nn.Conv1d(width,width,3,dilation=dilation);self.c2=nn.Conv1d(width,width,3,dilation=dilation)
        self.n1=nn.LayerNorm(width);self.n2=nn.LayerNorm(width);self.drop=nn.Dropout(.25)
    def forward(self,x):
        z=self.c1(F.pad(x,(self.pad,0)));z=self.drop(F.silu(self.n1(z.transpose(1,2)))).transpose(1,2)
        z=self.c2(F.pad(z,(self.pad,0)));z=self.drop(F.silu(self.n2(z.transpose(1,2)))).transpose(1,2)
        return x+z
class Tower(nn.Module):
    def __init__(self,patch):
        super().__init__();self.size=patch;self.patch=nn.Linear(patch*8,64);self.blocks=nn.Sequential(*[Block(64,d) for d in [1,2,4,8,16,32]])
    def forward(self,x):
        z=self.blocks(self.patch(x.reshape(len(x),-1,self.size*8)).transpose(1,2))
        return torch.cat([z[:,:,-1],z.mean(-1)],-1)

def quantiles(raw):
    # Ordered q10, q50, q90 for each scalar target.
    return torch.stack([raw[...,0]-F.softplus(raw[...,1]),raw[...,0],raw[...,0]+F.softplus(raw[...,2])],-1)

class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.fixed_context=None
        self.towers=nn.ModuleList([Tower(p) for p in [4,6,4]])
        self.context=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,192),nn.SiLU(),nn.Dropout(.25))
        hidden=191 if kind=='static' else 128
        self.fc1=nn.Linear(192,hidden);self.fc2=nn.Linear(hidden,64);self.drop=nn.Dropout(.25)
        self.heads=nn.ModuleDict({k:nn.Linear(64,n) for k,n in [('past',12),('activity',12),('coarse',12),('final',72)]})
        if kind=='hyper':
            self.hypernet=nn.Sequential(nn.Linear(192,16),nn.SiLU(),nn.Linear(16,768))
            nn.init.normal_(self.hypernet[-1].weight,std=.02);nn.init.zeros_(self.hypernet[-1].bias)
        elif kind!='static':raise ValueError(kind)
    def features(self,x):
        return self.context(torch.cat([net(v) for net,v in zip(self.towers,x)],-1))
    def factors(self,c):
        raw=self.hypernet(c);return raw[:,:256].reshape(-1,64,4),raw[:,256:].reshape(-1,4,128)
    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c=self.features(x);h=self.drop(F.silu(self.fc1(c)));v=self.fc2(h)
            if self.kind=='hyper':
                condition=c if self.fixed_context is None else self.fixed_context.expand(len(c),-1)
                a,b=self.factors(condition);v=v+torch.bmm(a,torch.bmm(b,h.unsqueeze(-1))).squeeze(-1)/2
            v=self.drop(F.silu(v));raw={k:head(v).float() for k,head in self.heads.items()}
        final=quantiles(raw['final'].reshape(-1,8,3,3))
        # Project each quantile's low/mean/high into physical order; order-statistic
        # sorting also preserves the quantile ordering already established above.
        ordered=final.sort(dim=2).values
        final=ordered[:,:,[1,0,2],:]  # dataset order mean Close, min Low, max High
        return dict(past=raw['past'].reshape(-1,4,3),activity=quantiles(raw['activity'].reshape(-1,4,3)),coarse=quantiles(raw['coarse'].reshape(-1,4,3)),final=final)
