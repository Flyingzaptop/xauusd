"""Width-scaled version of the original TCN; unchanged depth, inputs and OHLC map."""
import torch
from torch import nn
from torch.nn import functional as F
from full_train import Generator as OriginalGenerator, CTX, LATENT, H


def parameter_count(w):return 36*w*w+216*w+32


def width_ladder():
    widths=[];previous=128
    while previous>2:
        candidates=range(2,previous)
        w=min(candidates,key=lambda w:abs(parameter_count(w)-parameter_count(previous)/2))
        widths.append(w);previous=w
    return widths


class Block(nn.Module):
    def __init__(self,w,dilation):
        super().__init__();self.dilation=dilation
        self.conv1=nn.Conv1d(w,w,3,dilation=dilation)
        self.conv2=nn.Conv1d(w,w,3,dilation=dilation)
        self.norm1=nn.LayerNorm(w);self.norm2=nn.LayerNorm(w)
    def forward(self,x):
        z=self.conv1(F.pad(x,(2*self.dilation,0)))
        z=F.silu(self.norm1(z.transpose(1,2))).transpose(1,2)
        z=self.conv2(F.pad(z,(2*self.dilation,0)))
        z=self.norm2(z.transpose(1,2)).transpose(1,2)
        return F.silu(x+z)


class Encoder(nn.Module):
    def __init__(self,w):
        super().__init__();self.w=w
        self.patch=nn.Linear(80,w);self.side=nn.Parameter(torch.zeros(2,w))
        self.blocks=nn.Sequential(*[Block(w,d) for d in [1,2,4,8,16]])
        self.norm=nn.LayerNorm(2*w)
    def forward(self,x):
        v=x.reshape(len(x),2,CTX,10)
        v=torch.stack([v[:,0],v[:,1].flip(1)],1).reshape(len(x),2,64,80)
        v=self.patch(v)+self.side[None,:,None]
        z=self.blocks(v.reshape(len(x)*2,64,self.w).transpose(1,2))[:,:,-1]
        return self.norm(z.reshape(len(x),2*self.w))


class Generator(OriginalGenerator):
    def __init__(self,w):
        nn.Module.__init__(self)
        self.encoder=Encoder(w)
        self.decoder=nn.Sequential(nn.Linear(2*w+LATENT,2*w),nn.SiLU(),nn.Linear(2*w,w),nn.SiLU(),nn.Linear(w,H*4))
        nn.init.normal_(self.decoder[-1].weight,std=.005);nn.init.zeros_(self.decoder[-1].bias)
        assert sum(p.numel() for p in self.parameters())==parameter_count(w)
