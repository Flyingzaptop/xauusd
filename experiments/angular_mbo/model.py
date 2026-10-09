"""Slow angular TCN + asynchronous raw-event TCN with learned attention pooling."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from config import *

def event_features(raw, cutoff, s0):
    valid=raw['ts_recv']>0
    assert np.all(raw['ts_recv'][valid] < np.broadcast_to(cutoff[:,None],valid.shape)[valid]),'Future MBO event in input'
    # All reference values are known at decision time, including the last raw event price.
    last=np.maximum(raw['price'][:,-1].astype(float),1)
    px=np.maximum(raw['price'].astype(float),1)
    age=(cutoff[:,None].astype(np.int64)-raw['ts_recv'].astype(np.int64))/1e9
    dt=np.zeros_like(age);dt[:,1:]=np.maximum(0,np.diff(raw['ts_recv'].astype(np.int64),axis=1)/1e9)
    dt[:,1:][~(valid[:,1:]&valid[:,:-1])]=0
    numeric=[np.arctan(np.log(px/last[:,None])/s0)*2/np.pi,
             np.log1p(raw['size'].astype(float))/10,np.log1p(np.maximum(age,0))/8,
             np.log1p(dt*1000)/10,np.arcsinh(raw['ts_in_delta'].astype(float)/1e6)/10]
    features=numeric+[(raw['action']==x).astype(float) for x in [b'A',b'C',b'M',b'T',b'F',b'R']]
    features += [(raw['side']==x).astype(float) for x in [b'B',b'A',b'N']]
    features += [((raw['flags']&128)!=0).astype(float)]
    out=np.stack(features,axis=1).astype(np.float32)
    out*=valid[:,None,:]
    return out,valid

class Block(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.d=dilation
        self.conv=nn.Conv1d(width,width,2,dilation=dilation)
        self.point=nn.Conv1d(width,width,1);self.norm=nn.LayerNorm(width)
        self.drop=nn.Dropout(.25)
    def forward(self,x):
        y=F.gelu(self.norm(self.conv(F.pad(x,(self.d,0))).transpose(1,2))).transpose(1,2)
        return x+self.drop(self.point(self.drop(y)))

class Model(nn.Module):
    def __init__(self,variant):
        super().__init__();assert variant in ['m1','gc','mbo'];self.variant=variant
        self.slow=nn.Sequential(nn.Conv1d(36,64,1),*[Block(64,2**i) for i in range(9)])
        extra=0
        if variant in ['gc','mbo']:
            self.gc=nn.Sequential(nn.Linear(26,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.25));extra+=32
        if variant=='mbo':
            self.fast_input=nn.Conv1d(15,32,1)
            self.fast=nn.ModuleList([Block(32,2**i) for i in range(8)])
            self.attention=nn.Conv1d(32,1,1);extra+=64
        self.head=nn.Sequential(nn.Linear(64+extra,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,3))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,gc=None,events=None,mask=None):
        parts=[self.slow(x)[:,:,-1]]
        if self.variant in ['gc','mbo']:parts.append(self.gc(gc))
        if self.variant=='mbo':
            assert mask.any(1).all()
            z=self.fast_input(events)*mask[:,None,:]
            for block in self.fast:z=block(z)*mask[:,None,:]
            weights=self.attention(z).masked_fill(~mask[:,None,:],-1e4).softmax(-1)
            parts += [(z*weights).sum(-1),z[:,:,-1]]
        return torch.tanh(self.head(torch.cat(parts,1)))*(1-1e-5)

def decode(pred,left,s0):
    return left[:,COLS]+s0*torch.tan(pred.clamp(-1+1e-5,1-1e-5)*(np.pi/2))

def loss(pred,target,left,truth,s0):
    path=decode(pred,left,s0)
    angle=(pred-target).square().mean()
    price=F.smooth_l1_loss((path-truth[:,COLS])/s0,torch.zeros_like(path))
    op=left[:,2];hi,cl,lo=path.unbind(1)
    invalid=(F.relu(torch.maximum(op,cl)-hi)+F.relu(lo-torch.minimum(op,cl))).mean()/s0
    return angle+.25*price+.05*invalid
