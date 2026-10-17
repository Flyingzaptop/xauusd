"""Capacity-matched static TCN and a context-generated rank-four weight update."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from classification_base import Classifier


class Model(Classifier):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.rank=4;self.fixed_context=None
        if kind=='static':
            self.head=nn.Sequential(nn.Linear(256,335),nn.SiLU(),nn.Linear(335,128),nn.SiLU(),nn.Linear(128,3))
        elif kind=='hyper':
            self.hypernet=nn.Sequential(nn.Linear(256,16),nn.SiLU(),nn.Linear(16,128*4+4*256))
            nn.init.normal_(self.hypernet[-1].weight,std=.05);nn.init.zeros_(self.hypernet[-1].bias)
        else:raise ValueError(kind)
    def context(self,x):
        z=self.blocks(self.patch(x.reshape(len(x),64,80)).transpose(1,2))
        return self.norm(torch.cat([z[:,:,-1],z.mean(-1)],-1))
    def factors(self,c):
        raw=self.hypernet(c)
        return raw[:,:512].reshape(-1,128,4),raw[:,512:].reshape(-1,4,256)
    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c=self.context(x)
            if self.kind=='static':return self.head(c).float()
            h=self.head[1](self.head[0](c))
            condition=c if self.fixed_context is None else self.fixed_context.expand(len(c),-1)
            a,b=self.factors(condition)
            delta=torch.bmm(a,torch.bmm(b,h.unsqueeze(-1))).squeeze(-1)/math.sqrt(self.rank)
            v=self.head[2](h)+delta
            return self.head[4](self.head[3](v)).float()
    @torch.no_grad()
    def dynamics(self,x):
        if self.kind!='hyper':return None
        with torch.autocast('cuda',dtype=torch.bfloat16):c=self.context(x);a,b=self.factors(c)
        dw=torch.bmm(a.float(),b.float())/math.sqrt(self.rank)
        mean=dw.mean(0);base=self.head[2].weight.float().norm()
        return dict(mean_correction_norm=float(dw.flatten(1).norm(dim=1).mean()),
                    context_variation_norm=float((dw-mean).flatten(1).norm(dim=1).mean()),
                    base_weight_norm=float(base),relative_variation=float((dw-mean).flatten(1).norm(dim=1).mean()/base))
