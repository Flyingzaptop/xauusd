"""Stage-one Transformer weights, new ordered-quantile activity heads."""
import torch
from torch import nn
from torch.nn import functional as F
from stage1_model import Model as First
class Model(First):
    def __init__(self,kind):
        super().__init__(kind);self.activity_heads=nn.ModuleList([nn.Linear(64,6),nn.Linear(64,6)])
        for p in self.readout.parameters():p.requires_grad_(False)
    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c=self.contexts(x);h=self.drop(F.gelu(self.fc1(c)));v=self.fc2(h)
            if self.kind=='hyper':
                condition=c if self.fixed_context is None else self.fixed_context.expand(len(c),-1,-1)
                a,b=self.factors(condition);v=v+(torch.bmm(a,torch.bmm(b,h.reshape(-1,128,1))).squeeze(-1)/2).reshape(len(c),8,64)
            v=self.drop(F.gelu(v));pooled=[v[:,4:].mean(1),v.mean(1)]
            raw=torch.stack([head(z) for head,z in zip(self.activity_heads,pooled)],1).float().reshape(-1,2,2,3)
        return F.softplus(raw).cumsum(-1)
    def describe_past(self,x):return First.forward(self,x)
