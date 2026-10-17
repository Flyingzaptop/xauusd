"""Factorised temporal/market Transformer. Teacher is a stochastic reward policy."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
class MarketTransformer(nn.Module):
 def __init__(self,teacher=False,d=192,depth=6,heads=6,features=48,instruments=222,dropout=.25,checkpointing=True,patch=10):
  super().__init__();self.teacher=teacher;self.checkpointing=checkpointing;self.patch=patch;self.tokens=120//patch;assert 120%patch==0
  self.proj=nn.Linear(features*2*patch,d);self.pos=nn.Parameter(torch.randn(self.tokens,d)*.01);self.symbol=nn.Parameter(torch.randn(instruments,d)*.01)
  self.temporal=nn.ModuleList([nn.TransformerEncoderLayer(d,heads,4*d,dropout=dropout,activation='gelu',batch_first=True,norm_first=True) for _ in range(depth)])
  self.market=nn.ModuleList([nn.TransformerEncoderLayer(d,heads,4*d,dropout=dropout,activation='gelu',batch_first=True,norm_first=True) for _ in range(depth)])
  self.query=nn.Parameter(torch.randn(1,1,d)*.01);self.clock=nn.Linear(8,d);self.norm=nn.LayerNorm(d)
  if teacher:self.future=nn.Linear(8*patch,d);self.future_pos=nn.Parameter(torch.randn(self.tokens,d)*.01);self.head=nn.Linear(d,4)
  else:self.head=nn.Linear(d,3);self.value=nn.Linear(d,1)
  nn.init.normal_(self.head.weight,std=.005);nn.init.zeros_(self.head.bias)
 def blocks(self,x,blocks):
  for b in blocks:
   if self.training and self.checkpointing and torch.is_grad_enabled():x=checkpoint(b,x,use_reentrant=False,preserve_rng_state=True)
   else:x=b(x)
  return x
 def forward(self,x,clock,future=None):
  # Input B,instrument,120,96 = 48 normalised causal features plus finite masks.
  B,I,T,F=x.shape;v=self.proj(x.reshape(B*I,self.tokens,self.patch*F));v=v+self.pos
  v=self.blocks(v,self.temporal).mean(1).reshape(B,I,-1)+self.symbol
  q=self.query.expand(B,-1,-1)+self.clock(clock)[:,None];parts=[q,v]
  if self.teacher:
   assert future is not None;parts.append(self.future(future.reshape(B,self.tokens,8*self.patch))+self.future_pos)
  z=self.norm(self.blocks(torch.cat(parts,1),self.market)[:,0]);raw=self.head(z).float()
  if self.teacher:
   # Beta concentration >1 keeps continuous scores strictly inside (0,1).
   return torch.distributions.Beta(1+torch.nn.functional.softplus(raw[:,:2]),1+torch.nn.functional.softplus(raw[:,2:]))
  return torch.distributions.Categorical(logits=raw),self.value(z).squeeze(-1).float()
