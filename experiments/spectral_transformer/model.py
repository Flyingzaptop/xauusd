import math,numpy as np,torch
from torch import nn
from torch.nn import functional as F
CONTEXT=128;H=[15,30,60,120];SIZES=[31,127,2048,75,75]
class Data:
 def __init__(self,path):
  from pathlib import Path
  p=Path(path);self.raw=dict(np.load(p/'targets.npz'));self.ids={k:self.raw[k] for k in ['train','val','development','test']}
  self.s=torch.tensor(np.load(p/'spectral.npy',mmap_mode='r'),device='cuda',dtype=torch.float16)
  x=np.column_stack([np.load(p/'base.npy'),np.load(p/'extra.npy')]);self.nbase=np.load(p/'base.npy',mmap_mode='r').shape[1];self.x=torch.tensor(x,device='cuda',dtype=torch.float32);del x
  self.p=torch.tensor(np.load(p/'periods.npy'),device='cuda',dtype=torch.float32);self.omega=2*math.pi/self.p;self.y=torch.tensor(self.raw['y'],device='cuda');self.event=torch.tensor(self.raw['event'],device='cuda');self.norm('train')
 def norm(self,split):
  ids=torch.tensor(self.ids[split],device='cuda');e=self.s[ids,:,0].float();self.emu=e.mean(0);self.esd=e.std(0).clamp_min(.05);del e
  x=self.x[ids];self.xmu=x.mean(0);self.xsd=x.std(0).clamp_min(.01);del x
 def batch(self,ids,kind):
  ix=torch.as_tensor(ids,device='cuda',dtype=torch.long);rows=(ix[:,None]-torch.arange(CONTEXT-1,-1,-1,device='cuda')[None,:]).flatten();aux=((self.x[rows]-self.xmu)/self.xsd).clamp(-8,8).reshape(len(ix),CONTEXT,-1)
  if kind=='price':return (None,aux[:,:,:self.nbase]),self.y[ix],self.event[ix]
  z=self.s[rows].float();e=z[:,:,0];co=z[:,:,1];si=z[:,:,2];e0=((e-self.emu)/self.esd).clamp(-8,8)
  if kind=='raw':features=torch.stack([e0,co,si,*[torch.zeros_like(e0) for _ in range(6)]],-1)
  else:
   z1=self.s[rows-1].float();z5=self.s[rows-5].float();e30=self.s[rows-30,:,0].float();dco=co*z1[:,:,1]+si*z1[:,:,2];dsi=si*z1[:,:,1]-co*z1[:,:,2];c1=torch.cos(self.omega);s1=torch.sin(self.omega);dco5=co*z5[:,:,1]+si*z5[:,:,2];dsi5=si*z5[:,:,1]-co*z5[:,:,2]
   features=torch.stack([e0,co,si,((e-z1[:,:,0])/self.esd).clamp(-8,8),((e-z5[:,:,0])/self.esd).clamp(-8,8),((e-e30)/self.esd).clamp(-8,8),dco*c1+dsi*s1,dsi*c1-dco*s1,dsi5*torch.cos(5*self.omega)-dco5*torch.sin(5*self.omega)],-1)
  if kind=='raw':aux=aux[:,:,:self.nbase]
  return (features.to(torch.bfloat16).reshape(len(ix),CONTEXT,2356,9),aux),self.y[ix],self.event[ix]
class Model(nn.Module):
 def __init__(self,kind,naux,nbase):
  super().__init__();self.kind=kind
  if kind!='price':self.spectral=nn.ModuleList([nn.Linear(n*9,64) for n in SIZES])
  self.aux=nn.Sequential(nn.Linear(naux if kind=='engineered' else nbase,64),nn.GELU(),nn.Linear(64,64));self.pre=nn.LayerNorm(64);self.patch=nn.Linear(4*64,96);self.pos=nn.Parameter(torch.randn(32,96)*.01)
  layer=nn.TransformerEncoderLayer(96,4,192,.25,activation='gelu',batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
  for l in self.encoder.layers:
   nn.init.xavier_uniform_(l.self_attn.in_proj_weight);nn.init.xavier_uniform_(l.linear1.weight);nn.init.xavier_uniform_(l.linear2.weight)
  self.out=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,96),nn.GELU(),nn.Dropout(.25),nn.Linear(96,25))
 def forward(self,x):
  z,aux=x;v=self.aux(aux)
  if z is not None:
   offset=0;sp=0
   for size,linear in zip(SIZES,self.spectral):sp=sp+linear(z[:,:,offset:offset+size,:].flatten(2));offset+=size
   v=v+sp/math.sqrt(5)
  v=self.pre(v);v=self.patch(v.reshape(len(v),32,-1))+self.pos;v=self.encoder(v);r=self.out(v[:,-1]).float();mean=r[:,:4];qraw=r[:,4:16].reshape(-1,4,3);mid=qraw[:,:,0];q=torch.stack([mid-F.softplus(qraw[:,:,1]),mid,mid+F.softplus(qraw[:,:,2])],-1);return mean,q,r[:,16:]
 def initialize(self,y,e):
  with torch.no_grad():
   layer=self.out[-1];layer.weight.normal_(std=.001);bias=layer.bias;bias[:4]=y.mean(0);quant=torch.quantile(y,torch.tensor([.1,.5,.9],device=y.device),dim=0).T;gap=torch.stack([quant[:,1]-quant[:,0],quant[:,2]-quant[:,1]],1).clamp_min(.001);inv=gap+torch.log(-torch.expm1(-gap));bias[4:16]=torch.stack([quant[:,1],inv[:,0],inv[:,1]],1).flatten();bias[16:]=torch.bincount(e,minlength=9).float().clamp_min(1).log()
def loss(pred,y,event):
 mean,q,logits=pred;er=y[:,:,None]-q;levels=torch.tensor([.1,.5,.9],device=y.device);pin=torch.maximum(levels*er,(levels-1)*er).mean();mse=F.mse_loss(mean,y);ce=F.cross_entropy(logits,event);return mse+.5*pin+.25*ce,{'mse':float(mse.detach()),'pinball':float(pin.detach()),'event_ce':float(ce.detach())}
