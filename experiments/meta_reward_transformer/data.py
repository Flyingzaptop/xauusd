from pathlib import Path
import json,time
import numpy as np,pandas as pd,torch
class Data:
 def __init__(self,root='/content/meta_reward',device='cuda',cache=True):
  self.root=Path(root);self.meta=json.loads((self.root/'prepared.json').read_text());self.ts=np.load(self.root/'anchors.npy');self.ohlcv=np.load(self.root/'gold_ohlcv.npy');self.clock=torch.tensor(np.load(self.root/'clock.npy').astype('float32'),device=device);self.device=device
  self.raw=np.memmap(self.root/'features.f16',mode='r',dtype='float16',shape=tuple(self.meta['shape']));self.N=len(self.ts);self.I=self.meta['instruments'];self.F=48
  # Keep train/meta/dev in GPU memory, final years stay outside training cache.
  self.cache_end=int(np.searchsorted(self.ts,int(pd.Timestamp('2025-01-01',tz='UTC').timestamp())))
  self.x=torch.from_numpy(np.array(self.raw[:,:self.cache_end],copy=True)).to(device) if cache else None
  # Every source feature already uses causal past-relative units. No full-training
  # statistics, so mutating a later observation cannot change earlier inputs.
  self.gpu_ts=torch.tensor(self.ts,device=device);self.gpu_ohlc=torch.tensor(self.ohlcv[:,:4],dtype=torch.float32,device=device)
  diff=np.r_[np.nan,np.diff(self.ohlcv[:,3])];rms=pd.Series(diff**2).rolling(119,min_periods=119).mean().shift(1).to_numpy();s=np.sqrt(rms*120);s[~np.isfinite(s)|(s<=1e-6)]=np.nan
  self.scale=torch.tensor(s,dtype=torch.float32,device=device)
  # A temporary, deterministic exit rule. No future-gap based entry filtering.
  planned=np.minimum(self.ts+7200,(self.ts//86400+1)*86400-60)
  self.exits=np.searchsorted(self.ts,planned);valid=self.exits<self.N;safe=self.exits.clip(max=self.N-1);self.pnl=(self.ohlcv[safe,0]-self.ohlcv[:,0]).astype('float32');self.pnl[~valid]=np.nan
  self.duration=(self.ts[safe]-self.ts)/60
  self.horizon_valid=valid;self.secs={'train_end':int(pd.Timestamp('2023-01-01',tz='UTC').timestamp()),'meta_end':int(pd.Timestamp('2024-01-01',tz='UTC').timestamp()),'dev_end':int(pd.Timestamp('2025-01-01',tz='UTC').timestamp())}
  # Purge 1440m feature lookback +120m future around split boundaries.
  self.splits={};bounds=[('train',self.ts[0]+1560*60,self.secs['train_end']-120*60),('meta',self.secs['train_end']+1440*60,self.secs['meta_end']-120*60),('dev',self.secs['meta_end']+1440*60,self.secs['dev_end']-120*60),('test',self.secs['dev_end']+1440*60,self.ts[-1]-120*60)]
  for name,lo,hi in bounds:self.splits[name]=np.flatnonzero((self.ts>=lo)&(self.ts<hi)&np.isfinite(self.pnl)&(self.ts[safe]<hi))
  self.reward_scale=max(float(np.nanstd(self.pnl[self.splits['train']])),.01)
 def batch(self,ids,teacher=False):
  ii=torch.as_tensor(ids,device=self.device,dtype=torch.long);t=self.gpu_ts[ii];target=t[:,None]+torch.arange(-120,0,device=self.device)[None]*60;j=torch.searchsorted(self.gpu_ts,target).clamp(max=self.N-1);valid=self.gpu_ts[j]==target
  if self.x is not None and int(j.max())<self.cache_end:v=self.x[:,j].permute(1,0,2,3)
  else:v=torch.tensor(np.array(self.raw[:,j.cpu().numpy()],copy=True),device=self.device).permute(1,0,2,3)
  finite=torch.isfinite(v)&valid[:,None,:,None];z=torch.where(finite,torch.asinh(v.float()),0)
  x=torch.cat([z,finite],-1).to(torch.bfloat16 if self.device=='cuda' else torch.float32);clock=self.clock[ii]
  if not teacher:return x,clock
  target=t[:,None]+torch.arange(120,device=self.device)[None]*60;fj=torch.searchsorted(self.gpu_ts,target).clamp(max=self.N-1);fm=self.gpu_ts[fj]==target;prices=self.gpu_ohlc[fj];entry=self.gpu_ohlc[ii,0];sc=self.scale[ii];sc=torch.where(torch.isfinite(sc),sc,torch.ones_like(sc));previous=torch.cat([self.gpu_ohlc[(ii-1).clamp(min=0)][:,None],prices[:,:-1]],1)
  delta=(prices-previous)/sc[:,None,None];cumulative=(prices[:,:,[1,2,3]]-entry[:,None,None])/sc[:,None,None];f=torch.cat([delta,cumulative,fm[:,:,None]],-1);f=torch.where(fm[:,:,None],f,0).clamp(-50,50)
  return x,clock,f.to(x.dtype)
 def eval_ids(self,split,days=4,seed=0):
  ids=self.splits[split];unique=np.unique(self.ts[ids]//86400);rng=np.random.default_rng(seed);chosen=rng.choice(unique,min(days,len(unique)),replace=False);return ids[np.isin(self.ts[ids]//86400,chosen)]
 def execute(self,ids,actions,cost=.25):
  # Policy evaluated on every observed minute when flat, one position only.
  total=0.;count=0;blocked=-1;records=[]
  for i,a in zip(ids,actions):
   if self.ts[i]<blocked or a==0:continue
   direction=1 if a==1 else -1;profit=direction*float(self.pnl[i])-cost;blocked=int(self.ts[self.exits[i]]);total+=profit;count+=1;records.append((int(self.ts[i]),blocked,direction,profit))
  days=len(np.unique(self.ts[ids]//86400));return total/max(days,1),count,records
