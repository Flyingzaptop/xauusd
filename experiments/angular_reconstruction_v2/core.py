"""Differentiable candle geometry, UTC aggregation and controlled decoders."""
import os,importlib.util
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/angular_reconstruction_v2_20261008'
spec=importlib.util.spec_from_file_location('infill_reference',ROOT/'experiments/angular_infill/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
DEVICE=old.DEVICE;COLS=[0,2,3];MAX=40
CHECKPOINT=ROOT/'reports/angular_infill_512_20261008/checkpoints/epoch_12.pt'

class ResidualInfiller(old.Infiller):
    def forward(self,x):
        left=self.left(x);left=F.pad(left,(1,0))[:,:,:x.shape[-1]]
        right=self.right(x.flip(-1)).flip(-1)
        return self.head(torch.cat([left,right],1))[:,:,old.LEFT:old.LEFT+MAX].transpose(1,2)

def build(kind,checkpoint,diagnostic=False):
    torch.manual_seed(1729)
    net=(ResidualInfiller() if kind=='C' else old.Infiller()).to(DEVICE)
    net.load_state_dict(checkpoint['model'])
    if kind=='C':nn.init.zeros_(net.head.weight);nn.init.zeros_(net.head.bias)
    if diagnostic:
        for layer in net.modules():
            if isinstance(layer,nn.Dropout):layer.p=0.
    return net

def batch(data,lp,ts,anchors,lengths):
    ix=anchors[:,None]+np.arange(1,MAX+1);center=lp[anchors,2]
    target,mask=data.y(anchors,lengths)
    return dict(x=data.x(anchors,lengths),angles=target.double(),mask=mask,
        left=torch.as_tensor(lp[anchors]-center[:,None],device=DEVICE,dtype=torch.float64),
        right=torch.as_tensor(lp[anchors+lengths+1]-center[:,None],device=DEVICE,dtype=torch.float64),
        truth=torch.as_tensor(lp[ix]-center[:,None,None],device=DEVICE,dtype=torch.float64),
        lengths=torch.as_tensor(lengths,device=DEVICE),
        minutes=torch.as_tensor(ts[anchors+1]//60,device=DEVICE),s0=data.s0)

def assemble(hcl,left):
    op=torch.cat([left[:,None,2],hcl[:,:-1,1]],dim=1)
    return torch.stack([hcl[:,:,0],op,hcl[:,:,1],hcl[:,:,2]],dim=-1)

def decode_angles(p,left,s0):
    increments=s0*torch.tan(p.double().clamp(-1+1e-6,1-1e-6)*(np.pi/2))
    return assemble(left[:,None,COLS]+torch.cumsum(increments,dim=1),left)

def encode_path(path,left,s0):
    prev=torch.cat([left[:,None,:],path[:,:-1]],1)
    return (2/np.pi)*torch.atan((path[:,:,COLS]-prev[:,:,COLS])/s0)

def bridge(b,right_open=False):
    h=torch.arange(1,MAX+1,device=b['left'].device,dtype=torch.float64)[None,:,None]
    hcl=b['left'][:,None,COLS]+(b['right'][:,None,COLS]-b['left'][:,None,COLS])*h/(b['lengths'][:,None,None]+1)
    if right_open:
        close=b['left'][:,None,2]+(b['right'][:,None,1]-b['left'][:,None,2])*h[:,:,0]/b['lengths'][:,None]
        hcl=torch.stack([hcl[:,:,0],close,hcl[:,:,2]],-1)
    return assemble(hcl,b['left'])

def output(net,b,kind):
    raw=net(b['x'])
    if kind=='C':
        base=bridge(b)
        hcl=base[:,:,COLS]+raw.double()*b['s0']*torch.sqrt(b['lengths'].double())[:,None,None]
        path=assemble(hcl,b['left']);angles=encode_path(path,b['left'],b['s0'])
    else:
        angles=raw.double();path=decode_angles(angles,b['left'],b['s0'])
    return angles,path

def masked_mean(values,mask):
    while mask.ndim<values.ndim:mask=mask.unsqueeze(-1)
    mask=mask.expand_as(values)
    return values.masked_fill(~mask,0).sum()/mask.sum().clamp_min(1)

def candle_shape(p):
    high,op,close,low=p.unbind(-1)
    return torch.stack([close-op,high-low,high-torch.maximum(op,close),torch.minimum(op,close)-low],-1)

def aggregate(p,lengths,minutes,k):
    groups=MAX//k;offset=torch.remainder(-minutes,k)
    starts=offset[:,None]+torch.arange(groups,device=p.device)[None,:]*k
    ix=starts[:,:,None]+torch.arange(k,device=p.device)[None,None,:]
    block=p[torch.arange(len(p),device=p.device)[:,None,None],ix.clamp_max(MAX-1)]
    result=torch.stack([block[:,:,:,0].amax(2),block[:,:,0,1],block[:,:,-1,2],block[:,:,:,3].amin(2)],-1)
    return result,starts+k<=lengths[:,None]

def losses(angles,path,b):
    s0=b['s0'];mask=b['mask'];truth=b['truth']
    angle=masked_mean((angles-b['angles'])**2,mask)
    price=masked_mean(F.smooth_l1_loss((path[:,:,COLS]-truth[:,:,COLS])/s0,torch.zeros_like(path[:,:,COLS]),reduction='none',beta=1.),mask)
    shape=masked_mean(F.smooth_l1_loss((candle_shape(path)-candle_shape(truth))/s0,torch.zeros_like(path),reduction='none',beta=1.),mask)
    multi=angle*0;present=angle*0
    for k in [3,5,10]:
        pa,valid=aggregate(path,b['lengths'],b['minutes'],k);ta,_=aggregate(truth,b['lengths'],b['minutes'],k)
        term=F.smooth_l1_loss((pa[:,:,COLS]-ta[:,:,COLS])/s0,torch.zeros_like(pa[:,:,COLS]),reduction='none',beta=1.)
        active=(valid.sum()>0).to(angle.dtype);multi=multi+masked_mean(term,valid)*active;present=present+active
    multi=multi/present.clamp_min(1)
    invalid=masked_mean((F.relu(torch.maximum(path[:,:,1],path[:,:,2])-path[:,:,0])+F.relu(path[:,:,3]-torch.minimum(path[:,:,1],path[:,:,2])))/s0,mask)
    total=angle+.25*price+.10*shape+.10*multi+.05*invalid
    return dict(total=total,angle=angle,path=price,shape=shape,multiscale=multi,invalid=invalid)

@torch.no_grad()
def diagnostic_scores(angles,path,b):
    mask=b['mask'];err=path-b['truth']
    prev_pred=torch.cat([b['left'][:,None,2],path[:,:-1,2]],1)
    prev_true=torch.cat([b['left'][:,None,2],b['truth'][:,:-1,2]],1)
    pred_tv=masked_mean(abs(path[:,:,2]-prev_pred),mask)
    true_tv=masked_mean(abs(b['truth'][:,:,2]-prev_true),mask)
    return dict(angle_rmse_degrees=float(torch.sqrt(masked_mean((angles-b['angles'])**2,mask))*90),
        price_mae_bps=float(masked_mean(abs(err[:,:,COLS]),mask)*10000),
        price_mse_bps2=float(masked_mean(err[:,:,COLS]**2,mask)*1e8),
        close_total_variation_ratio=float(pred_tv/true_tv.clamp_min(1e-12)))
