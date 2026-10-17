"""Causality checks: (1) model outputs at fine step i do not depend on inputs at steps > i;
(2) batch features for anchor g are identical when data after second g is corrupted."""
import sys,torch,numpy as np
from model import SeqModel
from data import Market
torch.manual_seed(0)
B,Lf,Lc,K=2,40,10,8
m=SeqModel(d=32,depth=4,heads=4,backend='attn',Lc=Lc,Lf=Lf).eval()
b=dict(ev=torch.randn(B,2,Lf,K,6),mask=torch.rand(B,2,Lf,K)>.3,ssum=torch.randn(B,2,Lf,6),msum=torch.randn(B,2,Lc,6))
with torch.no_grad():
    p0,_=m(b);b2={k:v.clone() for k,v in b.items()};i=25
    b2['ev'][:,:,i:]+=5;b2['ssum'][:,:,i:]-=3
    p1,_=m(b2)
d=(p0-p1).abs().amax((0,2,3));assert d[:i].max()<1e-5 and d[i:].max()>1e-3,d;print('model causal ok')
if len(sys.argv)>2:
    mk=Market([sys.argv[1]],[sys.argv[2]]);g=torch.tensor([30000,50000])
    x=mk.batch(g,120,30,32)
    for t in mk.t:                                           # corrupt everything after the anchors' last closed second
        cut=int(t['off'][30001]);t['ev'][cut:]=7;t['ssum'][30001:]=9;t['msum'][30000//60:]=9
    y=mk.batch(torch.tensor([30000]),120,30,32)
    for k in ['ev','mask','ssum','msum']:assert torch.equal(x[k][:1],y[k]),k
    print('batch causal ok')
