import numpy as np
import torch
torch.set_num_threads(2)
from core import MAX,COLS,decode_angles,encode_path,bridge,assemble,losses,aggregate,output

def fixture():
    torch.manual_seed(7)
    left=torch.tensor([[.0004,-.0001,0.,-.0003]],dtype=torch.float64)
    increments=torch.randn(1,MAX,3,dtype=torch.float64)*.0001
    hcl=left[:,None,COLS]+increments.cumsum(1)
    truth=assemble(hcl,left)
    n=torch.tensor([20]);mask=torch.arange(MAX)[None,:]<n[:,None]
    return dict(left=left,right=torch.tensor([[.0012,.0007,.0009,.0005]],dtype=torch.float64),
        lengths=n,mask=mask,s0=.0001,truth=truth,angles=encode_path(truth,left,.0001),minutes=torch.tensor([1]),
        x=torch.zeros(1,73,1064))

def test_differentiable_decode_roundtrip_and_gradients():
    b=fixture();a=b['angles'].detach().clone().requires_grad_();p=decode_angles(a,b['left'],b['s0'])
    torch.testing.assert_close(p[:,:,COLS],b['truth'][:,:,COLS],rtol=1e-10,atol=1e-12)
    p.square().sum().backward();assert torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
    assert torch.equal(p[:,1:,1],p[:,:-1,2])

def test_padding_does_not_affect_objective():
    b=fixture();a=torch.zeros_like(b['angles']);p=decode_angles(a,b['left'],b['s0']);before=losses(a,p,b)
    a[:,20:]=.9;p[:,20:]=123.
    changed=losses(a,p,b)
    for k in before:torch.testing.assert_close(before[k],changed[k])

def test_utc_aggregation_keeps_only_complete_hidden_bars():
    b=fixture();b['lengths']=torch.tensor([10]);p=b['truth'];a,valid=aggregate(p,b['lengths'],b['minutes'],5)
    assert valid.sum()==1
    torch.testing.assert_close(a[:,0,0],p[:,4:9,0].max(1).values)
    torch.testing.assert_close(a[:,0,2],p[:,8,2])
    torch.testing.assert_close(a[:,0,3],p[:,4:9,3].min(1).values)

def test_zero_residual_is_exact_linear_bridge():
    class Zero(torch.nn.Module):
        def forward(self,x):return torch.zeros(len(x),40,3)
    b=fixture();angles,path=output(Zero(),b,'C')
    torch.testing.assert_close(path,bridge(b),rtol=0,atol=0)
    b['truth']=b['truth']+1000
    _,again=output(Zero(),b,'C');torch.testing.assert_close(path,again,rtol=0,atol=0)

def test_geometry_loss_gradients_are_finite():
    b=fixture();a=torch.full_like(b['angles'],.2,requires_grad=True);p=decode_angles(a,b['left'],b['s0'])
    l=losses(a,p,b);l['total'].backward()
    assert all(torch.isfinite(v) for v in l.values()) and torch.isfinite(a.grad).all()
    assert a.grad[:,:20].abs().sum()>0 and a.grad[:,20:].abs().sum()==0

def test_open_is_always_previous_predicted_close():
    b=fixture();a=torch.randn_like(b['angles'])*.1;p=decode_angles(a,b['left'],b['s0'])
    assert p[0,0,1]==b['left'][0,2]
    assert torch.equal(p[:,1:,1],p[:,:-1,2])
