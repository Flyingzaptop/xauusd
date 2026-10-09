import importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)

spec=importlib.util.spec_from_file_location('angular_tcn',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def candles(n=500):
    rng=np.random.default_rng(17);cl=np.cumsum(rng.normal(0,.0002,n));op=np.r_[0.,cl[:-1]]
    return np.column_stack([np.maximum(op,cl)+.0001,op,cl,np.minimum(op,cl)-.0001])

def test_roundtrip_and_order():
    p=candles();a=m.encode_next(p[1:],p[:-1],.0001)
    assert np.max(abs(m.decode_log(a,p[:-1],.0001)-p[1:]))<1e-12
    f=m.AngularHistory(p,.0001,.5,device='cpu')
    assert f.x([250]).shape==(1,36,128)
    np.testing.assert_allclose(f.y([250]).numpy()[0],a[250],atol=1e-7)
    np.testing.assert_allclose(f.x([250])[0,:4,-1].numpy(),a[249],atol=1e-7)

def test_future_and_price_level_invariance():
    p=candles();a=m.AngularHistory(p,.0001,.5,device='cpu').x([250])
    changed=p.copy();changed[251:]+=.2
    assert torch.equal(a,m.AngularHistory(changed,.0001,.5,device='cpu').x([250]))
    b=m.AngularHistory(p+np.log(4),.0001,.5,device='cpu').x([250])
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)

def test_temporal_causality_not_trivial_zero_head():
    torch.manual_seed(5);net=m.AngularTCN().eval()
    torch.nn.init.normal_(net.head.weight,std=.1)
    x=torch.randn(2,36,128);y=net.sequence(x)
    changed=x.clone();changed[:,:,80:]+=torch.randn_like(changed[:,:,80:])*10
    yy=net.sequence(changed)
    assert torch.allclose(y[:,:,:80],yy[:,:,:80],atol=1e-6,rtol=1e-6)
    assert not torch.allclose(y[:,:,80:],yy[:,:,80:])

def test_continuity_boundaries_and_support():
    p=np.exp(candles());ts=np.arange(len(p))*60;ts[260:]+=60
    ix,good=m.eligible(ts,p)
    assert good.all() and len(ix)>0
    assert not np.any((ix-182<260)&(ix+1>=260))
    for j in ix:assert np.all(np.diff(ts[j-182:j+2])==60)

def test_model_bounds_gradients_and_receptive_field():
    torch.manual_seed(8);net=m.AngularTCN().eval();torch.nn.init.normal_(net.head.weight,std=.1)
    x=torch.randn(2,36,128,requires_grad=True);y=net(x)
    assert y.shape==(2,4) and torch.isfinite(y).all() and (y.abs()<1).all()
    y.square().mean().backward()
    assert x.grad[:,:,0].abs().sum()>0
    assert net.projection.weight.grad.abs().sum()>0

def test_flat_baseline_uses_previous_close():
    p=candles();const=dict(mean_angle=np.zeros(4),median_offsets=np.zeros(4));ix=np.array([200,300])
    a=m.baseline_angles(p,ix,.0001,const)['flat_at_close']
    restored=m.decode_log(a,p[ix],.0001)
    np.testing.assert_allclose(restored,np.broadcast_to(p[ix,2:3],(2,4)),atol=1e-12)
