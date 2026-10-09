import importlib.util
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('experiment',Path(__file__).with_name('run_experiment.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def fixture():
    rng=np.random.default_rng(5)
    close=np.cumsum(rng.normal(0,.0002,400))
    op=np.r_[0.,close[:-1]]
    return np.column_stack([op,np.maximum(op,close)+.0001,np.minimum(op,close)-.0001,close])

def test_masks_remove_hidden_raw_dependencies():
    p=fixture();ix=np.array([100,250])
    for rep in ['log','angular']:
        f=e.Features(p,.0001,.5,rep)
        xt=f.x(ix,'teacher');xs=f.x(ix,'student')
        for a in ix:p[a+1:a+21]+=.1
        changed=e.Features(p,.0001,.5,rep)
        assert torch.equal(xt,changed.x(ix,'teacher'))
        assert torch.equal(xs,changed.x(ix,'student'))
        for a in ix:p[a+21:a+41]-=.05
        changed=e.Features(p,.0001,.5,rep)
        assert torch.equal(xs,changed.x(ix,'student'))
        assert not torch.equal(xt,changed.x(ix,'teacher'))

def test_exact_causal_support_and_target_units():
    p=fixture();ix=np.array([100]);f=e.Features(p,.0001,.5,'log')
    x=f.x(ix,'student').reshape(1,60,80)
    assert x[:,20:].count_nonzero()==0
    assert x[0,:20,40:].min()==1
    y=f.y(ix).cpu().numpy().reshape(1,20,4)*(.0001*np.sqrt(np.arange(1,21))*10000)[None,:,None]
    assert np.max(abs(y-e.truth(p,ix)))<.0001
    # Right t+21, lag13 reaches hidden t+8: feature and mask must both vanish.
    xt=f.x(ix,'teacher').reshape(1,60,80)
    col=4+list(e.LAGS).index(13)*4
    assert xt[0,40,col:col+4].count_nonzero()==0
    assert xt[0,40,40+col:44+col].count_nonzero()==0

def test_continuity_filter_excludes_every_cross_gap_window():
    p=np.exp(fixture());ts=np.arange(len(p))*60;ts[200:]+=60
    ix,good,_=e.eligible(ts,p)
    assert good.all()
    assert not np.any((ix-74<200)&(ix+40>=200))
    assert len(ix)>0

def test_baseline_and_model_shapes():
    p=fixture();f=e.Features(p,.0001,.5,'angular');m=e.new_model(17)
    y=m(f.x(np.array([100,200]),'student'))
    assert y.shape==(2,80)
    assert torch.isfinite(y).all()
    loss=(y-f.y(np.array([100,200]))).square().mean();loss.backward()
    assert m.net[-1].weight.grad.abs().sum()>0

def test_price_level_invariance_and_shared_representation_information():
    p=fixture();ix=np.array([100,200])
    a=e.Features(p,.0001,.5,'angular')
    b=e.Features(p+np.log(4),.0001,.5,'angular')
    assert torch.allclose(a.x(ix,'teacher'),b.x(ix,'teacher'),atol=1e-6,rtol=1e-6)
    linear=e.Features(p,.0001,.5,'log').x(ix,'teacher').reshape(-1,60,80)
    angular=a.x(ix,'teacher').reshape(-1,60,80)
    assert torch.equal(linear[:,:,40:],angular[:,:,40:])
    assert torch.equal(linear[:,:,:4],angular[:,:,:4])
    assert torch.allclose(torch.atan(linear[:,:,4:40])*(2/np.pi),angular[:,:,4:40])
