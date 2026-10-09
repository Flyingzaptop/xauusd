import importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('walk_model',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def net():
    torch.manual_seed(19);n=m.TCN().eval();torch.nn.init.normal_(n.head.weight,std=.1);return n

def test_dense_predictions_equal_every_independent_window():
    n=net();x=torch.randn(1,36,128+17)
    with torch.no_grad():
        dense=n(x)[:,:,127:]
        separate=torch.stack([n(x[:,:,i:i+128])[:,:,-1] for i in range(18)],dim=2)
    assert dense.shape==separate.shape
    assert torch.allclose(dense,separate,atol=2e-6,rtol=2e-6)

def test_exact_128_token_receptive_field_and_no_future():
    n=net();x=torch.randn(1,36,192)
    with torch.no_grad():
        baseline=n(x);changed=x.clone();changed[:,:,:64]+=100
        assert torch.allclose(n(changed)[:,:,-1],baseline[:,:,-1],atol=1e-6,rtol=1e-6)
        changed=x.clone();changed[:,:,140:]-=100
        assert torch.allclose(n(changed)[:,:,:140],baseline[:,:,:140],atol=1e-6,rtol=1e-6)

def test_chronological_chunks_cover_all_origins_once():
    origins=np.r_[np.arange(182,3000),np.arange(3400,3450),np.arange(5000,10000)]
    chunks=m.chunks_for(origins)
    reconstructed=np.concatenate([np.arange(start,start+n) for start,n in chunks])
    assert np.array_equal(reconstructed,origins)
    assert all(n<=2048 for _,n in chunks)
    assert np.all(np.diff(reconstructed)>0)

def test_three_outputs_regularization_and_learning_rate():
    n=m.TCN();assert n.head.out_channels==3
    drops=[v for v in n.modules() if isinstance(v,torch.nn.Dropout)]
    assert len(drops)==7 and all(v.p==.25 for v in drops)
    opt,scheduler=m.setup_optimizer(n);assert opt.param_groups[0]['lr']==1e-4
    assert opt.param_groups[0]['weight_decay']==.01
    rates=[]
    for _ in range(20):scheduler.step(1.);rates.append(opt.param_groups[0]['lr'])
    assert all(a>=b for a,b in zip(rates,rates[1:])) and rates[-1]<1e-4

def test_chunk_labels_are_next_HCL_and_loss_covers_every_position():
    lp=np.random.default_rng(7).normal(0,.01,(500,4)).cumsum(0)
    data=m.ref.AngularHistory(lp,.001,.5,device='cpu')
    x,y=m.xy(data,250,25)
    assert x.shape==(1,36,127+25) and y.shape==(25,3)
    np.testing.assert_allclose(y.numpy(),m.ref.encode_next(lp[251:276],lp[250:275],.001)[:,[0,2,3]],atol=1e-7)
    n=net();pred=n(x)[0,:,-25:].T;assert pred.shape==y.shape
    loss=(pred-y).square().mean();loss.backward();assert n.head.weight.grad.abs().sum()>0
