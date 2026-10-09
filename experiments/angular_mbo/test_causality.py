import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from config import RAW,EVENTS
from prepare import raw_windows,price_features,select_contract
from model import Model,event_features

def sample():
    a=np.zeros(3000,dtype=RAW)
    a['ts_recv']=np.arange(1,3001,dtype=np.uint64)*10**9
    a['price']=(3000+np.sin(np.arange(3000)/10))*1e9
    a['size']=3;a['action']=b'T';a['side']=b'B'
    return a

def test_strict_boundary_and_recent_event_limit():
    a=sample();cutoff=np.array([2500*10**9],np.uint64);w=raw_windows(a,cutoff)
    valid=w['ts_recv']>0
    assert w.shape==(1,EVENTS)
    assert (w['ts_recv'][valid]<cutoff[0]).all()
    assert (w['ts_recv'][valid]>=700*10**9).all()
    assert w['ts_recv'][0,-1]==2499*10**9

def test_future_events_cannot_change_either_input():
    a=sample();cutoff=np.array([2400*10**9],np.uint64)
    b=a.copy();b['price'][b['ts_recv']>=cutoff[0]]*=7;b['size'][b['ts_recv']>=cutoff[0]]+=100
    assert np.array_equal(raw_windows(a,cutoff),raw_windows(b,cutoff))
    assert np.array_equal(price_features(a,cutoff,.0001,np.array([8.])),price_features(b,cutoff,.0001,np.array([8.])))

def test_empty_window_padding_and_finite_features():
    w=raw_windows(sample(),np.array([2*10**9],np.uint64))
    f,m=event_features(w,np.array([2*10**9],np.uint64),.0001)
    assert m.sum()==1 and np.isfinite(f).all()
    assert not f[:,:,~m[0]].any()

def test_contract_choice_uses_supplied_completed_day_only():
    assert select_contract({'GCZ5':100,'GCG6':50})=='GCZ5'
    assert select_contract({}) is None

def test_fast_branch_receives_gradient_and_event_order_matters():
    torch.set_num_threads(2);torch.manual_seed(17)
    net=Model('mbo').eval()
    torch.nn.init.normal_(net.head[-1].weight,std=.1)
    x=torch.randn(2,36,512);gc=torch.randn(2,26);e=torch.randn(2,15,256);mask=torch.ones(2,256,dtype=torch.bool)
    p=net(x,gc,e,mask);q=net(x,gc,e.flip(-1),mask)
    assert not torch.allclose(p,q)
    p.square().sum().backward()
    assert net.fast_input.weight.grad.abs().sum()>0
    assert net.slow[0].weight.grad.abs().sum()>0

def test_price_features_use_trade_not_fill_and_no_current_minute():
    a=sample();cutoff=np.array([2400*10**9],np.uint64)
    b=a.copy();past=(b['ts_recv']>=2340*10**9)&(b['ts_recv']<cutoff[0])
    b['action'][past]=b'F'
    f=price_features(b,cutoff,.0001,np.array([8.]))
    assert f[0,24]==0  # Last completed minute has fills only, no T trades.
    assert f[0,23]==1
