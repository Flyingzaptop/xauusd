import importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('infill_model',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def sample(n=2000):
    rng=np.random.default_rng(7);cl=np.cumsum(rng.normal(0,.0002,n));op=np.r_[0.,cl[:-1]]
    return np.column_stack([np.maximum(op,cl)+.0001,op,cl,np.minimum(op,cl)-.0001,])

def test_hidden_mutation_does_not_change_any_input_feature():
    lp=sample();anchor=750
    for length in [1,5,10,20,40]:
        before=m.MaskedData(lp,.0001,.5,device='cpu').x([anchor],[length])
        changed=lp.copy();changed[anchor+1:anchor+length+1]+=.1
        after=m.MaskedData(changed,.0001,.5,device='cpu').x([anchor],[length])
        assert torch.equal(before,after)

def test_visible_chords_survive_hidden_endpoint_features_do_not():
    d=m.MaskedData(sample(),.0001,.5,device='cpu');x=d.x([750],[5])[0].T
    # First right token is rel6. lag5 starts at rel1 hidden; lag8 starts at rel-2 observed.
    right=m.LEFT+5;col5=list(m.ref.LAGS).index(5)*4;col8=list(m.ref.LAGS).index(8)*4
    assert x[right,col5:col5+4].count_nonzero()==0
    assert x[right,36+col5:40+col5].count_nonzero()==0
    assert x[right,36+col8:40+col8].min()==1
    assert x[m.LEFT:m.LEFT+5,:72].count_nonzero()==0
    assert x[m.LEFT:m.LEFT+5,72].min()==1

def test_left_only_input_independent_of_all_future():
    lp=sample();a=m.MaskedData(lp,.0001,.5,device='cpu').x([750],[5],right=False)
    lp[751:]+=.3;b=m.MaskedData(lp,.0001,.5,device='cpu').x([750],[5],right=False)
    assert torch.equal(a,b)

def test_recursive_decoder_and_open_use_no_hidden_prices():
    left=np.array([[1.01,1.,1.005,.99]])
    a=np.full((1,40,3),.2);s0=.0001;path=m.decode(a,left,s0)
    step=s0*np.tan(np.pi*.2/2)
    np.testing.assert_allclose(path[0,:,m.COLS].T,left[0,m.COLS][None,:]+np.arange(1,41)[:,None]*step)
    assert path[0,0,1]==left[0,2]
    assert np.array_equal(path[:,1:,1],path[:,:-1,2])

def test_dynamic_spans_cover_every_eligible_target_once():
    pieces=[(0,1500),(1800,3600)]
    expected=np.concatenate([np.arange(a+567,b-512) for a,b in pieces])
    for cap in [5,10,20,40]:
        anchors,lengths=m.make_spans(pieces,cap,17)
        actual=np.concatenate([np.arange(a+1,a+n+1) for a,n in zip(anchors,lengths)])
        assert np.array_equal(actual,expected) and lengths.min()>=1 and lengths.max()<=cap

def test_pretrained_initial_first_angle_matches_causal_model():
    torch.manual_seed(17);old=m.wf.TCN().eval();torch.nn.init.normal_(old.head.weight,std=.1)
    new=m.Infiller(old.state_dict()).eval();data=m.MaskedData(sample(),.0001,.5,device='cpu')
    with torch.no_grad():
        initial=new(data.x([750],[5]))[:,0]
        reference=old(data.base.x([750]))[:,:,-1]
    assert torch.allclose(initial,reference,atol=2e-6,rtol=2e-6)

def test_curriculum_gate_requires_performance_and_consecutive_checks():
    rows=[dict(gap=k,price_HCL_ratio=.75,price_Close_ratio=.7,angle_ratio=.9) for k in [1,2,3,4,5]]
    assert m.gate_check(rows,5,3,0)==(True,1,False)
    assert m.gate_check(rows,5,3,1)==(True,2,True)
    rows[-1]['price_Close_ratio']=.9
    assert m.gate_check(rows,5,10,9)==(False,0,False)

def test_forward_finite_shapes_and_hidden_loss_gradients():
    torch.manual_seed(2);net=m.Infiller();d=m.MaskedData(sample(),.0001,.5,device='cpu')
    p=net(d.x([700,800],[1,5]));y,mask=d.y([700,800],[1,5])
    assert p.shape==(2,40,3) and torch.isfinite(p).all() and mask.sum()==6
    loss=((p-y).square()*mask[:,:,None]).sum()/(mask.sum()*3);loss.backward()
    assert net.head.weight.grad[:,64:].abs().sum()>0

def test_causal_rollout_uses_only_past_and_matches_first_step():
    torch.manual_seed(9);old=m.wf.TCN().to(m.DEVICE).eval();torch.nn.init.normal_(old.head.weight,std=.01)
    lp=sample();anchors=np.array([700,800]);scale=dict(S0=.0001,H=.5)
    out=m.causal_rollout(old,lp,anchors,3,scale)
    data=m.ref.AngularHistory(lp,.0001,.5)
    with torch.no_grad():first=old(data.x(anchors))[:,:,-1].cpu().numpy()
    assert out.shape==(2,3,3) and np.isfinite(out).all()
    np.testing.assert_allclose(out[:,0],first,atol=2e-4,rtol=2e-4)
    changed=lp.copy();changed[801:]+=.1
    # Compare only the second anchor; its future mutation must have no effect.
    again=m.causal_rollout(old,changed,anchors[1:],3,scale)
    np.testing.assert_allclose(out[1:],again,atol=2e-4,rtol=2e-4)

def test_right_open_bridge_reaches_known_boundary_without_hidden_anchors():
    lp=sample();a=np.array([700,800]);n=np.array([1,5]);s0=.0001
    paths=m.bridges(lp,a,n,s0)
    decoded=m.decode(paths['right_open_bridge'],lp[a],s0)
    np.testing.assert_allclose(decoded[np.arange(2),n-1,2],lp[a+n+1,1],atol=1e-12)
