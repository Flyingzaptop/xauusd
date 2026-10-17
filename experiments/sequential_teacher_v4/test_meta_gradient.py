import unittest
import numpy as np,torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from environment import Episode,Execution,backward_oracle
from model import Teacher
from synthetic import FEATURES
from meta_gradient import adapted,expected_return
class MetaGradient(unittest.TestCase):
    def test_expected_return_matches_one_hot_oracle(self):
        price=np.array([100,101,103,99,104.]);ts=np.arange(5)*60;e=Episode(ts,price,np.zeros((5,8,6),np.float32),240)
        v,a=backward_oracle(e,Execution(4,.05));p=torch.nn.functional.one_hot(torch.tensor(a[:-1]),3).float()[None]
        delta=torch.tensor(price[-1]-price[:-1]);reward=torch.stack([torch.zeros(4),delta-.05,-delta-.05],-1)[None]
        self.assertAlmostEqual(float(expected_return(p,reward)),v[0],places=5)
    def test_meta_gradient_finite_difference(self):
        torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(571);t=Teacher(FEATURES,width=16,depth=1,future_steps=4)
        with sdpa_kernel(SDPBackend.MATH):
            out=adapted(t,11,steps=2,track=True);g=torch.autograd.grad(out,t.head.bias)[0][0].item();original=t.head.bias.detach().clone();eps=.002
            with torch.no_grad():t.head.bias[0]+=eps
            plus=float(adapted(t,11,steps=2).detach())
            with torch.no_grad():t.head.bias.copy_(original);t.head.bias[0]-=eps
            minus=float(adapted(t,11,steps=2).detach());numerical=(plus-minus)/(2*eps)
        self.assertTrue(np.isfinite(g));self.assertGreater(abs(g),1e-6);self.assertAlmostEqual(g,numerical,delta=max(.003,abs(g)*.05))
if __name__=='__main__':unittest.main()
