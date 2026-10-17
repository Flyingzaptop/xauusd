import unittest,tempfile,json
from pathlib import Path
import numpy as np,torch
from environment import Execution
from synthetic import make_episode,FEATURES
from model import Actor,Teacher
from learning import rollout,update
from real_data import load_episodes

class Learning(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(80)
    def test_actor_update_finite_and_real(self):
        a=Actor(FEATURES);before=a.policy.weight.detach().clone();eps=[make_episode('direction',j,length=8) for j in range(4)]
        paths,_,_=rollout(a,eps,Execution(2,.05),4);log=update(a,torch.optim.Adam(a.parameters(),lr=.001),paths,4)
        self.assertFalse(torch.equal(before,a.policy.weight));self.assertTrue(all(np.isfinite(v) for v in log.values()))
    def test_teacher_rng_does_not_change_actor_actions(self):
        a=Actor(FEATURES);t=Teacher(FEATURES,future_steps=2);eps=[make_episode('direction',j,length=8) for j in range(3)]
        p,_,_=rollout(a,eps,Execution(2,.05),4);q,_,credit=rollout(a,eps,Execution(2,.05),4,teacher=t,track_teacher=True)
        self.assertEqual([[r['action'] for r in z] for z in p],[[r['action'] for r in z] for z in q])
        (-torch.stack(credit).sum()).backward();self.assertTrue(any(v.grad is not None and float(v.grad.abs().sum())>0 for v in t.parameters()))
    def test_rewards_discounted_at_execution(self):
        a=Actor(FEATURES)
        with torch.no_grad():a.policy.weight.zero_();a.policy.bias.copy_(torch.tensor([-100.,100.,-100.]))
        p,m,_=rollout(a,[make_episode('direction',2,length=2)],Execution(2,0,.9),4)
        self.assertAlmostEqual(p[0][0]['reward'],m[0]['net']/10*.9)
    def test_causal_real_panel_window(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);grid=np.arange(12)*60;raw=np.arange(12*2*2,dtype=float).reshape(12,2,2);np.save(p/'timestamps.npy',grid);np.save(p/'opens.npy',100+np.arange(12));np.save(p/'features.npy',raw)
            meta={'gold_index':0,'windows_bounds':[[480,600]],'config':{'windows':[{'date':'test','split':'train'}]}}
            (p/'panel.json').write_text(json.dumps(meta));e,_=load_episodes(p);obs=e['train'][0].observations
            self.assertEqual(obs.shape,(3,8,8));np.testing.assert_allclose(obs[0,-1,:2],np.arcsinh(raw[7,0]),rtol=1e-6)
            raw[8:]=1e6;np.save(p/'features.npy',raw);changed,_=load_episodes(p);np.testing.assert_array_equal(obs[0],changed['train'][0].observations[0])
            ablated,_=load_episodes(p,True);self.assertTrue((ablated['train'][0].observations[:,:,4:]==0).all())
if __name__=='__main__':unittest.main()
