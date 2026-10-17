import unittest
import numpy as np
from environment import Episode,Execution,TradingEnv,UnpriceableExit,backward_oracle
from learning import advantages

def ep(price,missing=False):
    n=len(price);return Episode(np.arange(n)*60,np.array(price,float),np.zeros((n,8,6),np.float32),(n-1)*60)
class Mechanics(unittest.TestCase):
    def test_wait_opportunity(self):
        e=ep([100,91,101,101]);v,a=backward_oracle(e,Execution(2,0));self.assertEqual(a[0],0);self.assertEqual(v[0],10)
        env=TradingEnv(e,Execution(2,0));env.step(0);env.step(1);self.assertEqual(env.realized,10)
    def test_entry_blocks_and_cost(self):
        env=TradingEnv(ep([100,91,101,101]),Execution(2,.25));_,r,_,t=env.step(1);self.assertEqual(r,.75);self.assertEqual(env.i,2);self.assertEqual(t['minutes'],2)
    def test_long_short_cost_symmetry(self):
        vals=[]
        for a in [1,2]:
            env=TradingEnv(ep([100,99,103]),Execution(2,.25));vals.append(env.step(a)[1])
        self.assertAlmostEqual(sum(vals),-.5)
    def test_end_no_zero_duration(self):
        env=TradingEnv(ep([100,101]),Execution(120,.25));env.step(1);self.assertTrue(env.done);self.assertEqual(env.trades[0]['duration_minutes'],1)
        with self.assertRaises(RuntimeError):env.step(1)
    def test_missing_exit_not_hidden_in_mask(self):
        a=TradingEnv(ep([100,101,102]),Execution(2,0));b=TradingEnv(ep([100,101,np.nan]),Execution(2,0))
        np.testing.assert_equal(a.mask(),b.mask())
        with self.assertRaises(UnpriceableExit):b.step(1)
    def test_missing_current_quote(self):
        e=TradingEnv(ep([np.nan,101,102]),Execution(2,0));np.testing.assert_equal(e.mask(),[True,False,False]);e.step(0)
    def test_intratrade_drawdown(self):
        e=TradingEnv(ep([100,90,110]),Execution(2,0));e.step(1);self.assertEqual(e.metrics()['max_mark_to_market_drawdown'],10)
    def test_no_overnight(self):
        e=TradingEnv(ep([100,101,102]),Execution(120,0));e.step(1);self.assertEqual(e.trades[0]['exit_time'],120)
    def test_elapsed_discount(self):
        e=TradingEnv(ep([100,101,102]),Execution(2,0,.9));self.assertAlmostEqual(e.step(1)[3]['discount'],.81)
    def test_gae_terminal_and_wait(self):
        p=[[dict(reward=0.,value=0.,discount=1.,minutes=1),dict(reward=10.,value=0.,discount=1.,minutes=2)]]
        a=advantages(p,lam_per_minute=1.);self.assertEqual(a[0]['target_value'],10);self.assertEqual(a[1]['target_value'],10)
if __name__=='__main__':unittest.main()
