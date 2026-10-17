"""Shared training/evaluation semi-Markov simulator. No future-based entry masks."""
from dataclasses import dataclass
import numpy as np

WAIT,LONG,SHORT=0,1,2
class UnpriceableExit(RuntimeError):
    pass

@dataclass(frozen=True)
class Execution:
    horizon_minutes:int=120
    cost_per_trade:float=.25  # round-trip cost per one ounce, explicit approximation
    gamma_per_minute:float=1.0
    def __post_init__(self):
        if self.horizon_minutes<1 or self.cost_per_trade<0 or not 0<self.gamma_per_minute<=1:
            raise ValueError('Invalid execution settings')

@dataclass
class Episode:
    timestamps:np.ndarray
    opens:np.ndarray
    observations:np.ndarray  # causal, including only completed bars/exogenous known state
    scheduled_close:int     # declared before episode; never inferred from future data gaps
    name:str=''
    def __post_init__(self):
        if len(self.timestamps)<2 or len(self.opens)!=len(self.timestamps) or len(self.observations)!=len(self.timestamps):
            raise ValueError('Episode shape')
        if np.any(np.diff(self.timestamps)!=60):raise ValueError('Explicit minute grid required, gaps must be NaN quotes')
        if self.scheduled_close!=int(self.timestamps[-1]):raise ValueError('Episode ends at declared close')
        if np.any(~np.isfinite(self.observations)):raise ValueError('Observations require explicit finite masks')
        if np.any(self.opens[np.isfinite(self.opens)]<=0):raise ValueError('Prices must be positive')

class TradingEnv:
    def __init__(self,episode,execution=Execution()):
        self.episode=episode;self.execution=execution;self.reset()
    def reset(self):
        self.i=0;self.realized=0.;self.trades=[];self.equity=[(int(self.episode.timestamps[0]),0.)];self.transitions=[]
        return self.observation()
    @property
    def done(self):return self.i>=len(self.episode.timestamps)-1
    def observation(self):return self.episode.observations[self.i]
    def mask(self):
        # Only CURRENT quote and predeclared session close affect permission.
        allowed=not self.done and np.isfinite(self.episode.opens[self.i])
        return np.array([not self.done,allowed,allowed],dtype=bool)
    def step(self,action):
        if self.done:raise RuntimeError('Episode complete')
        if action not in [WAIT,LONG,SHORT] or not self.mask()[action]:raise ValueError('Illegal action')
        e=self.episode;start=self.i;entry_time=int(e.timestamps[start]);reward=0.
        if action==WAIT:
            end=start+1;self.equity.append((int(e.timestamps[end]),self.realized))
        else:
            target=min(entry_time+60*self.execution.horizon_minutes,e.scheduled_close)
            end=int(np.searchsorted(e.timestamps,target))
            # A missing execution quote is an unscorable episode, not a free exit,
            # a weekend extension, or a retroactive exclusion from the entry mask.
            if not np.isfinite(e.opens[end]):
                raise UnpriceableExit(f'{e.name}: no quote at scheduled exit {target}; keep and report failed episode')
            direction=1 if action==LONG else -1
            reward=direction*float(e.opens[end]-e.opens[start])-self.execution.cost_per_trade
            for j in range(start+1,end+1):
                mark=direction*float(e.opens[j]-e.opens[start])-self.execution.cost_per_trade if np.isfinite(e.opens[j]) else np.nan
                self.equity.append((int(e.timestamps[j]),self.realized+mark))
            self.realized+=reward
            self.trades.append({'entry_time':entry_time,'exit_time':int(e.timestamps[end]),'side':direction,'net':reward,'duration_minutes':end-start})
        self.i=end;duration=end-start
        tr={'start':start,'end':end,'action':int(action),'reward':reward,'minutes':duration,'discount':self.execution.gamma_per_minute**duration,'done':self.done}
        self.transitions.append(tr)
        return self.observation(),reward,self.done,tr
    def metrics(self):
        eq=np.array([v for _,v in self.equity]);finite=np.isfinite(eq)
        dd=float(np.max(np.maximum.accumulate(np.r_[0,eq[finite]])-np.r_[0,eq[finite]]))
        return {'net':self.realized,'trades':len(self.trades),'long_trades':sum(x['side']==1 for x in self.trades),
                'short_trades':sum(x['side']==-1 for x in self.trades),'exposure_minutes':sum(x['duration_minutes'] for x in self.trades),
                'max_mark_to_market_drawdown':dd,'drawdown_complete':bool(finite.all()),'missing_mark_minutes':int((~finite).sum()),'trade_records':self.trades,'equity':self.equity}

def backward_oracle(episode,execution):
    """Hindsight DP for testing mechanics only; NEVER a deployable policy or label."""
    n=len(episode.timestamps);v=np.zeros(n);actions=np.zeros(n,dtype=int)
    for i in range(n-2,-1,-1):
        values=[execution.gamma_per_minute*v[i+1],-np.inf,-np.inf]
        if np.isfinite(episode.opens[i]):
            j=min(i+execution.horizon_minutes,n-1)
            if not np.isfinite(episode.opens[j]):raise UnpriceableExit('Oracle also rejects missing exit')
            delta=episode.opens[j]-episode.opens[i]
            values[1]=execution.gamma_per_minute**(j-i-1)*(delta-execution.cost_per_trade)+execution.gamma_per_minute**(j-i)*v[j]
            values[2]=execution.gamma_per_minute**(j-i-1)*(-delta-execution.cost_per_trade)+execution.gamma_per_minute**(j-i)*v[j]
        actions[i]=int(np.argmax(values));v[i]=values[actions[i]]
    return v,actions
