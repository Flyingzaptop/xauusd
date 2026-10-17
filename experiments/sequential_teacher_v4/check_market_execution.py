"""CPU quote-availability smoke test. No learned policy, no performance claim."""
from pathlib import Path
import json
import numpy as np,pandas as pd,pyarrow.parquet as pq
from environment import Episode,Execution,TradingEnv,UnpriceableExit
P=Path('/workspace/xauusd/reports/sequential_teacher_v4_20261016')
windows=[]
for date in ['2022-02-28','2022-03-01','2022-03-02','2022-03-03','2022-03-04','2022-03-07','2022-03-08','2022-03-10','2022-03-11']:
    windows.append({'date':date,'open':'08:00','close':'18:00','split':'train' if date<='2022-03-04' else 'dev' if date<='2022-03-08' else 'test'})
config={'windows':windows,'meaning':'Predeclared UTC strategy windows for data/execution diagnostics; not a broker calendar or final test protocol.'}
(P/'pilot_windows.json').write_text(json.dumps(config,indent=2))
f=Path('/workspace/xauusd/reports/intermarket_lightgbm_20261015/samples/XAUUSD_41.parquet');df=pq.read_table(f,columns=['timestamp','open']).to_pandas().sort_values('timestamp');ts=pd.DatetimeIndex(df.timestamp).as_unit('s').asi8;prices=df.open.to_numpy();out=[]
for w in windows:
    start=int(pd.Timestamp(w['date']+'T08:00',tz='UTC').timestamp());grid=np.arange(start,start+600*60+60,60);idx=np.searchsorted(ts,grid);valid=idx<len(ts);safe=idx.clip(max=len(ts)-1);valid&=ts[safe]==grid;quotes=np.where(valid,prices[safe],np.nan)
    ep=Episode(grid,quotes,np.zeros((len(grid),8,1),np.float32),int(grid[-1]),w['date']);r={'date':w['date'],'missing_quotes':int((~valid).sum()),'policies':{}}
    for name,action in [('always_long',1),('always_short',2),('wait',0)]:
        env=TradingEnv(ep,Execution())
        try:
            while not env.done:env.step(action if env.mask()[action] else 0)
            durations=[t['duration_minutes'] for t in env.trades];r['policies'][name]={'scorable':True,'trades':len(durations),'minimum_duration':min(durations,default=None),'maximum_duration':max(durations,default=None),'all_exits_within_window':all(t['exit_time']<=ep.scheduled_close for t in env.trades)}
        except UnpriceableExit as e:r['policies'][name]={'scorable':False,'error':str(e)}
    out.append(r)
(P/'market_execution_smoke.json').write_text(json.dumps({'complete':True,'not_a_profitability_test':True,'results':out},indent=2));print(json.dumps(out))
