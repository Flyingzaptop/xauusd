"""Look up a point in an annual output partition; UTC minute open = entry."""
import argparse,json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('timestamp',help='UTC entry minute, e.g. 2025-06-16T10:00:00Z');p.add_argument('--points',type=Path,default=Path('/workspace/xauusd/reports/m1_entry_evaluator_20261015/points'));a=p.parse_args()
t=pd.Timestamp(a.timestamp)
if t.tzinfo is None:t=t.tz_localize('UTC')
else:t=t.tz_convert('UTC')
d=pd.read_parquet(a.points/f'{t.year}.parquet',filters=[('timestamp','==',t)])
if len(d)!=1:raise SystemExit('No observed M1 bar at this timestamp')
print(d.to_json(orient='records',date_format='iso',indent=2))
