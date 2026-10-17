from pathlib import Path
import json,hashlib,time,sys
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from core import metadata,block
BASE=Path('/workspace/xauusd');OUT=BASE/'reports/m1_entry_evaluator_20261015';SRC=BASE/'reports/intermarket_lightgbm_20261015/samples/XAUUSD_41.parquet'
OUT.mkdir(exist_ok=True);(OUT/'points').mkdir(exist_ok=True);started=time.monotonic()
df=pq.read_table(SRC).to_pandas();ts=pd.DatetimeIndex(df.timestamp).as_unit('s').asi8;ohlc=df[['open','high','low','close']].to_numpy();del df
assert np.isfinite(ohlc).all() and (ohlc>0).all()
assert np.all(ohlc[:,1]>=np.maximum(ohlc[:,0],ohlc[:,3])) and np.all(ohlc[:,2]<=np.minimum(ohlc[:,0],ohlc[:,3]))
length,past,scale=metadata(ts,ohlc[:,3]);years=pd.to_datetime(ts,unit='s',utc=True).year.to_numpy()
compact=[];files=[]
for year in np.unique(years):
 idx=np.flatnonzero(years==year);path=OUT/'points'/f'{year}.parquet';writer=None
 try:
  for start in range(int(idx[0]),int(idx[-1])+1,25000):
   stop=min(start+25000,int(idx[-1])+1);b=block(ts,ohlc,length,past,scale,start,stop)
   table=pa.Table.from_pandas(b,preserve_index=False)
   if writer is None:writer=pq.ParquetWriter(path,table.schema,compression='zstd',compression_level=4)
   writer.write_table(table)
   keep=['timestamp','available_future_minutes','full_120_intraday','past_120_complete','past_scale_dollars']
   for side in ['long','short']:keep +=[side+'_'+x for x in ['mean_pnl','median_pnl','max_adverse','max_favourable','profitable_close_fraction','pnl_120m']]
   compact.append(b[keep])
 finally:
  if writer:writer.close()
 files.append({'year':int(year),'rows':len(idx),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 status={'year_complete':int(year),'rows_complete':int(idx[-1])+1,'seconds':round(time.monotonic()-started,1)}
 (OUT/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
small=pd.concat(compact,ignore_index=True);small.to_parquet(OUT/'analysis_inputs.parquet',compression='zstd',index=False)
manifest={'source':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'rows':len(ts),'range_utc':[str(pd.Timestamp(ts[0],unit='s',tz='UTC')),str(pd.Timestamp(ts[-1],unit='s',tz='UTC'))], 'full_120_intraday_rows':int((length==120).sum()),'past_120_complete_rows':int(past.sum()),'full_future_and_past_rows':int(((length==120)&past&np.isfinite(scale)).sum()),'point_files':files,'source_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'seconds':time.monotonic()-started,'gpu_used':False}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
