"""Inference on the last 183 consecutive closed XAUUSD M1 candles of a local file.

python experiments/angular_tcn/predict.py --input data/M1/xauusd_m1.parquet --output forecast.json
Columns: timestamp, high, open, close, low. No future candle is needed.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from run import AngularHistory,CHANNELS,CONTEXT,DEVICE,OUT,ROOT,SEEDS,SUPPORT,decode_log,model,predictions

def read_recent(path):
    path=Path(path)
    if path.suffix.lower()=='.parquet':
        pf=pq.ParquetFile(path);pieces=[];n=0
        for rg in range(pf.num_row_groups-1,-1,-1):
            piece=pf.read_row_group(rg,columns=['timestamp','high','open','close','low']).to_pandas()
            pieces.insert(0,piece);n+=len(piece)
            if n>=SUPPORT+1:break
        return pd.concat(pieces,ignore_index=True).tail(SUPPORT+1)
    return pd.read_csv(path).tail(SUPPORT+1)

def predict_frame(frame,checkpoints=None):
    if len(frame)<SUPPORT+1:raise ValueError('Need at least 183 consecutive CLOSED minute candles')
    frame=frame.tail(SUPPORT+1)
    ts=pd.to_datetime(frame.timestamp,utc=True)
    seconds=ts.to_numpy(dtype='datetime64[s]').astype(np.int64)
    if ts.isna().any() or not np.all(np.diff(seconds)==60):raise ValueError('Last 183 candles must be sorted, unique and one minute apart')
    prices=frame[['high','open','close','low']].to_numpy(float)
    if not np.isfinite(prices).all() or not (prices>0).all():raise ValueError('Prices must be finite and positive')
    if not ((prices[:,0]>=prices.max(1))&(prices[:,3]<=prices.min(1))).all():raise ValueError('Invalid source OHLC order')
    paths=checkpoints or [OUT/'models'/str(seed)/'best.pt' for seed in SEEDS]
    logp=np.log(prices);angle_predictions=[];config=None
    for path in paths:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        config_now=(ck['s0'],ck['H'],ck['context'],tuple(ck['lags']),tuple(ck['channels']))
        if config is not None and config_now!=config:raise ValueError('Checkpoint configurations differ')
        config=config_now
        if config[2]!=CONTEXT or list(config[4])!=CHANNELS:raise ValueError('Unsupported context/channel order')
        data=AngularHistory(logp,ck['s0'],ck['H'])
        angle_predictions.append(predictions(model(ck['seed'],ck['state']),data,np.array([len(logp)-1]))[0])
    a=np.mean(angle_predictions,axis=0);predicted=np.exp(decode_log(a,logp[-1],config[0]))
    return dict(input_last_bar_utc=str(ts.iloc[-1]),forecast_bar_utc=str(ts.iloc[-1]+pd.Timedelta(minutes=1)),
        mode='next_bar_after_supplied_closed_history',input_tokens=CONTEXT,raw_history_bars=SUPPORT+1,
        angle_normalized=dict(zip(CHANNELS,a.tolist())),angle_degrees=dict(zip(CHANNELS,(a*90).tolist())),
        ohlc=dict(zip(CHANNELS,predicted.tolist())),
        valid_ohlc=bool(predicted[0]>=predicted.max() and predicted[3]<=predicted.min()),
        close_change_dollars=float(predicted[2]-prices[-1,2]),
        seed_angle_standard_deviation=dict(zip(CHANNELS,np.std(angle_predictions,axis=0).tolist())),
        uncertainty_note='Seed disagreement is not a calibrated prediction interval. Prices are not projected or sorted.')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/M1/xauusd_m1.parquet')
    parser.add_argument('--output',type=Path,default=OUT/'forecast.json')
    args=parser.parse_args();result=predict_frame(read_recent(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
