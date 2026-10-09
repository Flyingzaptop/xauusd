"""Predict next H/C/L angles; reconstruct Open as last Close, using closed M1 bars."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from run import ROOT,OUT,TCN,ref,DEVICE,TARGET

def recent(path):
    path=Path(path)
    if path.suffix.lower()=='.parquet':
        file=pq.ParquetFile(path);parts=[];count=0
        for index in range(file.num_row_groups-1,-1,-1):
            p=file.read_row_group(index,columns=['timestamp','high','open','close','low']).to_pandas();parts.insert(0,p);count+=len(p)
            if count>=183:break
        return pd.concat(parts).tail(183)
    return pd.read_csv(path).tail(183)

def predict_frame(frame,checkpoint=OUT/'checkpoints/epoch_30.pt'):
    if len(frame)<183:raise ValueError('At least183 consecutive CLOSED M1 candles are required')
    frame=frame.tail(183);times=pd.to_datetime(frame.timestamp,utc=True)
    seconds=times.to_numpy(dtype='datetime64[s]').astype(np.int64)
    if times.isna().any() or not np.all(np.diff(seconds)==60):raise ValueError('History must be consecutive, sorted and unique at one-minute spacing')
    price=frame[['high','open','close','low']].to_numpy(float)
    if not np.isfinite(price).all() or not (price>0).all():raise ValueError('Invalid prices')
    if not ((price[:,0]>=price.max(1))&(price[:,3]<=price.min(1))).all():raise ValueError('Invalid OHLC order')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['context']!=128 or ck['targets']!=['H','C','L']:raise ValueError('Unsupported checkpoint configuration')
    lp=np.log(price);scale=ck['scale'];data=ref.AngularHistory(lp,scale['S0'],scale['H'])
    net=TCN().to(DEVICE);net.load_state_dict(ck['model']);net.eval()
    with torch.no_grad():three=net(data.x([182]))[0,:,-1].cpu().numpy()
    a=np.zeros(4);a[TARGET]=three;a[1]=ref.encode_next(lp[-1,2],lp[-1,1],scale['S0'])
    output=np.exp(ref.decode_log(a,lp[-1],scale['S0']));output[1]=price[-1,2]
    return dict(last_closed_bar_utc=str(times.iloc[-1]),forecast_bar_utc=str(times.iloc[-1]+pd.Timedelta(minutes=1)),
        checkpoint_epoch=ck['epoch'],neural_angles_normalized=dict(zip(['H','C','L'],three.tolist())),
        neural_angles_degrees=dict(zip(['H','C','L'],(three*90).tolist())),
        calculated_open_angle_degrees=float(a[1]*90),ohlc=dict(zip(['H','O','C','L'],output.tolist())),
        valid_ohlc=bool(output[0]>=output.max() and output[3]<=output.min()),
        open_source='Last observed Close; not a neural output',history_source='Supplied historical closed bars; not a live quote')

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=ROOT/'data/M1/xauusd_m1.parquet')
    p.add_argument('--output',type=Path,default=OUT/'forecast.json');p.add_argument('--checkpoint',type=Path,default=OUT/'checkpoints/epoch_30.pt');args=p.parse_args()
    result=predict_frame(recent(args.input),args.checkpoint);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(result,indent=2,ensure_ascii=False))
