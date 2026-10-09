"""Reconstruct an explicitly hidden span from a local OHLC file and a saved checkpoint."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run as model_code

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--gap-start-row',type=int,required=True,help='Zero-based source row index of the first hidden candle')
    p.add_argument('--length',type=int,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=model_code.OUT/'reconstruction.csv');args=p.parse_args()
    if not 1<=args.length<=model_code.MAX_GAP:raise ValueError('Gap length outside configured limits')
    path=args.checkpoint
    if path is None:path=Path(json.loads((model_code.OUT/'resume.json').read_text(encoding='utf-8'))['path'])
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if args.length>ck['trained_cap']:raise ValueError(f"Checkpoint has only trained lengths1..{ck['trained_cap']}")
    if ck['code_sha256']!=model_code.ref.sha(Path(__file__).with_name('run.py')):raise ValueError('Model source differs from checkpoint version')
    source=pd.read_parquet(args.input) if args.input.suffix.lower()=='.parquet' else pd.read_csv(args.input)
    start=args.gap_start_row;left=model_code.LEFT+55;right=model_code.RIGHT
    if start<left or start+args.length+right>len(source):raise ValueError('Need567 raw bars before the gap and512 after it')
    frame=source.iloc[start-left:start+args.length+right].copy()
    times=pd.to_datetime(frame.timestamp,utc=True)
    if times.isna().any() or not np.all(np.diff(times.to_numpy(dtype='datetime64[s]').astype(np.int64))==60):raise ValueError('Window must contain consecutive one-minute timestamps including hidden rows')
    price=frame[['high','open','close','low']].to_numpy(float)
    visible=np.ones(len(price),bool);visible[left:left+args.length]=False
    if not np.isfinite(price[visible]).all() or not (price[visible]>0).all():raise ValueError('Visible prices must be finite and positive')
    # Erase the actual hidden data before feature generation; arbitrary placeholder.
    price[left:left+args.length]=price[left-1]
    lp=np.log(price);scale=ck['scale'];data=model_code.MaskedData(lp,scale['S0'],scale['H'])
    net=model_code.Infiller().to(model_code.DEVICE);net.load_state_dict(ck['model']);net.eval()
    angles=model_code.predict(net,data,np.array([left-1]),np.array([args.length]))[:,:args.length]
    logs=model_code.decode(angles,lp[left-1:left],scale['S0'])[0];predicted=np.exp(logs)
    result=pd.DataFrame(predicted,columns=['high','open','close','low'])
    result.insert(0,'timestamp',times.iloc[left:left+args.length].astype(str).to_numpy())
    for j,ch in enumerate(['H','C','L']):result[f'angle_{ch}']=angles[0,:,j]
    result['valid_ohlc']=(predicted[:,0]>=predicted.max(1))&(predicted[:,3]<=predicted.min(1))
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(result.to_string(index=False));print('Saved',args.output)

if __name__=='__main__':main()
