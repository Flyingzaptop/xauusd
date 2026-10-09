"""Apply a validation-selected objective variant to an explicitly hidden local span."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import OUT,old,DEVICE,ResidualInfiller,output

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--gap-start-row',type=int,required=True);p.add_argument('--length',type=int,required=True)
    p.add_argument('--variant',choices=['best','A','B','C'],default='best');p.add_argument('--output',type=Path,default=OUT/'reconstruction.csv');args=p.parse_args()
    if not 1<=args.length<=40:raise ValueError('Length must be1..40')
    kind=args.variant
    if kind=='best':
        selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        if len(selection)!=3:raise ValueError('The comparison is not finished; best variant is not selected yet')
        kind=min(selection,key=lambda r:r['validation_ratio_to_bridge'])['kind']
    checkpoint=torch.load(OUT/'models'/kind/'best.pt',map_location='cpu',weights_only=False)
    if checkpoint['core_sha256']!=old.ref.sha(Path(__file__).with_name('core.py')):raise ValueError('Model core differs from checkpoint version')
    frame=pd.read_parquet(args.input) if args.input.suffix.lower()=='.parquet' else pd.read_csv(args.input)
    start=args.gap_start_row;n=args.length
    if start<567 or start+n+512>len(frame):raise ValueError('Need567 raw bars before the gap and512 after it')
    window=frame.iloc[start-567:start+n+512].copy();times=pd.to_datetime(window.timestamp,utc=True)
    seconds=times.to_numpy(dtype='datetime64[s]').astype(np.int64)
    if times.isna().any() or not np.all(np.diff(seconds)==60):raise ValueError('Consecutive minute timestamps required, including hidden rows')
    prices=window[['high','open','close','low']].to_numpy(dtype=float,copy=True);visible=np.ones(len(prices),bool);visible[567:567+n]=False
    if not np.isfinite(prices[visible]).all() or not (prices[visible]>0).all():raise ValueError('Invalid visible prices')
    prices[567:567+n]=prices[566]
    lp=np.log(prices);center=lp[566,2];scale=json.loads((OUT/'scale.json').read_text())
    data=old.MaskedData(lp,scale['S0'],scale['H']);net=(ResidualInfiller() if kind=='C' else old.Infiller()).to(DEVICE)
    net.load_state_dict(checkpoint['model']);net.eval()
    b=dict(x=data.x([566],[n]),left=torch.as_tensor(lp[566:567]-center,device=DEVICE),
        right=torch.as_tensor(lp[567+n:568+n]-center,device=DEVICE),lengths=torch.tensor([n],device=DEVICE),s0=scale['S0'])
    with torch.no_grad():angles,path=output(net,b,kind)
    predicted=np.exp(path.cpu().numpy()[0,:n]+center)
    result=pd.DataFrame(predicted,columns=['high','open','close','low']);result.insert(0,'timestamp',times.iloc[567:567+n].astype(str).to_numpy())
    for j,ch in enumerate(['H','C','L']):result[f'angle_{ch}']=angles.cpu().numpy()[0,:n,j]
    result['valid_ohlc']=(predicted[:,0]>=predicted.max(1))&(predicted[:,3]<=predicted.min(1))
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print('Variant',kind,'fine-tuning epoch',checkpoint['epoch']);print(result.to_string(index=False));print('Saved',args.output)

if __name__=='__main__':main()
