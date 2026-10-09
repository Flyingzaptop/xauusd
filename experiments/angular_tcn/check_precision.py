"""Reproduce batch-size inference discrepancy and isolate floating-point precision."""
import json,sys
import numpy as np
import pandas as pd
import torch
from run import ROOT,OUT,AngularHistory,model,SEEDS

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    d=pd.read_parquet(ROOT/'data/M1/xauusd_m1.parquet');lp=np.log(d[['high','open','close','low']].to_numpy(float))
    saved=np.load(OUT/'predictions/test.npz');n=len(saved['origins'])//2;origin=int(saved['origins'][n])
    scale=json.loads((OUT/'scale.json').read_text(encoding='utf-8'))
    full=AngularHistory(lp,scale['S0'],scale['H']);xfull=full.x([origin]);del full
    short=AngularHistory(lp[origin-182:origin+1],scale['S0'],scale['H']);x=short.x([182])
    result={'input_exactly_equal':torch.equal(xfull,x),'input_max_difference':float((xfull-x).abs().max()),'by_precision':{}}
    for label,tf32 in [('tf32',True),('ieee_fp32',False)]:
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
        single=[];batched=[]
        for seed in SEEDS:
            ck=torch.load(OUT/'models'/str(seed)/'best.pt',map_location='cpu',weights_only=False)
            net=model(seed,ck['state']).eval()
            with torch.no_grad():
                single.append(net(x).cpu().numpy()[0]);batched.append(net(x.repeat(512,1,1)).cpu().numpy()[0])
        a=np.mean(single,axis=0);b=np.mean(batched,axis=0)
        result['by_precision'][label]=dict(single_vs_batch512_max_difference=float(np.max(abs(a-b))),
            single_vs_saved_max_difference=float(np.max(abs(a-saved['ensemble'][n]))),
            batch512_vs_saved_max_difference=float(np.max(abs(b-saved['ensemble'][n]))),
            single=a.tolist(),batch512=b.tolist())
    (OUT/'precision_diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
